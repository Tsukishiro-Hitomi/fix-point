"""agent/tools.py —— fixpoint 的「手」：6 个工具 + ``TOOLS`` schema + ``guarded_execute`` 分发器。

loop 每一步从模型拿到 ``tool_use`` 请求后，都在这里被真正执行，结果（**永远是一个字符串**）
再喂回模型作为下一步观察。三条硬约束（DESIGN §6 引言）：

1. **对模型友好**：返回简洁、信息密度高的文本（带行号的代码、``path:line`` 的命中、
   pass/fail 计数 + 截断后的失败详情）。
2. **绝不崩溃、绝不越界**：任何输入（幻想路径 ``../../etc/passwd``、不存在的文件、二进制、
   死循环代码）都转成 ``"错误：…"`` 字符串返回，**永不向 loop 抛异常、永不触碰工作目录之外的文件**。
3. **确定性**：同样 workspace + 同样调用，输出稳定（``run_tests`` 不引入 flaky）。

统一形态（DESIGN §6.1）
-----------------------
- 每个工具是纯函数，签名恒为 ``handler(workdir: str, **kwargs) -> str``。
- handler 第一件事就是对每个路径参数调 ``sandbox.resolve_in_workdir(workdir, path)`` 拿安全绝对路径；
  tools 层触碰文件系统**只能**经由该绝对路径。
- 返回约定（契约级，测试断言）：成功 → 人类/模型可读文本，**不加** ``"错误："`` 前缀；
  失败 → 以 ``"错误："`` 开头的字符串。
- 对外文本里的路径**一律用 workdir 相对形式**（回显模型给的相对路径），绝不泄露机器绝对路径。
- 截断预算等标量由**调用方从 config 抽取后经参数传入**，handler 内不 ``import config``。
  本模块出现的默认值（``max_read_lines=400`` / ``max_search_hits=100`` / ``max_test_output=4000`` /
  ``timeout=60``）与 ``config`` 的同名字段（DESIGN §8）对齐，作为「未注入时」的兜底。

脚手架说明
----------
本文件是脚手架：所有 **handler** 与 **guarded_execute** 仅给「签名 + 契约 docstring」，
函数体一律 ``raise NotImplementedError``；**声明式**的 ``TOOLS`` schema 按 DESIGN §6.2 写全。
"""
from __future__ import annotations

from typing import Optional

# handlers 与 guarded_execute 经此访问 sandbox.resolve_in_workdir / sandbox.PathEscape
# （DESIGN §5 是路径封闭与隔离目录的唯一 owner；tools 只消费其接口，不重复实现）。
from agent import sandbox
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# TOOLS —— 6 个 anthropic tool schema（声明式，直接传给 client.messages.create(tools=TOOLS,...)）。
# 三键结构 {"name","description","input_schema"}；input_schema 为标准 JSON Schema，
# 一律带 "additionalProperties": false；工具描述用英文（DESIGN §6.1）。
# 注意：截断预算 / timeout 等 handler 参数**不进 schema**（非模型可见，由调用方注入或走默认）。
# ---------------------------------------------------------------------------
TOOLS: list[dict] = [
    {
        "name": "list_dir",
        "description": (
            "List the entries of a directory inside the task workspace. "
            "Entries are returned in alphabetical order with directories suffixed by '/'. "
            "Noise such as __pycache__, .pytest_cache, *.pyc and .git is filtered out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list, relative to the workspace root. Defaults to '.'.",
                    "default": ".",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file inside the task workspace and return its content with "
            "1-based, right-aligned line numbers. Use start_line/end_line (1-based, inclusive) "
            "to read a slice of large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to show (1-based, inclusive). Defaults to 1.",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to show (1-based, inclusive). Omit to read to the end of the file.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search",
        "description": (
            "Search for a literal (case-sensitive, non-regex) substring across text files inside "
            "the task workspace. Matches are returned as 'relative/path:line: text'. Restrict the "
            "scope with 'path' (a subdirectory, or a single file)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal substring to look for (case-sensitive).",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory or file to search under, relative to the workspace root. Defaults to '.'.",
                    "default": ".",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact, unique occurrence of old_string with new_string in an existing file. "
            "The edit is applied only when old_string appears exactly once. "
            "old_string matches the file's raw content; do NOT include the line-number / tab prefix "
            "that read_file shows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit, relative to the workspace root.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find; must appear exactly once in the file (raw content, no line-number prefix).",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a UTF-8 text file inside the task workspace, creating parent "
            "directories as needed. Use this to create new files; use edit_file to modify existing ones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to write, relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write (overwrites any existing content).",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run pytest inside the task workspace and report PASS/FAIL with a compact failure "
            "summary. Optionally pass 'path' to scope to a test file, a directory, or a node id "
            "(e.g. tests/test_parser.py::test_unary_minus); omit to run the full suite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Test file, directory, or node id to run, relative to the workspace root. Omit to run everything.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# 6 个工具 handler（脚手架桩：签名 + 契约 docstring + NotImplementedError）
# ---------------------------------------------------------------------------
def list_dir(workdir: str, path: str = ".") -> str:
    """列出 ``path`` 目录下的条目（DESIGN §6.2 工具 1）。

    行为
    ----
    - 先 ``sandbox.resolve_in_workdir(workdir, path)`` 解析出安全绝对路径。
    - 列出条目，按**字母序**排序；目录名追加尾部 ``/``。
    - 过滤噪声：``__pycache__`` / ``.pytest_cache`` / ``*.pyc`` / ``.git``。

    边界 / 错误串
    -------------
    - 目录不存在 → ``"错误：目录不存在：<path>"``。
    - ``path`` 指向文件（非目录）→ ``"错误：不是目录：<path>"``。
    - 过滤后为空目录 → ``"<path>/：（空目录）"``。

    返回格式（成功）
    ----------------
    首行为目录名，随后每行一个条目、两空格缩进。文本里的路径用 workdir 相对形式。
    """
    abs_path = sandbox.resolve_in_workdir(workdir, path)
    if not os.path.exists(abs_path):
        return f"错误：目录不存在：{path}"
    if not os.path.isdir(abs_path):
        return f"错误：不是目录：{path}"
    
    NOISE = {"__pycache__", ".pytest_cache", "*.pyc", ".git"}
    names = [
        n for n in os.listdir(abs_path)
        if n not in NOISE and not n.endswith(".pyc")
    ]
    names.sort()

    entries = []
    for name in names:
        full = os.path.join(abs_path, name)
        entries.append(name + "/" if os.path.isdir(full) else name)
    if not entries:
        return f"{path}/：（空目录）"

    lines = [f"{path}/："]                 # 首行：目录名
    lines += ["  " + e for e in entries]  # 每条两空格缩进
    return "\n".join(lines)

def read_file(
    workdir: str,
    path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    max_read_lines: int = 400,
) -> str:
    """读取文本文件并带行号返回（DESIGN §6.2 工具 2）。

    行为
    ----
    - 先 ``sandbox.resolve_in_workdir(workdir, path)`` 解析。
    - 读 utf-8、按 ``\\n`` 切行；取 ``[start_line, end_line]``（1-based、含端点）。
    - 每行前缀右对齐行号 + 制表符：``f"{n:>6}\\t{line}"``。

    边界 / 错误串
    -------------
    - 文件不存在 → ``"错误：文件不存在：<path>"``。
    - 目标是目录 → ``"错误：不是文件（是目录）：<path>"``。
    - 非文本 / 无法 utf-8 解码 → ``"错误：无法以文本读取（疑似二进制）：<path>"``
      （显式 ``except UnicodeDecodeError``）。
    - ``start_line < 1`` → 夹取到 1；``end_line`` 超总行数 → 夹取到末行。
    - ``start_line > 总行数 M`` → ``"错误：起始行 {start_line} 超过文件总行数 {M}"``。
    - ``start_line > end_line`` → ``"错误：start_line 大于 end_line"``。
    - 空文件 → 带头部的 ``"（空文件）"``。

    体量护栏
    --------
    单次最多显示 ``max_read_lines`` 行（默认 400，对齐 ``config.max_read_lines``）；
    超出只显示前 N 行并附 ``"…（已截断，共 {M} 行，用 start_line/end_line 分段读取）"``。

    返回格式（成功）
    ----------------
    头部给出总行数与显示范围，如 ``parser.py（共 78 行，显示 1-40 行）：``，随后为带行号的正文。
    """
    abs_path = sandbox.resolve_in_workdir(workdir, path)
    if not os.path.exists(abs_path):
        return f"错误：文件不存在：{path}"
    if os.path.isdir(abs_path):
        return f"错误：该路径对应目录而非文件：{path}"
    
    try:
        with open(abs_path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"错误：无法以文本读取文件：{path}"
    
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return f"空文件：{path}"
    if end_line is None:
        end_line = total          
    if start_line < 1:
        start_line = 1            
    if end_line > total:
        end_line = total          
    if start_line > total:
        return f"错误：起始行 {start_line} 超过文件总行数 {total}"
    if start_line > end_line:
        return f"错误：start_line {start_line} 大于 end_line {end_line}"
    truncated = False
    last = end_line
    if last - start_line + 1 > max_read_lines:
        last = start_line + max_read_lines - 1
        truncated = True
    header = f"{path}（共 {total} 行，显示 {start_line}-{last} 行）："
    body = [f"{n:>6}\t{lines[n-1]}" for n in range(start_line, last + 1)]
    out = header + "\n" + "\n".join(body)
    if truncated:
        out += f"\n…（已截断，共 {total} 行，用 start_line/end_line 分段读取）"
    return out

def search(
    workdir: str,
    query: str,
    path: str = ".",
    max_search_hits: int = 100,
) -> str:
    """仓库内字面子串检索（DESIGN §6.2 工具 3）。

    行为
    ----
    - 以 ``sandbox.resolve_in_workdir(workdir, path)`` 为根递归遍历（``path`` 指向文件则只搜该文件）。
    - 逐行**字面子串**匹配（大小写敏感，不做正则，MVP 保持简单）。
    - 命中输出 ``f"{relpath}:{lineno}: {line}"``（``relpath`` 为 workdir 相对路径）。
    - 跳过二进制 / 无法解码文件，以及 ``__pycache__`` / ``.pytest_cache`` / ``*.pyc`` / ``.git``。

    边界 / 错误串
    -------------
    - ``query`` 为空串 → ``"错误：搜索关键字不能为空"``。
    - 无命中 → ``"（无匹配）：<query>"``。

    体量护栏
    --------
    最多 ``max_search_hits`` 条（默认 100，对齐 ``config.max_search_hits``）；
    单行超 200 字符截断加 ``…``；超上限附
    ``"…（命中过多，仅显示前 100 条，请缩小 query 或 path）"``。

    返回格式（成功）
    ----------------
    命中行逐条 + 计数尾注 ``共 N 处匹配。``。
    """
    if len(query) == 0:
        return f"错误：搜索关键字不能为空"
    root = os.path.realpath(workdir)
    search_path = sandbox.resolve_in_workdir(workdir, path)

    NOISE = {"__pycache__", ".pytest_cache", ".git"}
    files = []
    if os.path.isfile(search_path):
        files.append(search_path)                          # path 指向单个文件
    else:
        for dirpath, dirnames, filenames in os.walk(search_path):
            dirnames[:] = sorted(d for d in dirnames if d not in NOISE)  # 剪枝 + 排序
            for name in sorted(filenames):
                if name.endswith(".pyc"):
                    continue
                files.append(os.path.join(dirpath, name))

    hits = []
    truncated = False
    for full in files:
        try:
            with open(full, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except UnicodeDecodeError:
            continue                                        
        rel = os.path.relpath(full, root)                   
        for lineno, line in enumerate(lines, start=1):      
            if query in line:                               
                if len(line) > 200:
                    line = line[:200] + "…"
                hits.append(f"{rel}:{lineno}: {line}")
                if len(hits) >= max_search_hits:
                    truncated = True
                    break
        if truncated:
            break

    if not hits:
        return f"（无匹配）：{query}"
    out = "\n".join(hits) + f"\n共 {len(hits)} 处匹配。"
    if truncated:
        out += f"\n…（命中过多，仅显示前 {max_search_hits} 条，请缩小 query 或 path）"
    return out

def edit_file(workdir: str, path: str, old_string: str, new_string: str) -> str:
    """唯一匹配字符串替换（DESIGN §6.2 工具 4）。

    「唯一匹配才替换」是关键教学点：把编辑变成**确定性**操作，逼模型先 ``read_file``
    看清上下文再改。

    行为
    ----
    - 先 ``sandbox.resolve_in_workdir(workdir, path)`` 解析。
    - 读文本，统计 ``old_string`` 出现次数 ``n``：
        * ``n == 0`` → ``"错误：old_string 未在文件中找到，未做修改。请检查空白与缩进是否完全一致。"``
        * ``n > 1``  → ``"错误：old_string 在文件中出现 {n} 次，不唯一，未做修改。请在 old_string 里加入更多上下文使其唯一。"``
        * ``n == 1`` → 替换、写回、返回成功。

    边界 / 错误串
    -------------
    - 文件不存在 → ``"错误：文件不存在：<path>（如需新建请用 write_file）"``。
    - 目标是目录 → ``"错误：不是文件（是目录）：<path>"``。
    - ``old_string == new_string`` → ``"错误：old_string 与 new_string 相同，无需修改。"``（拒绝空操作，防空转耗步数）。

    返回格式（成功）
    ----------------
    确认信息 + 替换处起始行号 + 修改后该处上下文（带行号，最多约 5 行）。
    失败分支**不得**改动磁盘文件内容。
    """
    if old_string == new_string:
        return "错误：old_string 与 new_string 相同，无需修改。"
    abs_path = sandbox.resolve_in_workdir(workdir, path)
    if not os.path.exists(abs_path):
        return f"错误：文件不存在：{abs_path} (如需新建请用 write_file)"
    if not os.path.isfile(abs_path):
        return f"错误：该路径对应的不是文件：{abs_path}"
    
    try:
        with open(abs_path, encoding="utf-8") as f:
            content = f.read()

        n = content.count(old_string)          
        if n == 0:
            return f"错误：old_string 未在文件中找到，未做修改。"
        if n > 1:
            return f"错误：old_string 在文件中出现 {n} 次，不唯一，未做修改。"

        new_content = content.replace(old_string, new_string)
        with open(abs_path, "w", encoding="utf-8") as f:   # "w" = 覆盖整个文件
            f.write(new_content)

        offset = content.index(old_string)         
        start_line = content[:offset].count("\n") + 1   # old_string 所在行数
        new_lines = new_content.splitlines()
        end = min(start_line + 4, len(new_lines))          # start_line 起，最多 5 行
        context = "\n".join(
            f"{i:>6}\t{new_lines[i - 1]}" for i in range(start_line, end + 1)
        )
        return f"已替换 {path}（第 {start_line} 行）：\n{context}"
    
    except UnicodeDecodeError:
        return f"错误：文件 {abs_path} 打开失败，请检查是否为二进制格式"
    
def write_file(workdir: str, path: str, content: str) -> str:
    """创建或覆盖文本文件（DESIGN §6.2 工具 5）。

    行为
    ----
    - ``sandbox.resolve_in_workdir(workdir, path)``（允许目标尚不存在，但规范路径须在 workdir 内）。
    - ``mkdir(parents=True)`` 建父目录；utf-8 写入，已存在则覆盖。
    - 记录写前是否存在，以区分「创建 / 覆盖」。

    边界 / 错误串
    -------------
    - 规范化后指向已存在目录 → ``"错误：目标是一个目录：<path>"``。

    返回格式（成功）
    ----------------
    ``f"已{创建|覆盖} <path>（{字节数} 字节，{行数} 行）。"``（``<path>`` 为 workdir 相对形式）。
    """
    abs_path = sandbox.resolve_in_workdir(workdir, path)
    if os.path.isdir(abs_path):
        return f"错误：目标是一个目录：{abs_path}"
    existed = os.path.exists(abs_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    n_bytes = len(content.encode("utf-8"))
    n_lines = len(content.splitlines())
    return f"已{'覆盖' if existed else '创建'} {path}（{n_bytes} 字节，{n_lines} 行）。"

def run_tests(
    workdir: str,
    path: Optional[str] = None,
    timeout: int = 60,
    max_test_output: int = 4000,
) -> str:
    """在 workspace 内跑 pytest 并回报 PASS/FAIL（DESIGN §6.2 工具 6）。

    ``timeout`` 由调用方（``guarded_execute``）从 ``config.run_tests_timeout_s``（默认 60）注入；
    ``max_test_output`` 默认 4000，对齐 ``config.max_test_output``。二者均不在 handler 内硬编码语义。

    (a) 如何用 subprocess 跑 pytest
    -------------------------------
    用当前 ``.venv`` 解释器以模块方式启动（``sys.executable -m pytest``，保证用到虚拟环境里的
    pytest 与 fixture 依赖）；``cwd=workdir`` 是关键（让 ``conftest.py`` 被加载、
    ``import tokenizer/…`` 解析）::

        cmd = [sys.executable, "-m", "pytest",
               "-q",                       # 安静：精简进度/汇总
               "--tb=short",               # 短回溯：保留关键帧、控体量
               "-rfE",                     # 末尾打印紧凑 FAILED/ERROR 清单（每个一行 + 一句异常）
               "--color=no",               # 去 ANSI
               "-p", "no:cacheprovider"]   # 不写 .pytest_cache，保持工作目录干净
        if path:
            cmd.append(path)              # path 的文件部分已由护栏校验
        result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)

    **node id 中的 ``::``**：``path`` 可能形如 ``tests/test_parser.py::test_unary_minus``。
    路径封闭只对 ``"::"`` **之前的文件部分**调 ``sandbox.resolve_in_workdir``（确认在 workdir 内），
    完整字符串原样作为 pytest 参数（先劈开 ``::`` 再校验文件部分）。

    (b) 双信号定性 + 展示
    ---------------------
    以 ``returncode`` 定性（权威绿/红）；用一条正则（类似
    ``r"(\\d+)\\s+(passed|failed|errors?|skipped|xfailed|xpassed)"``）从 stdout 末尾汇总行抽取
    ``(数字, 类别)`` 对，**仅用于展示、不作判定**。退出码映射：

        returncode == 0 → 顶行 ``结果：PASS``
        returncode == 1 → 顶行 ``结果：FAIL``
        returncode == 5 → ``警告：未收集到任何测试（检查 path / 测试是否存在）``
        returncode ∈ {2,3,4} → ``错误：pytest 自身异常（returncode=X）`` + stderr 尾部摘要

    汇总行里的 ``in 0.06s`` 是变化的计时，只展示、**不作判定**（否则单测 flaky）。判成败的
    权威始终是 §10 的独立复跑，本工具解析只服务模型中途决策。

    (c) 截断失败回溯
    ----------------
    分三段拼装，只对「失败详情」段做预算截断：
      1. 顶行 + 统计（永远保留）。
      2. 失败清单（来自 ``-rfE``，永远保留：点名哪些测试红了 + 一句原因）。
      3. 失败详情（``--tb=short`` 正文，``=== FAILURES ===`` 到短汇总之间）按 ``max_test_output``
         保留开头，超出砍尾并附
         ``"…（失败详情过长已截断，共 {X} 字符，先修上面的用例再重跑）"``。
    **不加 ``-x``**（需完整计数与回归信息）。

    (d) 超时处理
    ------------
    ``subprocess.run(..., timeout=timeout)`` 超时抛 ``subprocess.TimeoutExpired``，在本函数内
    **显式捕获**（不留给护栏，以便给专门信息并带部分输出）：取 ``e.stdout`` 尾部约 1000 字符，返回
    ``"错误：测试运行超时（>{timeout}s），已终止——可能改坏后引入死循环或无限递归。\\n部分输出：\\n{partial}"``。
    ``subprocess.run`` 超时会杀子进程；MVP 单进程 pytest 无需处理进程组
    （前瞻：将来接 ``-n``（pytest-xdist）需改用进程组 kill 清理孙进程）。

    (e) 返回给模型的格式
    --------------------
    - 全绿：``[run_tests] 目标：全量`` / ``结果：PASS（returncode=0）`` / ``统计：51 passed（用时 0.42s）``。
    - 有失败：多加「失败用例：」段（点名 node id + 一句原因）与
      「失败详情（--tb=short，已截断/未截断）：」段。
    """
    # 1. path 校验：只对 "::" 之前的文件部分做路径封闭（越界抛 PathEscape，交给护栏）。
    if path:
        file_part = path.split("::", 1)[0]
        sandbox.resolve_in_workdir(workdir, file_part)

    # 2. 组命令、在 workdir 里跑 pytest。
    cmd = [
        sys.executable, "-m", "pytest",
        "-q", "--tb=short", "-rfE", "--color=no",
        "-p", "no:cacheprovider",
    ]
    if path:
        cmd.append(path)
    try:
        result = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return (
            f"错误：测试运行超时（>{timeout}s），已终止——可能改坏后引入死循环或无限递归。\n"
            f"部分输出：\n{partial[-1000:]}"
        )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    rc = result.returncode
    target = path if path else "全量"

    # 3. 从汇总行抽统计（仅展示、不作判定）。
    stats = "、".join(
        f"{num} {kind}"
        for num, kind in re.findall(
            r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)", stdout
        )
    ) or "无统计"

    # 4. 按 returncode 定性。
    if rc == 5:
        return f"[run_tests] 目标：{target}\n警告：未收集到任何测试（检查 path / 测试是否存在）"
    if rc in (2, 3, 4):
        return (
            f"[run_tests] 目标：{target}\n错误：pytest 自身异常（returncode={rc}）\n"
            f"{stderr[-1000:]}"
        )

    verdict = "PASS" if rc == 0 else "FAIL"
    head = f"[run_tests] 目标：{target}\n结果：{verdict}（returncode={rc}）\n统计：{stats}"
    if rc == 0:
        return head

    # 5. rc == 1：失败清单（-rfE 短行，永远保留）+ 失败详情（按预算截断）。
    fail_lines = [ln for ln in stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    fails = "\n".join(fail_lines) if fail_lines else "（无 FAILED/ERROR 明细行）"

    detail = ""
    if "= FAILURES =" in stdout:
        detail = stdout.split("= FAILURES =", 1)[1]
        detail = detail.split("= short test summary", 1)[0].strip("=\n ")
    if len(detail) > max_test_output:
        detail = (
            detail[:max_test_output]
            + f"\n…（失败详情过长已截断，共 {len(detail)} 字符，先修上面的用例再重跑）"
        )

    return f"{head}\n失败用例：\n{fails}\n失败详情（--tb=short）：\n{detail}"


# ---------------------------------------------------------------------------
# guarded_execute —— 护栏分发器（所有工具的安全咽喉）
# ---------------------------------------------------------------------------
def guarded_execute(
    tool_name: str,
    tool_input: dict,
    workdir: str,
    *,
    test_timeout: int,
    max_result_chars: int,
) -> str:
    """loop 唯一调用的执行入口：把路径越界与任何异常都收敛成字符串（DESIGN §6.3）。

    保证 loop 侧**永远只拿到 ``str``、永不见异常**。``test_timeout`` / ``max_result_chars``
    由 loop 从 config 抽取传入（``config.run_tests_timeout_s`` / ``config.max_tool_result_chars``）。

    内部处理顺序
    ------------
    1. **未知工具**：``tool_name`` 不在
       ``{list_dir, read_file, search, edit_file, write_file, run_tests}`` →
       立即返回 ``f"错误：未知工具 {tool_name}"``（防模型幻觉出不存在的工具）。
    2. **分发**：查表拿 handler，``run_tests`` 额外注入 ``timeout=test_timeout``；用 ``**tool_input`` 传参。
    3. **异常兜底**（单一 try/except，由具体到兜底）：
         * ``except sandbox.PathEscape`` → 统一越界串
           ``"错误：路径越界，只能访问任务工作目录内的文件：<原始 path>"``
           （**护栏层承担的路径封闭兜底**：无论哪个工具触发越界都在此转成同一句，
           绝不冒泡进 loop、绝不落到 workdir 外）。
         * ``except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError)`` → 各自可读错误串。
         * ``except TypeError as e`` → 参数不匹配（模型给了多余/缺失字段）
           ``"错误：工具参数不合法：{e}"``。
         * ``except Exception as e`` → 终极兜底 ``f"错误：工具执行失败：{type(e).__name__}: {e}"``。
    4. **输出体量最后一道闸**：把最终字符串截断到 ``max_result_chars``，超出附
       ``"…（输出过长已截断）"``。与各工具内部截断构成双保险。

    不变式（测试逐条验证）
    ----------------------
    对**任意** ``tool_name`` / ``tool_input`` / ``workdir``，本函数**只返回 ``str``、永不抛异常**；
    任何最终会触碰 workdir 之外文件系统的操作都被拦成错误串、不产生实际读写。
    """
    handlers = {
        "list_dir": list_dir,
        "read_file": read_file,
        "search": search,
        "edit_file": edit_file,
        "write_file": write_file,
        "run_tests": run_tests,
    }
    # 1. 未知工具：立即拦下（防模型幻觉出不存在的工具）。
    if tool_name not in handlers:
        return f"错误：未知工具 {tool_name}"

    # 2. 分发：run_tests 额外注入 timeout；其余原样 **tool_input。
    handler = handlers[tool_name]
    kwargs = dict(tool_input)
    if tool_name == "run_tests":
        kwargs["timeout"] = test_timeout

    # 3. 异常兜底（由具体到兜底），保证只返回 str、永不抛。
    try:
        result = handler(workdir, **kwargs)
    except sandbox.PathEscape:
        original = tool_input.get("path", "")
        result = f"错误：路径越界，只能访问任务工作目录内的文件：{original}"
    except (FileNotFoundError, IsADirectoryError) as e:
        result = f"错误：文件访问失败：{e}"
    except UnicodeDecodeError:
        result = "错误：无法以文本读取（疑似二进制）"
    except TypeError as e:
        result = f"错误：工具参数不合法：{e}"
    except Exception as e:
        result = f"错误：工具执行失败：{type(e).__name__}: {e}"

    # 4. 输出体量最后一道闸。
    if len(result) > max_result_chars:
        result = result[:max_result_chars] + "…（输出过长已截断）"
    return result
