"""tests/test_tools.py —— agent/tools.py 的起步测试集（DESIGN §6.4 验收标准）。

策略（脚手架）
--------------
- **声明式的 ``TOOLS`` schema** 已在 tools.py 写全 → 这些断言是**绿灯**（当下即通过）。
- **handler / guarded_execute 是桩**（``raise NotImplementedError``）→ 针对它们行为的断言是
  **红灯规格（M0）**：现在失败，实现后转绿。安全相关（路径封闭经 guarded_execute）完整覆盖。
- 其余较复杂 / 依赖真实 pytest 子进程的契约用 ``# TODO(你来补)`` 明确列出。

约定：临时目录用 pytest 的 ``tmp_path`` 作 ``workdir``；guarded_execute 的两个护栏标量在测试里
以位置无关的关键字传入（对齐 config 默认 ``run_tests_timeout_s=60`` / ``max_tool_result_chars=8000``）。
"""
from __future__ import annotations

import pytest

from agent.tools import (
    TOOLS,
    edit_file,
    guarded_execute,
    list_dir,  # noqa: F401  (供 TODO 段落补测引用)
    read_file,
    run_tests,  # noqa: F401  (供 TODO 段落补测引用)
    search,
    write_file,
)

# guarded_execute 的护栏标量（loop 实际从 config 抽取；测试里给定值）。
GUARD_KW = {"test_timeout": 60, "max_result_chars": 8000}

# handler 名 ↔ 契约集合（用于 schema 一致性断言）。
EXPECTED_TOOL_NAMES = {
    "list_dir",
    "read_file",
    "search",
    "edit_file",
    "write_file",
    "run_tests",
}
EXPECTED_REQUIRED = {
    "list_dir": set(),
    "read_file": {"path"},
    "search": {"query"},
    "edit_file": {"path", "old_string", "new_string"},
    "write_file": {"path", "content"},
    "run_tests": set(),
}


# ===========================================================================
# TOOLS schema —— 声明式，已写全，当下即绿（DESIGN §6.4「TOOLS schema」）
# ===========================================================================
def test_tools_count_is_six():
    assert len(TOOLS) == 6


def test_tools_names_match_handlers():
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)), "工具名不得重复"
    assert set(names) == EXPECTED_TOOL_NAMES


def test_tools_schema_shape():
    for t in TOOLS:
        assert {"name", "description", "input_schema"} <= set(t.keys())
        assert isinstance(t["description"], str) and t["description"].strip()
        schema = t["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema and isinstance(schema["properties"], dict)
        # 一律带 additionalProperties: false
        assert schema.get("additionalProperties") is False


def test_tools_required_matches_contract():
    for t in TOOLS:
        required = set(t["input_schema"].get("required", []))
        assert required == EXPECTED_REQUIRED[t["name"]]
        # required 字段必须都在 properties 里声明
        assert required <= set(t["input_schema"]["properties"].keys())


# ===========================================================================
# 安全：路径封闭经 guarded_execute（红灯规格，安全相关，完整覆盖 DESIGN §6.4「路径封闭」）
# ===========================================================================
# 每个碰文件系统的工具都喂越界路径（../ 相对越界 + 绝对路径），断言：
#   (1) 返回类型是 str；(2) 以「错误：」开头；(3) 不抛异常。
ESCAPE_CASES = [
    ("read_file", {"path": "../secret.txt"}),
    ("read_file", {"path": "/etc/hosts"}),
    ("list_dir", {"path": ".."}),
    ("list_dir", {"path": "/etc"}),
    ("search", {"query": "x", "path": "../"}),
    ("edit_file", {"path": "../secret.txt", "old_string": "a", "new_string": "b"}),
    ("write_file", {"path": "../evil.txt", "content": "boom"}),
    ("run_tests", {"path": "../tests"}),
]


@pytest.mark.parametrize("tool_name,tool_input", ESCAPE_CASES)
def test_guarded_execute_path_escape_returns_error_string(tmp_path, tool_name, tool_input):
    out = guarded_execute(tool_name, tool_input, str(tmp_path), **GUARD_KW)
    assert isinstance(out, str)
    assert out.startswith("错误：")


def test_path_escape_does_not_read_outside_file(tmp_path):
    # work/ 是 workdir；outside/ 在其外，放一个机密文件。
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET", encoding="utf-8")

    out = guarded_execute(
        "read_file", {"path": "../outside/secret.txt"}, str(work), **GUARD_KW
    )
    assert out.startswith("错误：")
    assert "TOPSECRET" not in out  # 机密内容绝不能出现在返回里


def test_path_escape_does_not_write_outside_file(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    target = tmp_path / "evil.txt"  # 位于 workdir 之外

    out = guarded_execute(
        "write_file", {"path": "../evil.txt", "content": "boom"}, str(work), **GUARD_KW
    )
    assert out.startswith("错误：")
    assert not target.exists()  # 越界写入绝不能真的落盘


# ===========================================================================
# guarded_execute 不变式（红灯规格，DESIGN §6.4「guarded_execute 不变式」）
# ===========================================================================
def test_guarded_execute_unknown_tool(tmp_path):
    out = guarded_execute("frobnicate", {}, str(tmp_path), **GUARD_KW)
    assert isinstance(out, str)
    assert out.startswith("错误：")
    assert "未知工具" in out


def test_guarded_execute_missing_required_arg_returns_error(tmp_path):
    # read_file 缺必填 path → 不得抛异常，收敛成错误串。
    out = guarded_execute("read_file", {}, str(tmp_path), **GUARD_KW)
    assert isinstance(out, str)
    assert out.startswith("错误：")


def test_guarded_execute_unexpected_arg_returns_error(tmp_path):
    # 模型幻觉出多余字段 → TypeError 兜底成错误串，返回类型恒为 str。
    out = guarded_execute(
        "list_dir", {"bogus_field": 123}, str(tmp_path), **GUARD_KW
    )
    assert isinstance(out, str)
    assert out.startswith("错误：")


# ===========================================================================
# read_file：带行号（红灯规格，DESIGN §6.4「read_file」）
# ===========================================================================
def test_read_file_line_numbers(tmp_path):
    (tmp_path / "sample.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    out = read_file(str(tmp_path), "sample.py")
    # 每行前缀：右对齐到宽度 6 的行号 + 制表符（f"{n:>6}\t{line}"）。
    assert "     1\talpha" in out
    assert "     2\tbeta" in out
    assert "     3\tgamma" in out
    # 头部给出总行数。
    assert "共 3 行" in out
    # 成功不带「错误：」前缀。
    assert not out.startswith("错误：")


def test_read_file_missing_file_returns_error(tmp_path):
    out = read_file(str(tmp_path), "nope.py")
    assert out.startswith("错误：")


# ===========================================================================
# search：命中格式（红灯规格，DESIGN §6.4「search」）
# ===========================================================================
def test_search_hit_format(tmp_path):
    (tmp_path / "code.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    out = search(str(tmp_path), "foo")
    # 命中行格式 relpath:line: text（path 为 workdir 相对）。
    assert "code.py:1:" in out
    assert "foo" in out
    # 计数尾注。
    assert "共 1 处匹配" in out


def test_search_no_match(tmp_path):
    (tmp_path / "code.py").write_text("nothing here\n", encoding="utf-8")
    out = search(str(tmp_path), "zzz_not_present")
    assert out.startswith("（无匹配）")


# ===========================================================================
# edit_file：old 不存在 / 不唯一 / 唯一成功（红灯规格，DESIGN §6.4「edit_file」）
# ===========================================================================
def test_edit_file_old_not_found(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    out = edit_file(str(tmp_path), "m.py", "NOT_PRESENT", "y = 2")
    assert out.startswith("错误：")
    # 未做修改：磁盘内容不变。
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_file_old_not_unique(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a\na\n", encoding="utf-8")
    out = edit_file(str(tmp_path), "m.py", "a", "b")
    assert out.startswith("错误：")
    assert "2" in out  # 报告出现次数
    # 不唯一 → 未做修改。
    assert f.read_text(encoding="utf-8") == "a\na\n"


def test_edit_file_unique_replaces_on_disk(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    out = edit_file(str(tmp_path), "m.py", "x = 1", "x = 2")
    assert not out.startswith("错误：")
    # 唯一匹配 → 磁盘内容确实变更。
    assert f.read_text(encoding="utf-8") == "x = 2\n"


def test_edit_file_old_equals_new_rejected(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    out = edit_file(str(tmp_path), "m.py", "x = 1", "x = 1")
    assert out.startswith("错误：")
    assert f.read_text(encoding="utf-8") == "x = 1\n"


# ===========================================================================
# write_file：创建 / 覆盖（红灯规格，DESIGN §6.4「write_file」）
# ===========================================================================
def test_write_file_creates_with_parent_dirs(tmp_path):
    out = write_file(str(tmp_path), "sub/new.py", "print('hi')\n")
    assert not out.startswith("错误：")
    assert (tmp_path / "sub" / "new.py").read_text(encoding="utf-8") == "print('hi')\n"


# ===========================================================================
# TODO(你来补)：以下契约留待补测（DESIGN §6.2 / §6.4）
# ===========================================================================
# --- read_file ---
# TODO(你来补): 测试 start_line/end_line 切片（1-based、含端点）取到正确的行范围。
# TODO(你来补): 测试 start_line < 1 夹取到 1；end_line 超总行数夹取到末行。
# TODO(你来补): 测试 start_line > 总行数 → "错误：起始行 … 超过文件总行数 …"。
# TODO(你来补): 测试 start_line > end_line → "错误：start_line 大于 end_line"。
# TODO(你来补): 测试目录路径 → "错误：不是文件（是目录）：…"。
# TODO(你来补): 测试二进制/非 utf-8 文件 → "错误：无法以文本读取（疑似二进制）：…"。
# TODO(你来补): 测试空文件 → 带头部的 "（空文件）"。
# TODO(你来补): 测试超过 max_read_lines（默认 400）→ 只显示前 N 行 + "…（已截断，共 M 行 …）"。
#
# --- search ---
# TODO(你来补): 测试 query 为空串 → "错误：搜索关键字不能为空"。
# TODO(你来补): 测试限定子目录 / 单文件时搜索范围正确收窄。
# TODO(你来补): 测试跳过 __pycache__/.pytest_cache/*.pyc/.git 噪声与二进制文件。
# TODO(你来补): 测试单行超 200 字符被截断加 "…"。
# TODO(你来补): 测试超过 max_search_hits（默认 100）→ 截断 + "…（命中过多 …）"。
#
# --- list_dir ---
# TODO(你来补): 测试条目按字母序、目录名带尾 "/"。
# TODO(你来补): 测试过滤 __pycache__/.pytest_cache/*.pyc/.git 噪声。
# TODO(你来补): 测试过滤后空目录 → "<path>/：（空目录）"。
# TODO(你来补): 测试目录不存在 → "错误：目录不存在：…"；path 是文件 → "错误：不是目录：…"。
#
# --- write_file ---
# TODO(你来补): 测试覆盖已存在文件时返回文案区分「已覆盖」与「已创建」，且含字节数/行数。
# TODO(你来补): 测试目标规范化后是已存在目录 → "错误：目标是一个目录：…"。
#
# --- edit_file ---
# TODO(你来补): 测试文件不存在 → "错误：文件不存在：…（如需新建请用 write_file）"。
# TODO(你来补): 测试唯一替换成功返回里含替换处起始行号与上下文（带行号，约 5 行）。
#
# --- run_tests（核心，依赖真实 pytest 子进程；建议用 fixture 临时副本作 workdir）---
# TODO(你来补): 测试纯净副本 → 顶行 PASS、returncode=0、统计 passed 数与实际用例数一致。
# TODO(你来补): 测试 git apply 某 break.patch 后 → 顶行 FAIL、「失败用例」段点名目标测试。
# TODO(你来补): 测试传该 node id（tests/xxx::test_yyy）单跑也复现 FAIL；:: 前文件部分经路径封闭。
# TODO(你来补): 测试失败详情 ≤ max_test_output（默认 4000），超出带截断标记。
# TODO(你来补): 测试注入死循环 break + 很小 timeout → 返回 "错误：测试运行超时…"，函数在 timeout 附近返回。
# TODO(你来补): 测试不存在的测试路径 → returncode=5 分支的「未收集到任何测试」提示。
# TODO(你来补): 测试连续跑 3 次判定与计数一致（不因 "in 0.06s" 计时波动 flaky）。
#
# --- guarded_execute ---
# TODO(你来补): 测试对随机/畸形 tool_input 循环调用，返回类型恒为 str（属性测试式）。
# TODO(你来补): 测试最终输出超过 max_result_chars 被截断 + "…（输出过长已截断）"。
# TODO(你来补): 测试 run_tests 经 guarded_execute 时 timeout=test_timeout 被正确注入。
#
# --- 返回约定（横切）---
# TODO(你来补): 测试各工具对外文本中的路径均为 workdir 相对形式（断言不含 workdir 绝对前缀）。
