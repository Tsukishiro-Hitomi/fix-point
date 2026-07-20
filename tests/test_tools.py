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


def test_read_file_slice_is_inclusive(tmp_path):
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")
    out = read_file(str(tmp_path), "f.txt", start_line=2, end_line=4)
    assert not out.startswith("错误：")
    for n in (2, 3, 4):
        assert f"line{n}" in out
    assert "     1\t" not in out  # 第 1 行不在范围内
    assert "     5\t" not in out  # 第 5 行不在范围内


def test_read_file_clamps_out_of_range(tmp_path):
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 6)), encoding="utf-8")
    out = read_file(str(tmp_path), "f.txt", start_line=-3, end_line=999)
    assert not out.startswith("错误：")
    assert "     1\tline1" in out  # start 夹到 1
    assert "     5\tline5" in out  # end 夹到末行


def test_read_file_start_beyond_total(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")
    out = read_file(str(tmp_path), "f.txt", start_line=5)
    assert out.startswith("错误：")
    assert "超过文件总行数" in out


def test_read_file_start_gt_end(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")
    out = read_file(str(tmp_path), "f.txt", start_line=3, end_line=2)
    assert out.startswith("错误：")
    assert "大于" in out


def test_read_file_on_directory(tmp_path):
    (tmp_path / "d").mkdir()
    out = read_file(str(tmp_path), "d")
    assert out.startswith("错误：")
    assert "目录" in out


def test_read_file_binary_returns_error(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\xff\xfe\x00\x01\xff")
    out = read_file(str(tmp_path), "b.bin")
    assert out.startswith("错误：")
    assert "无法以文本读取" in out


def test_read_file_empty(tmp_path):
    (tmp_path / "e.txt").write_text("", encoding="utf-8")
    out = read_file(str(tmp_path), "e.txt")
    assert "空文件" in out
    assert not out.startswith("错误：")  # 空文件不是错误


def test_read_file_truncates_over_max(tmp_path):
    (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(1, 21)), encoding="utf-8")
    out = read_file(str(tmp_path), "big.txt", max_read_lines=5)
    assert "已截断" in out
    assert "     1\tline1" in out
    assert "     5\tline5" in out
    assert "     6\t" not in out  # 只显示前 5 行


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


def test_search_empty_query_returns_error(tmp_path):
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
    out = search(str(tmp_path), "")
    assert out.startswith("错误：")
    assert "不能为空" in out


def test_search_scope_subdir(tmp_path):
    (tmp_path / "top.py").write_text("needle here\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.py").write_text("needle here\n", encoding="utf-8")
    out = search(str(tmp_path), "needle", path="sub")  # 只搜 sub/
    assert "sub/inner.py:1:" in out
    assert "top.py" not in out
    assert "共 1 处匹配" in out


def test_search_scope_single_file(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("needle\n", encoding="utf-8")
    out = search(str(tmp_path), "needle", path="a.py")  # path 指向单文件
    assert "a.py:1:" in out
    assert "b.py" not in out
    assert "共 1 处匹配" in out


def test_search_skips_noise_and_binary(tmp_path):
    (tmp_path / "real.py").write_text("needle\n", encoding="utf-8")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("needle\n", encoding="utf-8")  # 噪声目录
    (tmp_path / "mod.pyc").write_text("needle\n", encoding="utf-8")   # .pyc
    (tmp_path / "blob.bin").write_bytes(b"needle\xff\xfe\x00")        # 二进制
    out = search(str(tmp_path), "needle")
    assert "real.py:1:" in out
    assert "__pycache__" not in out
    assert "mod.pyc" not in out
    assert "blob.bin" not in out
    assert "共 1 处匹配" in out


def test_search_truncates_long_line(tmp_path):
    long_line = "x" * 100 + "needle" + "y" * 200  # 命中且远超 200 字符
    (tmp_path / "long.py").write_text(long_line + "\n", encoding="utf-8")
    out = search(str(tmp_path), "needle")
    assert "long.py:1:" in out
    assert "…" in out
    assert "y" * 200 not in out  # 200 字符之后被砍掉


def test_search_caps_at_max_hits(tmp_path):
    (tmp_path / "many.py").write_text("hit\n" * 10, encoding="utf-8")  # 10 行都命中
    out = search(str(tmp_path), "hit", max_search_hits=3)
    assert "命中过多" in out
    assert "共 3 处匹配" in out  # 只取前 3 条


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


def test_write_file_create_then_overwrite(tmp_path):
    out1 = write_file(str(tmp_path), "f.txt", "hello\n")
    assert not out1.startswith("错误：")
    assert "创建" in out1
    assert "6 字节" in out1  # "hello\n" 编码为 6 字节
    assert "1 行" in out1
    out2 = write_file(str(tmp_path), "f.txt", "hi\n")  # 再写同一文件
    assert "覆盖" in out2
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "hi\n"


def test_write_file_target_is_dir_returns_error(tmp_path):
    (tmp_path / "d").mkdir()
    out = write_file(str(tmp_path), "d", "x")
    assert out.startswith("错误：")
    assert "目录" in out
    assert (tmp_path / "d").is_dir()  # 目录未被破坏


# ===========================================================================
# list_dir：排序 / 目录尾斜杠 / 过滤噪声 / 空目录 / 错误串（DESIGN §6.4「list_dir」）
# ===========================================================================
def test_list_dir_sorted_with_dir_slash(tmp_path):
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = list_dir(str(tmp_path))
    assert not out.startswith("错误：")
    # 首行是目录名，其后每行一个条目（两空格缩进）。
    entries = [ln.strip() for ln in out.splitlines()[1:]]
    assert entries == ["a.py", "b.py", "sub/"]  # 字母序；目录带尾 "/"


def test_list_dir_filters_noise(tmp_path):
    (tmp_path / "keep.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "mod.pyc").write_text("", encoding="utf-8")
    out = list_dir(str(tmp_path))
    assert "keep.py" in out
    for noise in ("__pycache__", ".pytest_cache", ".git", ".pyc"):
        assert noise not in out


def test_list_dir_empty_after_filter(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "__pycache__").mkdir()  # 只有噪声 → 过滤后视为空
    out = list_dir(str(tmp_path), "d")
    assert out == "d/：（空目录）"


def test_list_dir_missing_and_not_dir(tmp_path):
    assert list_dir(str(tmp_path), "nope") == "错误：目录不存在：nope"
    (tmp_path / "f.py").write_text("", encoding="utf-8")
    assert list_dir(str(tmp_path), "f.py") == "错误：不是目录：f.py"


# ===========================================================================
# TODO(你来补)：以下契约留待补测（DESIGN §6.2 / §6.4）
# ===========================================================================
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
