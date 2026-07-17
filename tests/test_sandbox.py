"""``agent.sandbox`` 的单测（DESIGN §5.4 验收标准）。

路径封闭是**安全关键**：一个漏洞就意味着 agent 能读写工作区之外的文件。因此
``resolve_in_workdir`` 的测试写得较完整（真实断言，作为红灯规格）；工作区生命周期
相关的用例先以 ``@pytest.mark.skip(TODO...)`` 占位、逐条列出待补行为。

注：sandbox 中的实现函数当前均为桩（``raise NotImplementedError``），故下方路径封闭
用例现在应为**红灯**——它们是待实现的规格；实现补齐后即应转绿。
"""

import difflib
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from agent.sandbox import (
    PathEscape,
    SandboxError,
    cleanup_workspace,
    make_workspace,
    resolve_in_workdir,
    task_sandbox,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tasks", "fixture"
)


@pytest.fixture
def workdir(tmp_path):
    """搭一个形似 fixture 副本的工作目录，返回其路径（str）。

    结构：``evaluator.py`` / ``parser.py`` / ``sub/`` / ``tests/test_parser.py``。
    断言里一律用 ``os.path.realpath(workdir)`` 作为期望 root，避免平台上临时目录
    本身带软链造成误判。
    """
    root = tmp_path / "work"
    root.mkdir()
    (root / "evaluator.py").write_text("# eval\n")
    (root / "parser.py").write_text("# parse\n")
    (root / "sub").mkdir()
    (root / "tests").mkdir()
    (root / "tests" / "test_parser.py").write_text("# t\n")
    return str(root)


# --------------------------------------------------------------------------- #
# 异常层级（声明式，立即可绿）
# --------------------------------------------------------------------------- #
def test_pathescape_is_subclass_of_sandboxerror():
    # 契约：捕 SandboxError 必须也能兜住 PathEscape
    assert issubclass(PathEscape, SandboxError)


# --------------------------------------------------------------------------- #
# 5.2 resolve_in_workdir —— 放行路径
# --------------------------------------------------------------------------- #
def test_relative_path_resolves_inside(workdir):
    result = resolve_in_workdir(workdir, "evaluator.py")
    assert os.path.isabs(result)
    assert result == os.path.join(os.path.realpath(workdir), "evaluator.py")


def test_nested_relative_path_resolves_inside(workdir):
    result = resolve_in_workdir(workdir, "tests/test_parser.py")
    assert result == os.path.join(os.path.realpath(workdir), "tests", "test_parser.py")


@pytest.mark.parametrize("p", ["", "."])
def test_empty_and_dot_resolve_to_root(workdir, p):
    # 边界：workdir 自身通过（含 root 本身，不只是 root 的子路径）
    assert resolve_in_workdir(workdir, p) == os.path.realpath(workdir)


def test_dotdot_that_stays_inside_is_allowed(workdir):
    # sub/../parser.py 规范化后仍在目录内 → 放行
    result = resolve_in_workdir(workdir, "sub/../parser.py")
    assert result == os.path.join(os.path.realpath(workdir), "parser.py")


def test_nonexistent_target_is_allowed(workdir):
    # 「存在与否」不是路径封闭的职责：尚不存在的新文件照常返回规范路径、不报错
    result = resolve_in_workdir(workdir, "brand_new_file.py")
    assert result == os.path.join(os.path.realpath(workdir), "brand_new_file.py")


def test_workdir_itself_by_absolute_path_is_allowed(workdir):
    root = os.path.realpath(workdir)
    assert resolve_in_workdir(workdir, root) == root


def test_absolute_path_inside_is_allowed(workdir):
    root = os.path.realpath(workdir)
    target = os.path.join(root, "evaluator.py")
    assert resolve_in_workdir(workdir, target) == target


# --------------------------------------------------------------------------- #
# 5.2 resolve_in_workdir —— 越界拒绝（PathEscape）
# --------------------------------------------------------------------------- #
def test_parent_traversal_rejected(workdir):
    with pytest.raises(PathEscape):
        resolve_in_workdir(workdir, "../secret.txt")


def test_deep_parent_traversal_rejected(workdir):
    # sub/../../ 越过 root 落到父目录之外
    with pytest.raises(PathEscape):
        resolve_in_workdir(workdir, "sub/../../outside.txt")


def test_absolute_path_outside_rejected(workdir):
    with pytest.raises(PathEscape):
        resolve_in_workdir(workdir, "/etc/passwd")


def test_sibling_prefix_dir_rejected(workdir):
    """`+ os.sep` 守卫：``<workdir>-evil`` 作为字符串以 ``<workdir>`` 开头，但不在其内。

    这是路径封闭最经典的洞——判定时漏掉 ``+ os.sep`` 就会把 ``<workdir>-evil/x``
    误当成「在 workdir 内」。
    """
    evil = os.path.realpath(workdir) + "-evil"
    os.mkdir(evil)
    victim = os.path.join(evil, "x.txt")
    with open(victim, "w") as fh:
        fh.write("secret")
    with pytest.raises(PathEscape):
        resolve_in_workdir(workdir, victim)


def test_symlink_pointing_outside_rejected(workdir, tmp_path):
    """指向目录外的软链经它访问 → 拒绝（realpath 解析到外部）。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    os.symlink(str(outside), os.path.join(os.path.realpath(workdir), "evil_link"))
    with pytest.raises(PathEscape):
        resolve_in_workdir(workdir, "evil_link/secret.txt")


def test_symlink_pointing_inside_allowed(workdir):
    """指向目录内的软链 → 放行（realpath 解析回内部）。"""
    root = os.path.realpath(workdir)
    os.symlink(os.path.join(root, "sub"), os.path.join(root, "inner_link"))
    result = resolve_in_workdir(workdir, "inner_link/deep.py")
    assert result == os.path.join(root, "sub", "deep.py")


@pytest.mark.skipif(
    os.path.realpath("/tmp") == "/tmp",
    reason="本平台 /tmp 非软链，无 root 规范化陷阱可测",
)
def test_root_realpath_normalization_consistency():
    """macOS root 规范化一致性：``/tmp/...`` 与等价 ``/private/tmp/...`` 结果一致且放行。

    tempfile 给的临时目录多在 ``/tmp`` 下，而 ``/tmp`` 是指向 ``/private/tmp`` 的软链；
    若 ``resolve_in_workdir`` 未先对 root 做 realpath，包含判断会全线误判。
    """
    raw = tempfile.mkdtemp(prefix="fixpoint_test_", dir="/tmp")
    try:
        canonical = os.path.realpath(raw)
        assert raw != canonical  # 前提：/tmp 确为软链
        with open(os.path.join(raw, "f.py"), "w") as fh:
            fh.write("x")
        via_raw = resolve_in_workdir(raw, "f.py")          # 用非规范 root 访问
        assert via_raw == os.path.join(canonical, "f.py")  # 返回规范路径、放行
        assert resolve_in_workdir(canonical, "f.py") == via_raw  # 与规范 root 一致
    finally:
        shutil.rmtree(raw, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 5.3 工作区生命周期 —— 起步集（TODO：待补真实断言）
#
# 这些用例需要真实的 tasks/fixture/ 与某个 break.patch 作为素材，先占位登记；
# 补齐时请对照 §5.4 的验收清单逐条落地。
# --------------------------------------------------------------------------- #
def test_make_workspace_populates_workdir():
    workdir = make_workspace(FIXTURE_DIR)
    try:
        # 绝对路径
        assert os.path.isabs(workdir)
        # 规范路径
        assert workdir == os.path.realpath(workdir)
        # 真实路径
        assert os.path.isdir(workdir)
    finally:
        cleanup_workspace(workdir)

def snapshot(root):
    """把 root 下每个文件（排除 __pycache__/*.pyc）读成 {相对路径: 字节} 的 dict。"""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]   # 不进 __pycache__
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)          # 用相对路径当 key
            with open(full, "rb") as f:
                snap[rel] = f.read()                    # 存原始字节
    return snap

def test_make_workspace_ignores_bytecode_and_leaves_fixture_pristine():
    before = snapshot(FIXTURE_DIR)
    workdir = make_workspace(FIXTURE_DIR)
    try:
        assert snapshot(FIXTURE_DIR) == before
        for dirpath, dirnames, filenames in os.walk(workdir):
            assert "__pycache__" not in filenames
            assert not any(f.endswith(".pyc") for f in filenames)
    finally:
        cleanup_workspace(workdir)

def test_make_workspace_applies_break_patch(tmp_path):
    # 基于真实 evaluator.py 造一个破坏除法的补丁（difflib 生成，带 a/ b/ 前缀以配合 -p1）
    target = "evaluator.py"
    with open(os.path.join(FIXTURE_DIR, target), encoding="utf-8") as f:
        original = f.read()
    broken = original.replace("float(left) / right", "float(left) * right")
    assert broken != original, "预设的破坏点不在 evaluator.py 里，请换一个片段"

    patch_text = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            broken.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
        )
    )
    patch_file = tmp_path / "break.patch"
    patch_file.write_text(patch_text, encoding="utf-8")

    # 对照：不打补丁的纯净副本，evaluator 测试全绿
    baseline = make_workspace(FIXTURE_DIR)
    try:
        r0 = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_evaluator.py", "-q"],
            cwd=baseline, capture_output=True, text=True,
        )
        assert r0.returncode == 0, f"纯净副本本应全绿：\n{r0.stdout}"
    finally:
        cleanup_workspace(baseline)

    # 打了补丁的副本：除法被改坏 → evaluator 测试应变红
    workdir = make_workspace(FIXTURE_DIR, str(patch_file))
    try:
        r1 = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_evaluator.py", "-q"],
            cwd=workdir, capture_output=True, text=True,
        )
        assert r1.returncode != 0, f"打补丁后本应变红，实际输出：\n{r1.stdout}"
    finally:
        cleanup_workspace(workdir)


def test_make_workspace_bad_patch_raises_sandboxerror(tmp_path):
    # 上下文对不上的补丁：git apply 在 evaluator.py 里找不到这段内容 → 应用失败
    bad_patch = (
        "--- a/evaluator.py\n"
        "+++ b/evaluator.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-NONEXISTENT_CONTEXT_LINE_ZZZ\n"
        "+replacement\n"
    )
    patch_file = tmp_path / "bad.patch"
    patch_file.write_text(bad_patch, encoding="utf-8")

    with pytest.raises(SandboxError) as excinfo:
        make_workspace(FIXTURE_DIR, str(patch_file))
    # message 里应带上 git 的 stderr（含出错文件名），供 harness 排障
    assert "evaluator.py" in str(excinfo.value)

def test_make_workspace_isolation_between_calls():
    try:
        workdir_1 = make_workspace(FIXTURE_DIR)
        workdir_2 = make_workspace(FIXTURE_DIR)
        assert workdir_1 != workdir_2
    finally:
        cleanup_workspace(workdir_1)
        cleanup_workspace(workdir_2)

def test_cleanup_workspace_removes_dir():
    workdir = make_workspace(FIXTURE_DIR)
    cleanup_workspace(workdir)
    assert os.path.exists(workdir) == False

def test_cleanup_workspace_swallows_errors(monkeypatch):
    cleanup_workspace("/any/path")  


def test_task_sandbox_yields_and_cleans_up():
    with task_sandbox(FIXTURE_DIR) as workdir:
        assert os.path.isabs(workdir)
        assert workdir == os.path.realpath(workdir)
        assert os.path.isdir(workdir)
    assert os.path.exists(workdir) == False

def test_task_sandbox_cleans_up_on_exception():
    with pytest.raises(SandboxError):
        with task_sandbox(FIXTURE_DIR) as workdir:
            assert os.path.isdir(workdir)
            raise SandboxError("模拟出错")
    assert os.path.exists(workdir) == False
