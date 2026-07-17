"""沙箱与路径封闭（DESIGN §5）。

本模块是 fixpoint 中「隔离目录的建/清/打补丁机制」与「路径封闭判定」的
**唯一 owner**。它向上（harness / loop / cli）交付「进出即建/清的隔离目录」，
向下（tools）交付「把路径关进目录的 ``resolve_in_workdir``」。

明确**不做**的事（归属其它章节）：
- 不实现任何具体工具（read/write/grep/run_tests 等，见 §6 ``agent/tools.py``）。
- 不跑 pytest、不做红绿判定（见 §10 ``eval/run_bench.py``）。
- 不管 token / 成本 / 超时预算（见 §7/§8）。

契约摘要（详见各函数 docstring）：
- ``resolve_in_workdir(workdir, user_path) -> str``：纯函数，把用户给的路径
  解析成落在 workdir 内的规范绝对路径；越界抛 ``PathEscape``。
- ``make_workspace`` / ``cleanup_workspace`` / ``task_sandbox``：一份干净、独立、
  可丢弃的代码副本的生命周期管理；搭建/打补丁失败抛 ``SandboxError``。

依赖：标准库 ``os`` / ``shutil`` / ``tempfile`` / ``subprocess`` / ``contextlib`` /
``logging``；外部命令 ``git``（仅用于 ``git apply``）。无第三方依赖。
"""

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 5.1 异常
# --------------------------------------------------------------------------- #
class SandboxError(Exception):
    """工作区搭建 / 清理 / 打补丁类失败。

    由 ``make_workspace``（含内部的 ``git apply``）在无法建立一个可用的坏状态
    工作区时抛出，message 应携带底层原因（例如 git 的 stderr、copytree 报错）。

    语义定位：这类失败是**任务加载错误（task-load error）**，harness 必须把它
    与「agent 没把题解出来」（正常的 fail）区分开——前者记为 task-load error，
    不计入 agent 的成败统计。
    """


class PathEscape(SandboxError):
    """路径越界：``resolve_in_workdir`` 判定用户路径逃出 workdir 时抛出。

    基类为 ``SandboxError``（故对 ``SandboxError`` 的捕获同样能兜住它）。
    message 应同时包含**原始输入**与**解析落点**，便于排障。

    职责边界：sandbox 只负责做出「越界」这一判断并抛此异常；**工具层负责捕获
    它并转成以「错误：」开头的字符串**回给模型（越界不抛给模型、只回错误串，
    是已锁定的工具契约，见 §6）。
    """


# --------------------------------------------------------------------------- #
# 5.2 路径封闭
# --------------------------------------------------------------------------- #
def resolve_in_workdir(workdir: str, user_path: str) -> str:
    """把 ``user_path`` 解析为落在 ``workdir`` 内的**规范绝对路径**并返回。

    这是路径封闭的安全核心：纯函数、无副作用、不读写文件内容，因而易于单测。
    它同时挡住 ``..`` 逃逸与符号链接逃逸——靠「先规范化、再判包含」这一招。

    参数
    ----
    workdir:
        隔离工作目录。可能是非规范路径（例如 macOS 上的 ``/tmp/xxx``，其真身在
        ``/private/tmp/xxx``），本函数会先对它做 ``realpath`` 再参与判断。
    user_path:
        模型 / 调用方给出的路径，可为相对或绝对，可含 ``..`` 或途径符号链接。
        ``""`` 与 ``"."`` 均表示 workdir 本身。

    行为（三步，务必照做）
    ----------------------
    1. ``root = os.path.realpath(workdir)``。**macOS 上必不可少**：``/tmp`` 是指向
       ``/private/tmp`` 的软链，``tempfile`` 给的临时目录多在 ``/tmp`` 下；不先规范化
       root，包含判断会因 ``/tmp/x`` 与 ``/private/tmp/x`` 对不上而全线误判。
    2. ``user_path`` 为绝对路径则直接作候选，否则拼到 ``root`` 后面；再对候选做一次
       ``os.path.realpath``（一次性解掉 ``..`` 与途中所有符号链接）。目标即便尚不存在
       也照常规范化、**不报错**（「存在与否」不是路径封闭的职责，交给具体工具处理）。
    3. 判定包含：候选**等于** ``root``，或以 ``root + os.sep`` 开头。任一成立即通过。
       ``+ os.sep`` 不可省——否则 ``/work`` 会把 ``/work-evil/x`` 误判为「在 /work 内」。

    返回
    ----
    通过 → 返回规范绝对路径字符串（可直接交给 ``open`` / ``os.listdir`` 等）。

    边界速查
    --------
    - ``"evaluator.py"`` → ``<root>/evaluator.py``
    - ``"tests/test_parser.py"`` → ``<root>/tests/test_parser.py``
    - ``""`` / ``"."`` → ``<root>``（workdir 自身，放行）
    - ``"sub/../parser.py"`` → ``<root>/parser.py``（规范化后仍在内，放行）
    - ``"../secret.txt"`` → ``PathEscape``
    - ``"/etc/passwd"`` → ``PathEscape``（绝对路径在外）
    - ``<root>`` 内某文件的绝对路径 → 放行
    - 指向目录外的软链 ``evil`` + ``"evil/x"`` → ``PathEscape``（realpath 落到外部）
    - 指向目录内的软链 → 放行（realpath 落回内部）
    - 尚不存在的 ``"new_file.py"`` → ``<root>/new_file.py``（不报错）

    异常
    ----
    PathEscape:
        规范化后的候选路径落在 ``root`` 之外时抛出，message 含原始输入与解析落点。
    """
    root = os.path.realpath(workdir)
    candidate = user_path if os.path.isabs(user_path) else os.path.join(root, user_path)
    candidate = os.path.realpath(candidate)

    if candidate == root or candidate.startswith(root + os.sep):
        return candidate
    else:
        raise PathEscape(f"{user_path}解析后的绝对路径为{candidate}，不在工作区{workdir}内")


# --------------------------------------------------------------------------- #
# 5.3 工作区生命周期
# --------------------------------------------------------------------------- #
def make_workspace(fixture_dir: str, patch_path: Optional[str] = None) -> str:
    """建一份干净、独立、可丢弃的 fixture 副本，可选地打上 break.patch。

    参数
    ----
    fixture_dir:
        纯净基座目录（唯一真相源 ``tasks/fixture/``）。**只读**，绝不原地修改。
    patch_path:
        可选。break.patch 的路径，**必须是绝对路径**——``git`` 相对进程 cwd 找补丁，
        而应用时 cwd 会被设成新建的 workdir，相对路径会解析失败。为 ``None`` 时跳过
        打补丁一步（供 harness 采「基线」用，见 §10）。

    行为
    ----
    1. ``tempfile.mkdtemp(prefix="fixpoint_task_")`` 在系统 tempdir 下建独立临时目录
       （随机名保证多任务顺序 / 并发互不干扰）；``workdir = os.path.realpath(该目录)``。
    2. ``shutil.copytree(fixture_dir, workdir,
       ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)``
       ——跳过陈旧字节码，避免干扰后续 pytest 观察源码改动（``dirs_exist_ok`` 因
       mkdtemp 已建好目录而必需，Python 3.8+ 可用）。拷完 workdir 内应有
       ``tokenizer.py / parser.py / evaluator.py / errors.py / conftest.py / tests/``。
    3. 若 ``patch_path`` 非 ``None``：以 ``cwd=workdir`` 运行
       ``git apply -p1 <patch_path>`` 制造坏状态（``-p1`` 因补丁由作者从 fixture 根
       用 ``git diff`` 生成，带 ``a/…`` / ``b/…`` 前缀）。``git apply`` 不要求目标是
       git 仓库，对普通工作树即可打补丁。

    返回
    ----
    新建 workdir 的**规范绝对路径**。此刻工作区处于「坏」状态（若打了补丁，目标测试应红）。

    异常
    ----
    SandboxError:
        copytree 失败，或 ``git apply`` 返回码非零（例如补丁上下文对不上）时抛出，
        message 须带底层原因（尤其把 git 的 stderr 带进来），供 harness 与「未解出」区分。

    备注
    ----
    调用方负责在用完后调用 ``cleanup_workspace``（或改用 ``task_sandbox`` 上下文管理器
    自动清理）。``eval`` 需跨「prepare→run→judge」持有 workdir，故用本函数 + 显式清理。
    """
    temp_dir = tempfile.mkdtemp(prefix="fixpoint_task_")   # 新建临时目录
    work_dir = os.path.realpath(temp_dir)   # 转为绝对路径

    try:
        shutil.copytree(fixture_dir, work_dir,
       ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)    # 拷贝，跳过字节缓存
    except Exception as e:
        raise SandboxError(f"工作树拷贝失败：{fixture_dir} -> {work_dir}: {e}") from e 

    # 手动打 bug 制造“坏”状态
    if patch_path is not None:
        result = subprocess.run(
            ["git", "apply", "-p1", patch_path],
            cwd=work_dir,              
            capture_output=True,  
            text=True,                   
        )
        if result.returncode != 0:
            cleanup_workspace(work_dir)
            raise SandboxError(f"git apply 失败：{result.stderr}")

    return work_dir

def cleanup_workspace(workdir: str) -> None:
    """尽力删除一个由 ``make_workspace`` 建出的工作区。

    参数
    ----
    workdir:
        待删除的工作目录（``make_workspace`` 的返回值）。

    行为
    ----
    ``shutil.rmtree(workdir)`` **尽力**删除。删除失败**只记日志、不抛异常**——以免
    清理阶段的失败掩盖掉任务本身的结果（成/败判定优先于清理是否干净）。

    返回
    ----
    None。
    """
    try:
        shutil.rmtree(workdir)
    except Exception as e:
        logger.warning(f"清理工作区{workdir}失败：{e}")


@contextmanager
def task_sandbox(
    fixture_dir: str, patch_path: Optional[str] = None
) -> Iterator[str]:
    """工作区生命周期的上下文管理器：进入即建、退出必清。

    参数
    ----
    fixture_dir:
        纯净基座目录，转交给 ``make_workspace``。
    patch_path:
        可选 break.patch 绝对路径，转交给 ``make_workspace``；语义同上。

    行为
    ----
    ``make_workspace(fixture_dir, patch_path)`` → ``yield workdir``（此刻为「坏」状态，
    若打了补丁则目标测试应红）→ 在 ``finally`` 中 ``cleanup_workspace(workdir)``，
    无论 with 体正常结束、抛异常、还是撞护栏提前退出，都**保证**清理。

    产出
    ----
    workdir 的规范绝对路径（``with ... as workdir:``）。

    用途
    ----
    供 ``cli.py solve`` 与简单用法。``eval`` 因需跨「prepare→run→judge」持有 workdir，
    改用 ``make_workspace`` + 显式 ``cleanup_workspace``（``--keep`` 时跳过清理并打印
    路径供调试），不走本上下文管理器。

    异常
    ----
    SandboxError:
        建立阶段（copytree / git apply）失败时，由 ``make_workspace`` 透传抛出。
    """
    work_dir = make_workspace(fixture_dir, patch_path)
    try:
        yield work_dir
    finally:
        cleanup_workspace(work_dir)

