"""评测 harness 主体（DESIGN §10）。

本模块是 fixpoint 的裁判 + 记分系统，做四件事：
  ① 遍历任务集，为每个任务准备干净隔离副本并打 `break.patch`；
  ② 在副本上跑一遍 agent 主循环，全程计步 / token / 成本 / 耗时；
  ③ **独立复跑 pytest** 判定 solved（目标测试全绿 **且** 无回归）；
  ④ 汇总出记分卡 `scorecard.md`（每任务明细 + 汇总 + 消融对比表），并落一份
     机器可读 JSON（`eval/results/<label>.json`）供复现与再渲染。

铁律：**成败只由 harness 独立复跑 pytest 判定，绝不信任模型任何自述。**
判定用的 pytest 复跑（`run_pytest` + `judge`）与 agent 工具里的 `run_tests`
是两码事——后者只给模型看红绿；前者是评测方另起 pytest 子进程独立裁决。

── 消费的接口（本模块只消费，不实现；见 §10.1）─────────────────────────────
  · agent 主循环（agent/loop.py，§8）：
      result = run_agent(workdir=<str>, task=<str>, config=<Config>)
      读 result.num_steps / total_input_tokens / total_output_tokens /
      total_cost_usd / stop_reason ∈ {model_stop, max_steps, budget_exceeded, error}。
      不读 result.usage dict、不把 result.steps 当整数、不自算成本。
  · 沙箱（agent/sandbox.py，§5）：
      make_workspace(fixture_dir, patch_path) -> str /
      cleanup_workspace(workdir) / task_sandbox(...)。
      harness 不重复实现 copytree + git apply，一律经这些原语。
  · config（agent/config.py，§8）：按名引用 config.model / enable_retrieval /
      self_correction / max_steps / cost_budget_usd / run_tests_timeout_s /
      judge_timeout_s / price_per_mtok。复判成本直接读 result.total_cost_usd。

── 指标的精确定义（§10.5；聚合分母除非特别说明均为参与评测的任务总数 n_tasks，
   含未解决、含 status != "ok" 的任务——不解决 / 超预算本身就是要度量的信号）──
  · solve_rate     = n_solved / n_tasks，n_solved = Σ solved
  · pass@1         = solve_rate（默认 n_attempts=1 时二者恒等；n_attempts>1 时
                     取每任务多次尝试成功率均值的无偏估计）
  · avg_steps      = Σ steps / n_tasks（一步 = 主循环一轮工具调用往返）；
                     可选附列 avg_steps_solved（仅对 solved 求均值）
  · avg_tokens     = Σ tokens / n_tasks，tokens = input + output
  · avg_cost_usd   = Σ cost_usd / n_tasks；另给 total_cost_usd = Σ cost_usd
  · avg_wall_s     = Σ wall_s / n_tasks（仅 agent 主循环墙钟）
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Optional, Tuple

# —— 消费的接口（§10.1，本模块只调用不实现）——
from agent.config import Config
from agent.sandbox import make_workspace, cleanup_workspace, task_sandbox
from agent.loop import run_agent

# judge 复跑禁写 .pyc：避免"打补丁→跑→还原"同秒内 pyc 按秒失效误用旧字节码
_PYENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


# ─────────────────────────────────────────────────────────────────────────────
# Task 对象（§10.2，由本模块构造）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """一道题的运行期表示（DESIGN §10.2）。

    `task.json`（§9）只定义字段 schema；**把 task.json + 目录解析成 Task 对象是
    本模块 `discover_tasks` 的职责**。前五个字段来自 json，后两个是派生字段。

    字段：
      id            任务唯一 id，等于任务目录名（如 "001_mul_precedence"）
      title         一行英文摘要，进记分卡
      kind          题型枚举，只能是 "fix_bug" 或 "implement_stub"
      description   交给 agent 的自然语言提示（英文，只讲症状与目标、不泄露修法）
      target_tests  非空 list[str]；每项是相对 fixture 根的 pytest node id，
                    形如 "tests/<file>::<func>"（与 run_pytest 返回的 key 逐字符可比）
      dir           派生：任务目录的绝对路径
      break_patch   派生：<dir>/break.patch 的绝对路径，供 make_workspace 打补丁
    """

    id: str
    title: str
    kind: str
    description: str
    target_tests: List[str]
    dir: str
    break_patch: str


# ─────────────────────────────────────────────────────────────────────────────
# 任务发现与工作区准备
# ─────────────────────────────────────────────────────────────────────────────

def discover_tasks(tasks_dir: str) -> List[Task]:
    """扫描任务目录，解析出稳定排序的 Task 列表（§10.4 / §10.2）。

    行为：
      · 扫描 `tasks_dir` 下形如 `NNN_*/` 的子目录，把每个的 `task.json` 解析为
        Task（并构造 `dir` = 任务目录绝对路径、`break_patch` = <dir>/break.patch
        的绝对路径两个派生字段）。
      · **跳过 `fixture/`**（那是纯净基座，不是任务）。
      · 按 `task.id` 字典序**稳定排序**后返回（保证遍历顺序可复现，见 §10.8）。

    边界：
      · 某任务缺 `task.json`、JSON 解析失败、或字段非法（缺字段 /
        kind 不在 {fix_bug, implement_stub} / target_tests 为空或格式不对）→
        打印清晰错误并**跳过该任务**（不让一个坏任务毁掉整轮），被跳过的数量
        应能在最终汇总里以计数体现。

    参数：
      tasks_dir  任务集根目录（内含 fixture/ 与若干 NNN_*/），绝对或相对皆可。

    返回：
      list[Task]，按 id 升序。
    """
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        if name == "fixture" or not os.path.isdir(d) or not name[:1].isdigit():
            continue
        tj = os.path.join(d, "task.json")
        try:
            with open(tj, encoding="utf-8") as f:
                meta = json.load(f)
            for k in ("id", "title", "kind", "description", "target_tests"):
                if k not in meta:
                    raise ValueError(f"缺字段 {k}")
            if meta["kind"] not in ("fix_bug", "implement_stub"):
                raise ValueError(f"kind 非法：{meta['kind']}")
            if not meta["target_tests"]:
                raise ValueError("target_tests 为空")
            tasks.append(Task(
                id=meta["id"], title=meta["title"], kind=meta["kind"],
                description=meta["description"], target_tests=list(meta["target_tests"]),
                dir=os.path.abspath(d),
                break_patch=os.path.abspath(os.path.join(d, "break.patch")),
            ))
        except Exception as e:
            print(f"[discover_tasks] 跳过坏任务 {name}：{e}", file=sys.stderr)
    return sorted(tasks, key=lambda t: t.id)


def prepare_workspace(task: Task, dest_root: Optional[str] = None) -> str:
    """为一道题准备打好 break.patch 的隔离工作副本（§10.4）。

    契约：本函数是 `sandbox.make_workspace` 的**薄包装**——
        return make_workspace(fixture_dir, task.break_patch)
    （`task.break_patch` 已是绝对路径。）**不再自己 copytree + git apply**，
    沙箱是这两步的 owner（§5 / §9.6）；副本天然与纯净 fixture 及其它任务隔离。
    `fixture_dir` 由 harness 侧确定（通常是 `<tasks_dir>/fixture`）。`dest_root`
    透传给沙箱作为副本落点根（缺省 None → 沙箱用系统临时目录 tempfile.mkdtemp）。

    边界：
      · `make_workspace` 抛 `SandboxError`（补丁打不干净）→ **上抛**，由
        `run_one_task` 捕获落 status="patch_failed"。本函数不吞异常。

    返回：
      workspace 根目录的绝对路径（顶层就是 parser.py / tokenizer.py …，及 tests/）。
    """
    fixture_dir = os.path.join(os.path.dirname(task.dir), "fixture")
    return make_workspace(fixture_dir, task.break_patch)


# ─────────────────────────────────────────────────────────────────────────────
# 基线采集与 pytest 复跑
# ─────────────────────────────────────────────────────────────────────────────

def capture_baseline(fixture_dir: str) -> Dict[str, str]:
    """在纯净 fixture 的临时副本上采一次全量 pytest 基线（§10.4）。

    契约：
      · **不在 `tasks/fixture/` 原地跑**（会生成 __pycache__/*.pyc 污染纯净目录、
        与 §5.4「逐字节不变」验收冲突）——用沙箱在副本上采：
            with task_sandbox(fixture_dir, patch_path=None) as wd:
                return run_pytest(wd, <judge_timeout_s>)
        （patch_path=None 表示只复制、不打补丁，即纯净态。）
      · 返回 {node_id: outcome}，outcome ∈ {"passed","failed","error","skipped"}。
        因纯净态全绿（§4.7 不变量），正常所有值为 "passed"。
      · 整个任务集共用同一套测试文件，故**全程只需抓一次基线并缓存复用**（由
        `run_bench` 在开跑前调用一次，再把结果传给每个 `run_one_task`）。

    返回：
      dict[str, str]，键为 node id（形如 "tests/test_parser.py::test_x"）。
    """
    with task_sandbox(fixture_dir, None) as wd:
        return run_pytest(wd, 60)


def run_pytest(workspace: str, timeout_s: int) -> Dict[str, str]:
    """在给定工作副本上独立复跑全量 pytest，返回逐用例判定（§10.4）。

    契约：
      · 以 `workspace` 为 cwd，用 `sys.executable -m pytest`（与 §6 工具一致，
        避免取到错误的解释器）起**独立子进程**跑 `tests/`。
      · 加 `-p no:cacheprovider` 避免写 `.pytest_cache` 污染、保证判定无副作用。
      · 用内置 `--junitxml` 产出 XML 后解析为 {node_id: outcome}。
      · **node id 还原规则（务必照做，否则 judge 永远匹配不上）**：junitxml 的
        `<testcase>` 只给 `classname`（点分、无 .py）+ `name`（此版 pytest 无 `file` 属性）。
        **用 `classname` 点分转 `/` 再补 `.py`，与 `name` 拼 f"{file}::{name}"**
        （classname "tests.test_parser" → file "tests/test_parser.py"，name 如 "test_precedence_mul_over_add"；
        参数化用例的 name 已带 "[param]" 后缀，**原样保留**）。这样还原出的 key
        与 `target_tests` 的字符串**逐字符可比**。
        （若嫌 junitxml 繁琐，可改用 pytest json 报告插件——二选一，但契约里写死
        采用哪种；本项目默认 junitxml + file/name 拼接。）
      · outcome ∈ {"passed","failed","error","skipped"}。

    边界：
      · 子进程超过 `timeout_s` → 抛超时异常（上层 `run_one_task` 落
        status="judge_timeout"）。

    参数：
      workspace  待判定的工作副本绝对路径（复判前应已 restore_pristine_tests）。
      timeout_s  子进程墙钟上限（judge 场景取 config.judge_timeout_s）。

    返回：
      dict[str, str]，键为逐字符可比的 node id。
    """
    fd, xml_path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-p", "no:cacheprovider",
             "--junitxml", xml_path, "-q"],
            cwd=workspace, capture_output=True, text=True, timeout=timeout_s, env=_PYENV,
        )
        outcomes: Dict[str, str] = {}
        for tc in ET.parse(xml_path).iter("testcase"):
            # junit 只给 classname（点分、无 .py）+ name；还原成 node id
            # "tests.test_parser" + "test_x" -> "tests/test_parser.py::test_x"
            classname, name = tc.get("classname"), tc.get("name")
            if not classname or not name:
                continue
            file = classname.replace(".", "/") + ".py"
            if tc.find("failure") is not None:
                oc = "failed"
            elif tc.find("error") is not None:
                oc = "error"
            elif tc.find("skipped") is not None:
                oc = "skipped"
            else:
                oc = "passed"
            outcomes[f"{file}::{name}"] = oc
        return outcomes
    finally:
        try:
            os.remove(xml_path)
        except OSError:
            pass


def restore_pristine_tests(workspace: str, fixture_dir: str) -> None:
    """反作弊：用纯净测试覆盖工作副本里的测试（§10.4 / §9.6 第 5 步①）。

    契约：
      · 用纯净 `fixture_dir/tests/` + `fixture_dir/conftest.py` **覆盖**
        `workspace/tests/` 与 `workspace/conftest.py`。
      · **调用时机**：在 `run_agent` 之后、`run_pytest` 之前。这样即便 agent 在
        求解期偷偷改了测试文件（测试驱动求解本就允许**读**测试理解期望），最终
        判分用的仍是**受保护的原版测试**，作弊影响不了裁决。
      · 只覆盖测试与 conftest，**不动**被修的库模块（parser.py 等）——那正是要
        判定的 agent 产物。

    返回：None（原地覆盖 workspace 内文件）。
    """
    shutil.copytree(os.path.join(fixture_dir, "tests"),
                    os.path.join(workspace, "tests"), dirs_exist_ok=True)
    shutil.copy(os.path.join(fixture_dir, "conftest.py"),
                os.path.join(workspace, "conftest.py"))


# ─────────────────────────────────────────────────────────────────────────────
# 判分（只依赖 pytest 结果与基线，与模型自述无关）
# ─────────────────────────────────────────────────────────────────────────────

def judge(post: Dict[str, str], baseline: Dict[str, str],
          target_tests: List[str]) -> Tuple[bool, List[str]]:
    """依据回归规则裁决 solved 并列出回归（§10.4 / §9.6）。

    规格（**只给文字规格，不给集合代码**——判分核心，正是学习者该从回归规则
    自行推导实现的部分）：
      · 设 passing_now = `post` 中 outcome 为 "passed" 的 node id 集合；
        baseline_ok = `baseline` 中 "passed" 的 node id 集合。
      · regressions = baseline_ok 中**不在** passing_now 的 node id（原本绿、现在
        红），排序后返回。
      · solved ⟺ **`target_tests` 每一项都在 passing_now（全通过）** 且
        **regressions 为空**。
        （因基线全绿，「无回归」等价于「复判必须全绿」；`target_tests` 子句虽被
        「全绿」蕴含，仍单列以①记录题目意图②给 agent 的 run_tests 提供聚焦点
        ③配合测试还原堵住「靠删/跳过测试凑全绿」的作弊——被删的目标测试不会算
        作 pass。）

    边界：
      · 目标测试在 `post` 中**缺失**（node id 拼错 / 被删）→ 视为未通过 →
        solved=False。

    性质：本函数**只依赖 pytest 结果与基线**，与模型、与 agent 说了什么完全无关
    ——这是「绝不信任自述」的落点，也是评分可复现的根（§10.8）。

    返回：
      (solved: bool, regressions: list[str])，regressions 已排序。
    """
    passing_now = {k for k, v in post.items() if v == "passed"}
    baseline_ok = {k for k, v in baseline.items() if v == "passed"}
    regressions = sorted(baseline_ok - passing_now)
    solved = all(t in passing_now for t in target_tests) and not regressions
    return solved, regressions


# ─────────────────────────────────────────────────────────────────────────────
# 单任务生命周期与整轮 bench
# ─────────────────────────────────────────────────────────────────────────────

def run_one_task(task: Task, config: Config, baseline: Dict[str, str]) -> dict:
    """跑完一道题的完整生命周期，返回一个 TaskResult dict（§10.4 / §10.3）。

    流程（§9.6 六步中第 1/2/5 步在此、第 4 步是 agent）：
      prepare_workspace(task)                       # 复制 + 打补丁（经沙箱）
      t0 = perf_counter()
      result = run_agent(workdir, task.description, config)   # agent 求解
      wall_s = perf_counter() - t0
      restore_pristine_tests(workdir, fixture_dir)  # 反作弊还原
      post = run_pytest(workdir, config.judge_timeout_s)      # 独立复判
      solved, regressions = judge(post, baseline, task.target_tests)
      cost_usd = result.total_cost_usd              # 直接读，不在 eval 侧重算
      → 组装 TaskResult → cleanup_workspace(workdir)（--keep 时保留供调试）

    计时口径：`wall_s` **只计 agent 主循环**，不含准备副本与判定复跑（度量的是
    agent 本身）。

    边界（**绝不中断整轮 bench**）：
      · prepare_workspace 抛 SandboxError → status="patch_failed"
      · run_agent 抛异常          → status="agent_error"
      · run_pytest 超时           → status="judge_timeout"
      非 "ok" 时 solved 恒 False，但仍尽量记已获得的 steps / tokens / wall_s。
      命中 max_steps / 预算时 run_agent 正常返回对应 stop_reason，判定照跑
      （agent 可能已部分修好，成败仍以 pytest 为准）。

    TaskResult 格式（一个 dict，§10.3）：
      {
        "task_id": "003_multidigit_number",
        "status": "ok",                # ∈ {"ok","patch_failed","agent_error","judge_timeout"}
        "solved": true,
        "steps": 6,                    # = result.num_steps
        "input_tokens": 41230,
        "output_tokens": 3120,
        "tokens": 44350,               # = input_tokens + output_tokens
        "cost_usd": 0.284,             # = result.total_cost_usd（缺表按 0）
        "wall_s": 38.4,                # 仅 agent 主循环墙钟
        "stop_reason": "model_stop",   # loop 的规范值
        "target_tests": ["tests/test_tokenizer.py::test_multi_digit_number"],
        "regressions": []              # 基线绿、复判红的 node id 列表
      }
    """
    fixture_dir = os.path.join(os.path.dirname(task.dir), "fixture")
    tr = {
        "task_id": task.id, "status": "ok", "solved": False,
        "steps": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0,
        "cost_usd": 0.0, "wall_s": 0.0, "stop_reason": None,
        "target_tests": task.target_tests, "regressions": [],
    }
    try:
        workdir = prepare_workspace(task)
    except Exception as e:
        tr["status"] = "patch_failed"
        print(f"[run_one_task] {task.id} patch_failed：{e}", file=sys.stderr)
        return tr

    try:
        t0 = perf_counter()
        try:
            result = run_agent(workdir, task.description, config)
        except Exception as e:
            tr["status"], tr["wall_s"] = "agent_error", perf_counter() - t0
            print(f"[run_one_task] {task.id} agent_error：{e}", file=sys.stderr)
            return tr
        tr["wall_s"] = perf_counter() - t0
        tr["steps"] = result.num_steps
        tr["input_tokens"] = result.total_input_tokens
        tr["output_tokens"] = result.total_output_tokens
        tr["tokens"] = result.total_input_tokens + result.total_output_tokens
        tr["cost_usd"] = result.total_cost_usd or 0.0
        tr["stop_reason"] = result.stop_reason

        restore_pristine_tests(workdir, fixture_dir)
        try:
            post = run_pytest(workdir, config.judge_timeout_s)
        except subprocess.TimeoutExpired:
            tr["status"] = "judge_timeout"
            return tr
        tr["solved"], tr["regressions"] = judge(post, baseline, task.target_tests)
        return tr
    finally:
        cleanup_workspace(workdir)


def run_bench(tasks_dir: str, config: Config, label: str) -> dict:
    """跑完整个任务集，落盘并返回一个 BenchResult dict（§10.4 / §10.3）。

    流程：
      · fixture_dir = <tasks_dir>/fixture
      · capture_baseline(fixture_dir) 一次（全程缓存复用）
      · 顺序遍历 discover_tasks(tasks_dir)，每个任务跑 run_one_task(task, config,
        baseline)
      · 按 §10.5 定义聚合出 `summary`
      · 拍 `config_snapshot`（把当时旋钮原样拍进结果，**无 temperature 字段**——
        Opus 4.8 已移除该采样参数、传入会 400；可复现性靠 fixture 确定性而非采样
        固定，见 §10.8）与 `repo_commit`（git rev-parse --short HEAD）
      · 落盘 `eval/results/<label>.json` 并返回该 dict。`label` 已存在则**覆盖**。

    BenchResult 格式（§10.3）：
      {
        "label": "baseline",
        "timestamp": "2026-07-14T10:00:00",     # ISO
        "repo_commit": "c2d4b03",
        "config_snapshot": {                     # 复现与消融的关键；无 temperature
          "model": "anthropic/claude-opus-4.8",
          "enable_retrieval": false,
          "self_correction": false,
          "max_steps": 30,
          "cost_budget_usd": 0.50,
          "run_tests_timeout_s": 60,
          "judge_timeout_s": 60
        },
        "tasks": [ /* TaskResult ... */ ],
        "summary": {
          "n_tasks": 12, "n_solved": 9,
          "solve_rate": 0.75, "pass_at_1": 0.75,
          "avg_steps": 8.3, "avg_tokens": 51200,
          "avg_cost_usd": 0.24, "total_cost_usd": 2.88,
          "avg_wall_s": 42.1
        }
      }

    指标口径：见本模块顶部 docstring 的「指标的精确定义」（§10.5）；分母恒为
    n_tasks（含未解决 / status != "ok"）。
    """
    fixture_dir = os.path.join(tasks_dir, "fixture")
    baseline = capture_baseline(fixture_dir)
    tasks = discover_tasks(tasks_dir)

    task_results = []
    for t in tasks:
        print(f"[bench] ▶ {t.id} …", file=sys.stderr)
        tr = run_one_task(t, config, baseline)
        print(f"[bench]   {t.id}: {'SOLVED' if tr['solved'] else tr['status']} "
              f"(steps={tr['steps']}, ${tr['cost_usd']:.4f})", file=sys.stderr)
        task_results.append(tr)

    n = len(task_results)
    n_solved = sum(1 for tr in task_results if tr["solved"])

    def avg(key):
        return (sum(tr[key] for tr in task_results) / n) if n else 0.0

    summary = {
        "n_tasks": n, "n_solved": n_solved,
        "solve_rate": (n_solved / n) if n else 0.0,
        "pass_at_1": (n_solved / n) if n else 0.0,
        "avg_steps": avg("steps"), "avg_tokens": avg("tokens"),
        "avg_cost_usd": avg("cost_usd"),
        "total_cost_usd": sum(tr["cost_usd"] for tr in task_results),
        "avg_wall_s": avg("wall_s"),
    }

    root = os.path.dirname(os.path.abspath(tasks_dir.rstrip(os.sep)))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=root, capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    bench = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_commit": commit,
        "config_snapshot": {
            "model": config.model,
            "enable_retrieval": config.enable_retrieval,
            "self_correction": config.self_correction,
            "max_steps": config.max_steps,
            "cost_budget_usd": config.cost_budget_usd,
            "run_tests_timeout_s": config.run_tests_timeout_s,
            "judge_timeout_s": config.judge_timeout_s,
        },
        "tasks": task_results,
        "summary": summary,
    }

    results_dir = os.path.join(root, "eval", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
    return bench


def render_scorecard(results: List[dict], out_path: str = "eval/scorecard.md") -> None:
    """把一份或多份 BenchResult 渲染成 Markdown 记分卡（§10.4 / §10.6 / §10.7）。

    契约：
      · 以 `results[0]`（主条件，一般是 "baseline"）出：
          ① 运行元信息头（复现凭据，直接来自 config_snapshot；**无 temperature 行**）：
             日期、repo commit、model id、retrieval on/off、self-correction on/off、
             护栏（MAX_STEPS / cost budget / run_tests timeout / judge timeout）、
             依赖版本。
          ② 每任务明细表：| task | solved | steps | tokens | cost($) | wall(s) |
             stop_reason | regressions |；status != "ok" 的任务用 ⚠️ 标注。
          ③ 汇总块（Summary）：solve rate / pass@1、avg steps（可附 solved-only）、
             avg tokens、avg cost、total cost、avg wall。
      · 再用**全部** `results` 出消融对比表（§10.7）：每条件一行，列 solve@1 /
        avg steps / avg tokens / avg cost / total / avg wall，括号内为相对
        baseline 的增量。可选脚注声明 n_attempts 与「小任务集 + 采样随机 → 小差异
        可能是噪声」的诚实性说明（§10.7）。
      · 写到 `out_path`（默认 eval/scorecard.md，入库的展示产物）。

    可 `--render-only`（在 cli 层）直接读 `eval/results/*.json` 重建记分卡而不重跑
    ——即本函数以已落盘的 BenchResult 为唯一输入，无损重渲染出同一记分卡（§10.9）。

    返回：None（写文件）。
    """
    if not results:
        return
    primary = results[0]
    cs = primary["config_snapshot"]
    sm = primary["summary"]
    L = []
    L.append("# fixpoint scorecard\n")
    L.append(f"- date: `{primary['timestamp']}`  ·  commit: `{primary['repo_commit']}`")
    L.append(f"- model: `{cs['model']}`  ·  retrieval: `{cs['enable_retrieval']}`  ·  "
             f"self-correction: `{cs['self_correction']}`")
    L.append(f"- guardrails: max_steps=`{cs['max_steps']}`, cost_budget=`${cs['cost_budget_usd']}`, "
             f"run_tests_timeout=`{cs['run_tests_timeout_s']}s`, judge_timeout=`{cs['judge_timeout_s']}s`")
    L.append("")

    # 每任务明细
    L.append(f"## Per-task ({primary['label']})\n")
    L.append("| task | solved | steps | tokens | cost($) | wall(s) | stop_reason | regressions |")
    L.append("|---|:--:|--:|--:|--:|--:|---|---|")
    for tr in primary["tasks"]:
        flag = "✅" if tr["solved"] else ("⚠️ " + tr["status"] if tr["status"] != "ok" else "❌")
        regs = ", ".join(t.split("::")[-1] for t in tr["regressions"]) or "-"
        L.append(f"| {tr['task_id']} | {flag} | {tr['steps']} | {tr['tokens']} | "
                 f"{tr['cost_usd']:.4f} | {tr['wall_s']:.1f} | {tr['stop_reason'] or '-'} | {regs} |")
    L.append("")

    # 汇总
    L.append("## Summary\n")
    L.append(f"- **pass@1 = {sm['n_solved']}/{sm['n_tasks']} = {sm['solve_rate']:.0%}**")
    L.append(f"- avg steps: {sm['avg_steps']:.1f}  ·  avg tokens: {sm['avg_tokens']:.0f}  ·  "
             f"avg cost: ${sm['avg_cost_usd']:.4f}  ·  total cost: ${sm['total_cost_usd']:.2f}  ·  "
             f"avg wall: {sm['avg_wall_s']:.1f}s")
    L.append("")

    # 消融对比（全部 results）
    if len(results) > 1:
        L.append("## Ablations\n")
        L.append("| variant | model | retrieval | self-corr | pass@1 | avg steps | avg cost($) | total($) |")
        L.append("|---|---|:--:|:--:|:--:|--:|--:|--:|")
        for r in results:
            c, s = r["config_snapshot"], r["summary"]
            L.append(f"| {r['label']} | {c['model'].split('/')[-1]} | {c['enable_retrieval']} | "
                     f"{c['self_correction']} | {s['solve_rate']:.0%} | {s['avg_steps']:.1f} | "
                     f"{s['avg_cost_usd']:.4f} | {s['total_cost_usd']:.2f} |")
        L.append("")
        L.append("> 小任务集 + 采样随机性下，条件间的小差异可能是噪声；n_attempts=1。")
        L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
