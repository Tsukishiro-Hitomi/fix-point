"""fixpoint 命令行入口（DESIGN §14 / §8.3 / §10）。

两个子命令：
  · `solve <task_id>` —— 单任务求解：从纯净 fixture 拉起隔离副本、打该任务的
    break.patch、跑一遍 agent 主循环（流式打印），供人观察「红 → 迭代 → 绿」。
    （行为属 loop 层 §8，本文件只做 CLI 装配。）
  · `bench [--label] [--tasks] [--keep] [--render-only]` —— 整轮评测：遍历任务集
    产出 `eval/scorecard.md`（解决率 / pass@1、平均步数 / token / 成本）。
    （行为属 eval 层 §10，本文件只做 CLI 装配。）

**启动接线（§14.4 / §8.1，务必按此顺序，实现在 `main`）**：
  1. `load_dotenv()` —— 在构造 Config / LLMClient **之前**把 `.env` 读进环境；
  2. `Config.from_env()` —— 在默认值之上叠加**非密钥**旋钮
     （env → 字段映射：`FIXPOINT_MODEL → model`、`MAX_STEPS → max_steps`、
     `RUN_TESTS_TIMEOUT → run_tests_timeout_s`；其余用 Config 默认）；
  3. 解析参数 → 分发到 `cmd_solve` / `cmd_bench`。

**密钥红线**：`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` 由 anthropic SDK 直接从
环境读取——本文件**不硬编码、不打印、不写日志、不进 Config**。

用法（与 §10.10 / §14.1 Quickstart 逐字一致）：
    python cli.py solve 001_mul_precedence
    python cli.py bench [--label NAME] [--tasks tasks/] [--keep] [--render-only]
"""

import argparse
import glob
import json
import os
import sys
from typing import Optional, Sequence

from dotenv import load_dotenv

from agent.config import Config
from agent.sandbox import task_sandbox
from agent.loop import run_agent
from eval.run_bench import discover_tasks, run_bench, render_scorecard


# 任务集根目录默认值（§9.1；discover_tasks 会跳过其中的 fixture/）。
DEFAULT_TASKS_DIR = "tasks"
# 记分卡默认落点（入库的展示产物，README 指向它，§14.2）。
DEFAULT_SCORECARD = "eval/scorecard.md"


def build_parser() -> argparse.ArgumentParser:
    """构造 solve / bench 两子命令的 argparse 解析器（声明式 CLI 面）。

    这是本文件唯一「写全」的部分——它只描述 CLI 的**形状**（子命令、位置参数、
    可选开关、默认值），不含任何求解 / 评测逻辑（那些在 `cmd_solve` / `cmd_bench`
    / `main`）。CLI 表面与 §10.10、§14.1 Quickstart 逐字对齐：

        solve <task_id>
        bench [--label NAME] [--tasks DIR] [--keep] [--render-only]

    返回：配置好子命令的 ArgumentParser（`args.command ∈ {"solve","bench"}`）。
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="fixpoint — a test-driven autonomous coding agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{solve,bench}")

    # —— solve ——（§8.3：入参是任务目录名/id，与 discover_tasks 一致）
    p_solve = sub.add_parser(
        "solve",
        help="Solve a single task: apply its break.patch to a pristine fixture copy "
             "and run the agent loop until it stops.",
    )
    p_solve.add_argument(
        "task_id",
        help="Task id = task directory name, e.g. 001_mul_precedence.",
    )
    p_solve.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_DIR,
        metavar="DIR",
        help="Task-set root containing fixture/ and NNN_*/ (default: %(default)s).",
    )

    # —— bench ——（§10.10）
    p_bench = sub.add_parser(
        "bench",
        help="Run the whole task set and write eval/scorecard.md.",
    )
    p_bench.add_argument(
        "--label",
        default="baseline",
        metavar="NAME",
        help="Run label; writes eval/results/<NAME>.json, overwriting if it exists "
             "(default: %(default)s).",
    )
    p_bench.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_DIR,
        metavar="DIR",
        help="Task-set root (default: %(default)s).",
    )
    p_bench.add_argument(
        "--keep",
        action="store_true",
        help="Keep each task's temp workspace for debugging instead of cleaning it up.",
    )
    p_bench.add_argument(
        "--render-only",
        action="store_true",
        help="Skip running; just rebuild eval/scorecard.md from existing "
             "eval/results/*.json (loss-lessly re-renders, incl. the ablation table).",
    )
    return parser


def cmd_solve(args: argparse.Namespace, config: Config) -> int:
    """执行 `solve` 子命令：单任务从红跑到绿（行为属 §8，本文件只装配）。

    契约（§8.3 / §12 ROADMAP M6）：
      · fixture_dir = <args.tasks>/fixture。
      · 用 `discover_tasks(args.tasks)` 找出 id == `args.task_id` 的 Task；找不到 →
        打印清晰错误、返回非零退出码。
      · 用**上下文管理器** `task_sandbox(fixture_dir, task.break_patch)` 拉起纯净
        隔离副本并打补丁（此刻目标测试应为红），`with` 退出时自动 cleanup（§5.3）：
            with task_sandbox(fixture_dir, task.break_patch) as workdir:
                result = run_agent(workdir, task.description, config)
      · agent 循环流式打印到终端（`config.stream=True` 时模型文本边生成边显示，
        长任务不再黑屏等待，§12 V6）。
      · 收尾打印 result 摘要（`num_steps` / tokens / `total_cost_usd` /
        `stop_reason` / `final_text`）。**不做 solved 判定**——solve 只给人看过程，
        判分是 bench / harness 的事（§8.4 有意不放 solved 字段）。

    参数：
      args    已解析的命名空间（含 `task_id`、`tasks`）。
      config  由 `main` 经 `Config.from_env()` 构造好的实例。

    返回：进程退出码（0 成功装配并跑完；非 0 表示任务未找到等 CLI 级错误）。
    """
    fixture_dir = os.path.join(args.tasks, "fixture")
    task = next((t for t in discover_tasks(args.tasks) if t.id == args.task_id), None)
    if task is None:
        print(f"错误：找不到任务 {args.task_id}（在 {args.tasks}/ 下）", file=sys.stderr)
        return 1

    config.stream = True  # V7：solve 开流式，模型文本边生成边显示
    print(f"▶ solve {task.id} · {task.title}\n")
    with task_sandbox(fixture_dir, task.break_patch) as workdir:
        result = run_agent(
            workdir, task.description, config,
            on_text=lambda t: print(t, end="", flush=True),  # 实时打印模型文本
        )
        print("\n\n—— 轨迹 ——")
        for s in result.steps:
            names = "、".join(tc.name for tc in s.tool_calls) or "（收尾）"
            print(f"  #{s.index}: {names}")

    print(f"\nstop_reason={result.stop_reason}  steps={result.num_steps}  "
          f"tokens={result.total_input_tokens}/{result.total_output_tokens}  "
          f"cost=${result.total_cost_usd:.4f}")
    if result.total_output_tokens == 0 and result.num_steps > 0:
        print("（注：流式下本网关不回传 output tokens；成本为下界，准确值见 `cli.py bench`）")
    if result.final_text.strip():
        print("summary:", result.final_text.strip())
    return 0


def cmd_bench(args: argparse.Namespace, config: Config) -> int:
    """执行 `bench` 子命令：整轮评测 + 渲染记分卡（行为属 §10，本文件只装配）。

    契约（§10.4 / §10.7 / §14.1）：
      · `--render-only`：**不重跑**，直接读 `eval/results/*.json`（每个是一份
        BenchResult），调 `render_scorecard(results, DEFAULT_SCORECARD)` 无损重建
        记分卡（含消融对比表）。
      · 否则：
          results = run_bench(args.tasks, config, args.label)   # 落 eval/results/<label>.json
          然后渲染记分卡：读回 `eval/results/*.json`（至少含本次 label），
          `render_scorecard([...], DEFAULT_SCORECARD)`。
        （消融工作流见 §10.7：每个条件相对 baseline 只翻一个 config 旋钮、各跑一次
        `bench --label ...`，最后 `bench --render-only` 汇出对比表。）
      · `--keep`：把每个任务的临时工作副本保留供调试（透传到 eval 侧的清理开关，
        默认清理，§10.4 / §10.9）。

    参数：
      args    已解析的命名空间（含 `label`、`tasks`、`keep`、`render_only`）。
      config  由 `main` 经 `Config.from_env()` 构造好的实例。

    返回：进程退出码（0 = 跑完并写出记分卡）。
    """
    config.stream = False  # bench 关流式：本网关流式不回传 output_tokens，非流式才准

    def _load_all():
        files = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "eval", "results", "*.json")))
        rs = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                rs.append(json.load(f))
        rs.sort(key=lambda r: (r.get("label") != "baseline", r.get("label", "")))
        return rs

    if args.render_only:
        results = _load_all()
        if not results:
            print("错误：eval/results/ 下没有结果可渲染（先跑一次 bench）", file=sys.stderr)
            return 1
        render_scorecard(results, DEFAULT_SCORECARD)
        print(f"已从 {len(results)} 组结果重建 {DEFAULT_SCORECARD}")
        return 0

    run_bench(args.tasks, config, args.label)
    results = _load_all()
    render_scorecard(results, DEFAULT_SCORECARD)
    print(f"记分卡写入 {DEFAULT_SCORECARD}（{len(results)} 组结果）")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口：接线 + 分发（§14.4 / §8.1）。

    实现契约（按此顺序）：
      1. `load_dotenv()` —— 把 `.env` 读进环境（在构造 Config / LLMClient 之前）。
      2. `config = Config.from_env()` —— 默认值之上叠加非密钥旋钮（§8.1 映射表）。
      3. `args = build_parser().parse_args(argv)`。
      4. 按 `args.command` 分发：
             "solve" → return cmd_solve(args, config)
             "bench" → return cmd_bench(args, config)
      （密钥类环境变量由 anthropic SDK 直接读取，绝不进 Config、绝不打印。）

    参数：
      argv  参数序列，缺省 None 时用 `sys.argv[1:]`（便于测试注入）。

    返回：进程退出码，供 `sys.exit(main())` 使用。
    """
    load_dotenv()
    config = Config.from_env()
    # 流式在各子命令内按需设置：solve 开（实时显示）、bench 关（成本记账准确）。
    args = build_parser().parse_args(argv)
    if args.command == "solve":
        return cmd_solve(args, config)
    if args.command == "bench":
        return cmd_bench(args, config)
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
