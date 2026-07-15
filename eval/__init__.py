"""fixpoint 评测 harness 包（DESIGN §10）。

本包是「裁判 + 记分」子系统：遍历任务集、为每题准备干净隔离副本并打
`break.patch`、在副本上跑一遍 agent 主循环、**独立复跑 pytest** 判定 solved
（目标测试全绿 **且** 无回归），最后汇总出 `scorecard.md` 与机器可读的
`results/<label>.json`。判定权始终在 harness 手里，**绝不采信 agent 自述**。

对外入口见 `eval.run_bench`。
"""
