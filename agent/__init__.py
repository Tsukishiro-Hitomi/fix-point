"""fixpoint 的 agent 包。

测试驱动的自主编码 agent 的运行时核心，五个模块各司其职（见 DESIGN.md §2–3）：

- ``config``  —— 唯一旋钮面板 ``Config`` + 价格表 + 唯一计价函数 ``cost_of``。
- ``llm``     —— ``LLMClient``：Anthropic() 经聚合网关调用 + Usage 记账。
- ``tools``   —— 6 个工具 handler + TOOLS schema + ``guarded_execute`` 护栏分发。
- ``sandbox`` —— 路径封闭 ``resolve_in_workdir`` + 隔离工作区生命周期。
- ``loop``    —— ``run_agent`` ReAct 主循环 + ``AgentResult`` / ``StepRecord``。

本包只负责编排、执行与记账，**从不**自行判定任务 solved/failed——判定权唯一
归属评测 harness（``eval/run_bench.py``），它在 agent 停手后用受保护的原版
测试独立复跑 pytest。
"""
