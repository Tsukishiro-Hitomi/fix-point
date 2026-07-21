"""唯一旋钮面板：``Config`` 数据类 + 价格表 + 唯一计价函数 ``cost_of``。

契约摘要（详见 DESIGN.md §8.1 / §8.2）：

- 全部旋钮收在**一个** ``@dataclass Config`` 实例里，字段一律小写；各章统一用
  ``config.model`` / ``config.max_steps`` 等实例属性访问，**不存在** 模块级大写常量
  （无 ``MODEL`` / ``MAX_TOKENS`` 之类）。
- 计价的「价格表结构 + 单位 + 计价函数」各只有一份，owner 就是本模块：``llm`` 与
  ``loop`` 复用 ``cost_of``，``eval`` 直接读 ``AgentResult.total_cost_usd`` 不再自算。
- 密钥（``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``）由 anthropic SDK 直接从环境
  读取，**不进 Config**、不硬编码、不打印、不写日志。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import os

@dataclass
class Config:
    """承载全部运行旋钮的单一配置对象。

    字段分五组（大脑 / 护栏预算 / 能力开关 / 工具截断预算 / 价格表）。默认值即
    跨章唯一口径，其它模块（eval 的 config_snapshot、.env.example、工具 handler 默认）
    都对齐这里的数字。消融对照**不新增字段**：只换 ``model``（opus↔haiku）、翻
    ``enable_retrieval``、翻 ``self_correction``，由 bench 分别构造 ``Config`` 跑。

    谁强制执行各旋钮（见 §8.1 表）：
      model / max_tokens / stream / timeout_s / max_retries -> llm
      max_steps -> loop（终止 "max_steps"）；cost_budget_usd -> loop（"budget_exceeded"）
      run_tests_timeout_s -> 工具（loop 透传）；judge_timeout_s -> eval
      enable_retrieval / self_correction -> loop（挂载点 / build_system_prompt）
      max_tool_result_chars 等 -> loop 抽取后透传给工具；price_per_mtok -> cost_of
    """

    # —— 大脑 ——
    model: str = "anthropic/claude-opus-4.8"          # 正式跑
    model_haiku: str = "anthropic/claude-haiku-4.5"   # 仅消融对照（精确 id 以网关模型列表为准）
    max_tokens: int = 8192                            # 单次响应输出上限（write_file 重写整文件时需余量）
    stream: bool = True                               # 流式（llm 从此读取，见 §7.3）
    timeout_s: float = 120.0                          # 单次 HTTP 请求超时（秒；SDK 默认 600 收紧到此）
    max_retries: int = 2                              # 交给 SDK 自动重试（SDK 默认即 2）

    # —— 护栏 / 预算 ——
    max_steps: int = 30                               # 单任务最多迭代轮数（硬上限）
    cost_budget_usd: float = 0.50                     # 单任务成本预算（美元）
    run_tests_timeout_s: int = 60                     # run_tests 子进程超时（透传给工具层）
    judge_timeout_s: int = 60                         # harness 复判 pytest 超时（eval 用）

    # —— 能力开关（消融）——
    enable_retrieval: bool = False                    # embedding 代码检索（v1）；False = 仅靠 search 工具
    self_correction: bool = False                     # 为真时 system prompt 追加反思段（v1 消融）

    # —— 工具截断预算（loop 抽取后透传给 guarded_execute / 工具行级预算）——
    max_tool_result_chars: int = 8000
    max_read_lines: int = 400
    max_search_hits: int = 100
    max_test_output: int = 4000

    # —— 成本核算：模型 id -> (输入价, 输出价)，美元/百万 token（唯一价格表）——
    # 默认取第一方参考价（Opus 4.8 = $5/$25、Haiku 4.5 = $1/$5 每百万 token）；
    # reviewer 需按聚合网关实际计费校准。未知模型缺表 -> cost_of 返回 None
    # （调用方按 0 展示 + 留告警），绝不崩溃。
    price_per_mtok: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "anthropic/claude-opus-4.8": (5.0, 25.0),
            "anthropic/claude-haiku-4.5": (1.0, 5.0),
        }
    )

    @classmethod
    def from_env(cls) -> "Config":
        """在默认值之上叠加 ``.env`` 里的**非密钥**旋钮，返回新的 ``Config`` 实例。

        由 ``cli.py`` 在 ``load_dotenv()`` 之后调用。env 变量 -> 字段映射（§14.4）：

          - ``FIXPOINT_MODEL``    -> ``model``
          - ``MAX_STEPS``         -> ``max_steps``          （解析为 int）
          - ``RUN_TESTS_TIMEOUT`` -> ``run_tests_timeout_s``（解析为 int）

        契约：
          - 未设置的 env 变量 -> 对应字段保持 dataclass 默认值。
          - 密钥类（``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``）**绝不**被吸收进
            Config；它们由 SDK 直接读环境。
          - 其余旋钮（max_tokens、cost_budget_usd、开关、截断预算、price_per_mtok 等）
            本方法不从 env 覆盖，用 Config 默认。

        Returns:
            一个字段已按 env 覆盖的新 ``Config`` 实例。
        """
        overrides = {}
        model = os.environ.get("FIXPOINT_MODEL")
        if model is not None:
            overrides["model"] = model
        max_steps = os.environ.get("MAX_STEPS")
        if max_steps is not None:
            overrides["max_steps"] = int(max_steps)
        timeout = os.environ.get("RUN_TESTS_TIMEOUT")
        if timeout is not None:
            overrides["run_tests_timeout_s"] = int(timeout)
        return cls(**overrides)


def cost_of(
    in_tokens: int,
    out_tokens: int,
    config: "Config",
    model: Optional[str] = None,
) -> Optional[float]:
    """按唯一价格表把 token 用量折算成美元成本（项目内唯一计价函数）。

    Args:
        in_tokens: 输入（prompt）token 数。
        out_tokens: 输出（completion）token 数。
        config: 提供 ``price_per_mtok`` 价格表与默认 ``model`` 的配置实例。
        model: 计价所用模型 id；``None`` 时回落到 ``config.model``。

    计算：设该模型价格为 ``(pin, pout)``（美元/百万 token），则返回
    ``(in_tokens * pin + out_tokens * pout) / 1_000_000``。

    Returns:
        成本（美元）；若模型不在 ``config.price_per_mtok`` 中则返回 ``None``
        （调用方约定按 0 展示并记一条告警，绝不崩溃）。
    """
    if model is None:
        model = config.model
    if model not in config.price_per_mtok:
        return None
    pin, pout = config.price_per_mtok.get(model)
    return (float)(in_tokens * pin + out_tokens * pout) / 1000000
