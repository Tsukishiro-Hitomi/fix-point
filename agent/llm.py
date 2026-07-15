"""agent/llm.py —— LLM 封装（对外说话的唯一窗口）。

对应 DESIGN.md §7。本模块把「经网关调用 Claude」收拢成一层薄封装，
向上给 loop.py 一个稳定、可测的**单轮**接口，向下屏蔽 SDK 细节、重试、
超时与计费口径。

它只做三件事：
  1. 接线：经网关初始化 ``anthropic.Anthropic()``（凭据由 SDK 从环境变量读，
     不手传 api_key / base_url）。
  2. 发一轮请求：返回 SDK 原生 ``Message``——不解释 ``stop_reason``、不执行工具、
     不跑多轮循环。
  3. 记账：累加各轮 ``usage`` 的 tokens，按价格表（``config.cost_of``）折算成本，
     供记分卡使用。

非目标（明确不做）：agentic loop、工具执行、路径封闭、跑 pytest——这些分别属于
loop.py / tools.py / sandbox.py。

本文件顶部使用 ``from __future__ import annotations``，使 Python 3.9 也能安全书写
``str | None`` / ``list[dict]`` 之类注解（注解仅在解析期为字符串，不在运行期求值）。

【脚手架说明】本文件是契约脚手架：所有实现型方法只写签名 + 详尽 docstring 契约，
函数体一律 ``raise NotImplementedError``；声明式内容（``Usage`` 字段）写全。
学习者按契约填实现即可。
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import anthropic

# cost_of 是「唯一计价函数」，owner 是 config.py（DESIGN §8.2）；llm 与 loop 复用同一份，
# 消除三处重复计价。此处仅引用其字段/函数，不重复实现价格表逻辑。
from agent.config import Config, cost_of

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据表示（DESIGN §7.2）：Usage 记账快照——声明式内容，写全。
# ---------------------------------------------------------------------------
class Usage(NamedTuple):
    """一次记账快照（累计量），由 :meth:`LLMClient.snapshot` 返回。

    - ``cost_usd`` 是**估算值**，按 ``config.price_per_mtok`` 折算，仅供横向比较 /
      预算护栏参考；网关实际计费口径可能不同。
    - **token 计数以 ``response.usage`` 为准（权威）**。
    - 模型不在价格表时 ``cost_usd`` 为 ``None``（而非 0），以区分「零成本」与「无价可算」。
    """

    input_tokens: int            # 累计 prompt tokens（以各轮 response.usage 为准）
    output_tokens: int           # 累计 completion tokens
    calls: int                   # 累计成功返回的请求次数
    cost_usd: Optional[float]    # 估算成本；模型不在价格表时为 None


# ---------------------------------------------------------------------------
# LLMClient（DESIGN §7.3）
# ---------------------------------------------------------------------------
class LLMClient:
    """经网关调用 Claude 的薄封装 + Usage 记账器。

    典型用法（loop.py 侧）::

        client = LLMClient(config)                 # 默认 config.model（opus）
        msg = client.create(messages=msgs, system=sys, tools=TOOLS)
        # ... 读 msg.content / msg.stop_reason / msg.usage ...
        snap = client.snapshot()                   # 拿累计 token 与成本

    消融对照切模型：``LLMClient(config, config.model_haiku)``——记账自动按 haiku 价。
    """

    def __init__(self, config: Config, model: Optional[str] = None) -> None:
        """构造并接线（DESIGN §7.3「构造」）。

        契约：
        - 构造 ``self._client = anthropic.Anthropic(timeout=config.timeout_s,
          max_retries=config.max_retries)``——**不手传** ``api_key`` / ``base_url``，
          由 SDK 从环境变量 ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` 自动解析
          （锁定方案，便于换网关零改码）。
        - ``self._model = model or config.model``（传 ``config.model_haiku`` 即切消融模型）。
        - 保存 ``self._config = config``；把记账累加器（累计 input/output tokens、
          calls 次数）清零。
        - 重试次数与超时**只经构造参数注入**（``max_retries`` / ``timeout``），交给 SDK；
          本封装内**不得**再写重试 for/while 循环（DESIGN §7.4）。

        边界：
        - 若因缺 ``ANTHROPIC_API_KEY`` 导致构造失败，**快速失败并给清晰报错**
          （形如「缺少 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL，请先
          ``cp .env.example .env`` 并填值」），不静默吞。
        - ``model`` 不在价格表是**允许的**（不报错），仅后续 ``cost_usd`` 记为 ``None``
          并打一次 warning。

        Args:
            config: 全局旋钮面板（提供 model / timeout_s / max_retries / 价格表等）。
            model: 覆盖 ``config.model`` 的模型 id；``None`` 时用 ``config.model``。
        """
        raise NotImplementedError

    def create(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        stream: bool | None = None,
    ) -> "anthropic.types.Message":
        """发**一轮**请求，返回 SDK 原生 ``Message``（DESIGN §7.3「create」）。

        组装与调用契约：
        - ``model`` 与 ``max_tokens``（取 ``config.max_tokens``，默认 8192）**恒定传入**。
        - ``system`` / ``tools`` / ``tool_choice`` **仅在非 ``None`` 时才放入**请求参数
          （有些网关对显式 ``None`` 敏感）。
        - ``stream`` **缺省取 ``config.stream``**——这是流式开关的**唯一接线点**
          （loop 无需显式传）。
            * ``stream is False`` → ``self._client.messages.create(...)``；
            * ``stream is True``  → ``with self._client.messages.stream(...) as s:
              msg = s.get_final_message()``（把长输出聚合成完整 ``Message``，规避
              HTTP 超时）。
          两条路径**返回同型 ``Message``、记账口径一致**。
        - 只使用可移植请求子集：``model / max_tokens / system / messages / tools /
          tool_choice / stream``。``thinking`` / ``output_config.effort`` / prompt caching
          等 Anthropic 专属字段**不纳入本契约**（见 DESIGN §7.6 扩展点）。

        记账契约：
        - 调用成功后从 ``message.usage`` 把 ``input_tokens`` / ``output_tokens`` 累加进
          累加器，``calls += 1``，并按 ``cost_of`` 折算成本（成本计算见 :meth:`snapshot`）。
        - **原样返回** SDK 的 ``Message``（不拆包、不改写、不解释 ``stop_reason``、
          不执行工具、不发多轮）。调用方（loop.py）自行读 ``message.content``
          （``text`` / ``tool_use`` 块）、``message.stop_reason``、``message.usage``。

        边界与健壮性：
        - ``message.usage`` 为 ``None`` / 缺字段 → **跳过累加、打 warning、不崩**。
        - ``cache_*`` 字段一律可选、缺省按 0；成本只用 ``input_tokens`` /
          ``output_tokens`` 两项算。
        - **不在此层兜住 API 异常假装成功**：``anthropic.APIStatusError`` 及子类、
          ``APITimeoutError``、``APIConnectionError`` 等**照原样上抛**给 loop.py，由它
          决定该步处置；失败时可从异常/响应取 ``request_id`` 记日志。

        Args:
            messages: Anthropic 原生格式的消息列表（``system`` 不在其中，另经参数传）。
            system: system prompt；``None`` 时不放入请求。
            tools: 工具 schema 列表（如 ``agent.tools.TOOLS``）；``None`` 时不放入。
            tool_choice: 工具选择策略；``None`` 时不放入。
            stream: 是否流式；``None`` 时取 ``config.stream``。

        Returns:
            SDK 原生 ``anthropic.types.Message``，含 ``.content`` / ``.stop_reason`` /
            ``.usage``。
        """
        raise NotImplementedError

    def snapshot(self) -> Usage:
        """返回当前累计记账快照（含成本估算）。

        契约：
        - ``input_tokens`` / ``output_tokens`` / ``calls`` 为自构造（或上次 :meth:`reset`）
          以来各次成功 ``create`` 的累加值。
        - ``cost_usd`` 复用**唯一计价函数**：等价于
          ``cost_of(<累计 input>, <累计 output>, self._config, self._model)``；
          模型缺表时返回 ``None`` 并（首次）打 warning，而非 0（区分「零成本」与「无价可算」）。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """把记账累加器清零（tokens 与 calls 归零）。

        bench 每个任务开跑前调用，使各任务的 token / 成本互不串账。不影响 ``self._model``
        与 ``self._client``。
        """
        raise NotImplementedError

    @property
    def model(self) -> str:
        """当前生效的模型 id（消融对照会把它打进记分卡）。

        默认 ``config.model``（opus）；构造时传 ``config.model_haiku`` 则为 haiku id。
        """
        raise NotImplementedError
