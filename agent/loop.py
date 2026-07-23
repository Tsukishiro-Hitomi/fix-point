"""agent/loop.py —— Agent 核心循环（发动机）。

对应 DESIGN.md §8。给定一个**任务工作目录**（已被 ``git apply break.patch`` 改红的
fixture 副本）和一段**任务描述**，:func:`run_agent` 驱动模型反复
「观察 → 决策 → 调工具 → 读结果」，直到模型收尾或触护栏，然后交回一个
:class:`AgentResult`。

本层只负责**编排与记账**，不负责：
  - 工具怎么读写 / 跑 pytest —— ``tools.py`` + ``sandbox.py``；
  - 怎么调网关 / 收流 / 取 usage —— ``llm.py``；
  - 检索实现（v1）；
  - **solved / failed 判定**（属 harness）——``run_agent`` 只如实记录轨迹，
    **绝不声称「已解决」**（有意不设 ``solved`` 字段）。

【脚手架说明】声明式内容（三个 dataclass 的字段与默认值）写全；实现型函数
（``run_agent`` / ``build_system_prompt`` 及内部 helper）只写签名 + 契约 docstring，
函数体 ``raise NotImplementedError``。控制流骨架以伪码写进 :func:`run_agent` 的 docstring。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agent.config import Config, cost_of
from agent.llm import LLMClient
from agent.tools import TOOLS, guarded_execute


# ---------------------------------------------------------------------------
# 结果对象（DESIGN §8.4）——声明式内容，字段与默认值写全。
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """一次工具调用的轨迹记录（供记分卡 / 调试用，不存全文）。"""

    name: str
    input: dict
    result_preview: str          # guarded_execute 返回值的截断预览


@dataclass
class StepRecord:
    """单轮（一次模型调用 + 其触发的工具执行）的记录。"""

    index: int
    assistant_text: str          # 本轮所有 text 块拼接（模型的说明 / 思考）
    tool_calls: List[ToolCall]   # 本轮触发的工具调用（可多个——并行工具）
    stop_reason: str             # 本轮 response.stop_reason（原样，供排查；非循环级终止原因）
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class AgentResult:
    """一次 ``run_agent`` 的完整结果（harness 与记分卡消费）。

    重要口径（DESIGN §8.4）：
    - ``stop_reason`` 是**循环为什么停**，与「任务是否解决」无关；取值恒为四者之一：
      ``"model_stop"`` | ``"max_steps"`` | ``"budget_exceeded"`` | ``"error"``。
    - **有意不设 ``solved`` 字段**——pass@1 由 harness 停手后独立复跑 pytest 判定，
      杜绝「信任模型自述」。
    - ``steps`` 是**列表**；步数用 ``num_steps``（整数），不要把 ``steps`` 当整数。
    - 不存在 ``result.usage`` dict——token / 成本经下面三个标量字段暴露。
    """

    stop_reason: str
    steps: List[StepRecord] = field(default_factory=list)
    num_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    final_text: str = ""         # 模型最后一轮文字（收尾总结），仅供人看
    error: Optional[str] = None  # stop_reason == "error" 时的简述


# ---------------------------------------------------------------------------
# System prompt（DESIGN §8.8）
# ---------------------------------------------------------------------------
def build_system_prompt(config: Config) -> str:
    """构造**逐字稳定**的 system prompt（DESIGN §8.8）。

    契约：
    - 返回一段不含任务 / 时间 / 随机内容的 system 文本（利于 prompt caching）；
      具体任务与检索片段一律放**首条 user 消息**，不进 system。
    - 文本须逐条落进 DESIGN §8.8 的要点：角色与封闭工作目录、显式的
      「定位 → 改 → 立刻 run_tests → 读红绿 → 迭代」循环、每次改完必跑测试、
      最小改动原则、**反作弊**（绝不改测试文件 / 删断言 / raise/skip 绕过）、
      以真实 run_tests 输出为准、全绿即一句总结后停止且不再调工具、工具纪律
      （edit_file 的 old_string 需精确唯一匹配、路径限工作目录内）、简洁。
    - **``config.self_correction`` 为真时，在稳定正文后追加一个反思段**：要求模型读到
      pytest 失败后**先诊断根因再改**，而非急着改。追加段在同一次运行内仍**逐字稳定**
      （利于缓存）。为假时不含该段。

    Args:
        config: 仅读取 ``config.self_correction`` 决定是否追加反思段（其余旋钮不影响文本）。

    Returns:
        system prompt 字符串。
    """
    base = (
        "你是一个在受限工作目录内修复代码的自主 agent。你只能通过提供的工具"
        "（list_dir / read_file / search / edit_file / write_file / run_tests）操作，"
        "所有路径都限定在任务工作目录内。\n\n"
        "工作循环：定位相关代码 → 做最小改动 → 立刻用 run_tests 跑测试 → 读红/绿 → "
        "按结果迭代。每次改完都必须重新跑测试，以真实 run_tests 输出为准。\n\n"
        "纪律：\n"
        "- 只做让测试通过所需的最小改动，不顺手重构无关代码。\n"
        "- edit_file 的 old_string 必须与文件内容精确、唯一匹配（先 read_file 看清上下文）。\n"
        "- 反作弊：绝不修改测试文件、绝不删断言、绝不用 raise/skip 绕过测试。\n"
        "- 所有测试通过时，用一句话总结修改然后停止，不要再调用任何工具。\n"
        "- 保持简洁。"
    )
    if config.self_correction:
        base += (
            "\n\n反思：run_tests 报失败时，先读报错、诊断根因，再动手改；"
            "别在没弄清原因前反复试探性修改。"
        )
    return base

# ---------------------------------------------------------------------------
# 主循环（DESIGN §8.3 / §8.9）
# ---------------------------------------------------------------------------
def run_agent(workdir: str, task: str, config: Optional[Config] = None) -> AgentResult:
    """驱动 agent 循环，返回 :class:`AgentResult`（DESIGN §8.3 / §8.9）。

    输入：
        workdir: 任务工作目录的**绝对路径**（harness 已备好：纯净副本 + 已打 break.patch）；
            loop 当沙箱根**原样透传**给每次工具调用，自身不做路径校验（由工具层保证）。
        task: 自然语言任务描述（来自 ``task.json`` 的 ``description`` 字符串——**不是 Task 对象**）。
        config: 护栏 / 模型 / 开关；``None`` 时函数内 ``config = config or Config()``
            （避免可变对象作默认参数这一反模式）。

    行为：
        构造 system prompt 与首条 user 消息 → 进入迭代循环 → 每轮调 ``LLMClient.create``
        拿响应、累计 usage / 成本、把响应写回历史；若响应含 ``tool_use`` 块则逐个经
        ``guarded_execute`` 执行、结果并成**一条** user 消息回灌，否则视为收尾 → 直到命中
        某终止条件。

    终止条件（DESIGN §8.6，以**是否存在 tool_use 块**为准，比只看 stop_reason 字符串更稳）：
        - ``"model_stop"``：某轮响应**不含** tool_use 块（模型只说话、收尾）。
        - ``"max_steps"``：迭代计数达到 ``config.max_steps``。
        - ``"budget_exceeded"``：在**每次调用模型之前**检查
          ``total_cost_usd >= config.cost_budget_usd``，超则停。
        - ``"error"``：``create`` 或分发过程抛未预期异常（重试后仍失败等）；**兜底记
          ``error`` 并停，不上抛**。
        边角：``stop_reason == "max_tokens"`` 且该轮仍有 tool_use → 照常分发继续；无则按
        ``model_stop`` 停并在 ``StepRecord.stop_reason`` 标注（提示可能被截断）。
        ``"refusal"`` 视为一种 ``model_stop``（记录以便排查）。

    输出：
        :class:`AgentResult`。**不做 solved 判定**、**不修改 workdir 以外任何东西**。

    ------------------------------------------------------------------------
    控制流骨架（DESIGN §8.9，只给控制流 + 挖空 helper，学习者填实现）::

        config = config or Config()
        result = AgentResult(stop_reason="")
        client = LLMClient(config)
        system = build_system_prompt(config)          # 稳定不变；self_correction 时追加反思段
        # 首条 user 消息：enable_retrieval 时前置 retrieve_context(task, workdir, config)
        messages = [ ... ]

        for i in range(config.max_steps):
            # A: 调模型前先查预算 → 超则 stop_reason="budget_exceeded" 返回
            # B: try 调 client.create(system=system, messages=messages, tools=TOOLS)
            #    except → stop_reason="error" + result.error 返回
            # C: _accumulate_usage(result, resp, config) 累加 token/成本（用 config.cost_of）
            # D: 把 resp.content 完整块列表追加为 assistant 消息（**保留 tool_use 块原样**）
            # E: 抽出 tool_use 块；生成 StepRecord 追加进 result.steps
            # F: 若无 tool_use → 收尾：final_text、stop_reason="model_stop"、num_steps=i+1、返回
            # G: 对每个 tool_use 调
            #        guarded_execute(tu.name, tu.input, workdir,
            #                        test_timeout=config.run_tests_timeout_s,
            #                        max_result_chars=config.max_tool_result_chars)
            #    收集成**一条** user 消息（一组 tool_result，各带对应 tool_use_id）回灌 messages
        # for 正常结束 → stop_reason="max_steps"、num_steps=config.max_steps、返回

    消息历史管理要点（DESIGN §8.5）：
        - ``messages`` 是 Anthropic 原生格式 list；``system`` 独立传入（不进 messages）。
        - assistant 消息的 ``content`` 用**完整的 ``response.content`` 块列表**，必须保留
          ``tool_use`` 块原样，否则后续 ``tool_result`` 无法对应。
        - 一轮内**多个** tool_use 的结果**必须合并进同一条 user 消息**（一组 ``tool_result``
          块，每块带匹配的 ``tool_use_id`` 与字符串结果；工具层已把错误编码进字符串，
          一般不设 ``is_error``）。
        - MVP 不做历史裁剪。
    """
    config = config or Config()
    result = AgentResult(stop_reason="")
    client = LLMClient(config)                  # ← 测试里被 monkeypatch 成假 LLM
    system = build_system_prompt(config)

    user_text = task
    if config.enable_retrieval:                 # MVP 默认 False，不走检索
        user_text = retrieve_context(task, workdir, config) + "\n\n" + task
    messages = [{"role": "user", "content": user_text}]

    for i in range(config.max_steps):
        # A: 调模型前先查预算
        if result.total_cost_usd >= config.cost_budget_usd:
            result.stop_reason = "budget_exceeded"
            result.num_steps = i
            return result

        # B: 调模型（异常兜底成 error，不上抛）
        try:
            resp = client.create(messages=messages, system=system, tools=TOOLS)
        except Exception as e:
            result.stop_reason = "error"
            result.error = str(e)
            result.num_steps = i
            return result

        # C: 记账
        _accumulate_usage(result, resp, config)

        # D: 保留完整 content 作为 assistant 消息（tool_use 块必须原样留着）
        messages.append({"role": "assistant", "content": resp.content})

        # E: 拆 text / tool_use，算本步 token/成本
        assistant_text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        usage = getattr(resp, "usage", None)
        s_in = getattr(usage, "input_tokens", 0) if usage else 0
        s_out = getattr(usage, "output_tokens", 0) if usage else 0
        s_cost = step_cost(usage, config) if usage else 0.0

        # F: 无 tool_use → 收尾
        if not tool_uses:
            result.steps.append(StepRecord(
                index=i, assistant_text=assistant_text, tool_calls=[],
                stop_reason=resp.stop_reason,
                input_tokens=s_in, output_tokens=s_out, cost_usd=s_cost,
            ))
            result.final_text = assistant_text
            result.stop_reason = "model_stop"
            result.num_steps = i + 1
            return result

        # G: 逐个执行工具，结果合并进【一条】user 消息
        tool_calls, tool_results = [], []
        for tu in tool_uses:
            out = guarded_execute(
                tu.name, tu.input, workdir,
                test_timeout=config.run_tests_timeout_s,
                max_result_chars=config.max_tool_result_chars,
            )
            tool_calls.append(ToolCall(name=tu.name, input=tu.input, result_preview=out[:200]))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,       # 必须与对应 tool_use 的 id 匹配
                "content": out,
            })
        result.steps.append(StepRecord(
            index=i, assistant_text=assistant_text, tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            input_tokens=s_in, output_tokens=s_out, cost_usd=s_cost,
        ))
        messages.append({"role": "user", "content": tool_results})

    # for 正常跑完 → max_steps
    result.stop_reason = "max_steps"
    result.num_steps = config.max_steps
    return result 


# ---------------------------------------------------------------------------
# 挖空 helper（DESIGN §8.7 / §8.9 / §8.11）——只给签名 + 公式/契约说明，学习者填实现。
# ---------------------------------------------------------------------------
def _accumulate_usage(result: AgentResult, resp: "object", config: Config) -> None:
    """把一轮响应的 usage 累加进 ``result``（DESIGN §8.7）。

    契约：
    - 从 ``resp.usage`` 取 ``input_tokens`` / ``output_tokens`` 累加到
      ``result.total_input_tokens`` / ``result.total_output_tokens``。
    - 按**唯一计价函数** ``cost_of(in, out, config)`` 折算美元累加到
      ``result.total_cost_usd``；缺表（``cost_of`` 返回 ``None``）按 0 计并可在日志留痕，
      **不崩**。
    - ``resp.usage`` 为 ``None`` / 缺字段时跳过累加、不崩（与 llm 层记账口径一致）。

    Args:
        result: 就地累加的结果对象。
        resp: 一轮的 SDK ``Message``（读其 ``.usage``）。
        config: 提供计价表 / 计价函数。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return                                  # 缺 usage → 跳过、不崩
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    if in_tok is None or out_tok is None:
        return
    result.total_input_tokens += in_tok
    result.total_output_tokens += out_tok
    result.total_cost_usd += step_cost(usage, config)


def step_cost(usage: "object", config: Config) -> float:
    """单步成本折算（DESIGN §8.7，只给签名 + 公式说明）。

    定义为：``config.cost_of(usage.input_tokens, usage.output_tokens, config)``。
    缺表时按 0（``cost_of`` 返回 ``None`` → 记 0.0）。学习者据此自行实现。

    Args:
        usage: 含 ``input_tokens`` / ``output_tokens`` 的对象（如 ``Message.usage``）。
        config: 提供计价表 / 计价函数。

    Returns:
        本步估算成本（美元）；缺表按 0.0。
    """
    cost = cost_of(usage.input_tokens, usage.output_tokens, config)
    return cost if cost is not None else 0.0


def retrieve_context(task: str, workdir: str, config: Config) -> str:
    """检索层挂载点（DESIGN §8.11，v1 可选；MVP 不实现）。

    统一契约：loop 拥有这个挂载点 + 布尔 ``config.enable_retrieval``。
    - baseline（记分卡里叫 ``baseline``）= ``enable_retrieval=False``：**不做 embedding
      预注入，仅靠 ``search`` 工具**——此函数不被调用。
    - embedding = ``enable_retrieval=True``：v1 用 sentence-transformers + bge-small-en-v1.5
      预注入相关代码片段，返回一段**可前置进首条 user 消息**的文本。

    注意：「grep 检索」不是一种 retrieve_context——grep 是 agent 随时可调的 ``search``
    工具，不是开局注入。消融「只改一个字段」即翻 ``enable_retrieval``。

    Args:
        task: 任务描述。
        workdir: 任务工作目录绝对路径。
        config: 读取 ``enable_retrieval`` 等。

    Returns:
        可塞进首条 user 消息的检索文本。
    """
    raise NotImplementedError
