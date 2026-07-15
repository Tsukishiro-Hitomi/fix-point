"""tests/test_loop.py —— agent/loop.py 的起步测试集（DESIGN §8.12）。

loop 主要靠 bench 做**集成验证**；本文件只放：
- **可运行**的真实断言：三个 dataclass（AgentResult/StepRecord/ToolCall）的声明式契约
  （字段名与默认值、steps 是列表、无 solved 字段）——现在即绿。
- **一个用 monkeypatch 假 LLM 的骨架测试**：离线驱动 run_agent 走完「tool_use 一轮 →
  纯文字收尾」，断言 stop_reason=="model_stop"。因 run_agent 目前是桩，先 skip 标 TODO。
- 其余验收项以 **TODO 清单**列出，逐条对齐 DESIGN §8.12。

全程不触网络（假 LLM）；也不触真实工具 / 沙箱（monkeypatch 掉 guarded_execute）。
"""

from __future__ import annotations

import dataclasses
import types

import pytest

from agent.loop import AgentResult, StepRecord, ToolCall, run_agent
from agent.llm import Usage
from agent.config import Config

_TODO = "TODO(实现后取消 skip)：run_agent 及其 helper 目前是桩"


# ---------------------------------------------------------------------------
# 可运行的真实断言：dataclass 声明式契约（DESIGN §8.4）——现在即绿
# ---------------------------------------------------------------------------
def test_agent_result_defaults_and_shape():
    r = AgentResult(stop_reason="model_stop")
    assert r.stop_reason == "model_stop"
    assert r.steps == [] and isinstance(r.steps, list)   # steps 是列表，不是整数
    assert r.num_steps == 0
    assert r.total_input_tokens == 0
    assert r.total_output_tokens == 0
    assert r.total_cost_usd == 0.0
    assert r.final_text == ""
    assert r.error is None


def test_agent_result_has_no_solved_field():
    """有意不设 solved 字段——杜绝「信任模型自述」（DESIGN §8.4）。"""
    names = {f.name for f in dataclasses.fields(AgentResult)}
    assert "solved" not in names
    assert "usage" not in names                          # 不存在 result.usage dict


def test_step_and_toolcall_fields():
    step_names = [f.name for f in dataclasses.fields(StepRecord)]
    assert step_names == [
        "index", "assistant_text", "tool_calls", "stop_reason",
        "input_tokens", "output_tokens", "cost_usd",
    ]
    tc_names = [f.name for f in dataclasses.fields(ToolCall)]
    assert tc_names == ["name", "input", "result_preview"]


# ---------------------------------------------------------------------------
# 假 LLM 基建（scripted Message；不触网络）
# ---------------------------------------------------------------------------
def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, tool_input):
    return types.SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def _fake_usage(inp, out):
    return types.SimpleNamespace(input_tokens=inp, output_tokens=out)


def _fake_message(*, content, stop_reason, usage):
    return types.SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


def _make_fake_llm_class(script):
    """返回一个假 LLMClient 类，其 create 依次吐出 script 里的 Message。"""

    class _FakeLLM:
        def __init__(self, config, model=None):
            self._config = config
            self._model = model or config.model
            self._i = 0

        def create(self, messages, *, system=None, tools=None, tool_choice=None, stream=None):
            msg = script[self._i]
            self._i += 1
            return msg

        def snapshot(self):
            return Usage(0, 0, self._i, 0.0)

        @property
        def model(self):
            return self._model

    return _FakeLLM


@pytest.mark.skip(reason=_TODO)
def test_run_agent_model_stop_with_fake_llm(tmp_path, monkeypatch):
    """离线骨架：一轮 tool_use（read_file）→ 一轮纯文字收尾 → stop_reason=="model_stop"。

    覆盖 DESIGN §8.12 的「model_stop 出口 + 如实记录步数/工具调用 + 不判 solved」。
    """
    script = [
        _fake_message(
            content=[
                _text_block("先看看 parser.py"),
                _tool_use_block("tu_1", "read_file", {"path": "parser.py"}),
            ],
            stop_reason="tool_use",
            usage=_fake_usage(120, 30),
        ),
        _fake_message(
            content=[_text_block("测试全绿，改了乘除优先级。")],
            stop_reason="end_turn",
            usage=_fake_usage(60, 15),
        ),
    ]
    monkeypatch.setattr("agent.loop.LLMClient", _make_fake_llm_class(script))
    # 不触真实工具/沙箱：guarded_execute 回一个 PASS 字符串
    monkeypatch.setattr(
        "agent.loop.guarded_execute",
        lambda name, tool_input, workdir, **kw: "结果：PASS（returncode=0）",
    )

    result = run_agent(str(tmp_path), "让 test_precedence_* 重新通过", Config())

    assert isinstance(result, AgentResult)
    assert result.stop_reason == "model_stop"
    assert result.num_steps == 2
    assert len(result.steps) == 2
    # 第一轮触发了一次 read_file 工具调用
    assert result.steps[0].tool_calls[0].name == "read_file"
    # 收尾文字进入 final_text
    assert "优先级" in result.final_text
    assert not hasattr(result, "solved")


@pytest.mark.skip(reason=_TODO)
def test_build_system_prompt_self_correction_toggle():
    """build_system_prompt：self_correction=True 时含反思段、为假时不含；均逐字稳定
    （DESIGN §8.8）。"""
    from agent.loop import build_system_prompt

    base = build_system_prompt(Config(self_correction=False))
    reflective = build_system_prompt(Config(self_correction=True))
    # 同参数两次调用逐字稳定（利于 prompt caching）
    assert base == build_system_prompt(Config(self_correction=False))
    assert reflective == build_system_prompt(Config(self_correction=True))
    # 反思版更长且是基础版的超集（追加段），基础版不含反思段
    assert reflective != base
    assert len(reflective) > len(base)


# ---------------------------------------------------------------------------
# TODO 待补清单（逐条对齐 DESIGN §8.12 验收标准）
# ---------------------------------------------------------------------------
# TODO(你来补): max_steps —— 脚本让每轮都带 tool_use，跑满 config.max_steps（可调小 Config(max_steps=3)）
#               后 stop_reason=="max_steps"、num_steps==max_steps。
# TODO(你来补): budget_exceeded —— 令累计成本触 cost_budget_usd，断言在**下次模型调用前**停、
#               stop_reason=="budget_exceeded"（可用小预算 + 高 token 假 usage 触发）。
# TODO(你来补): error —— 让假 LLM.create 抛异常，断言 stop_reason=="error" 且 result.error 有简述、
#               **不上抛**。
# TODO(你来补): 历史管理 —— assistant 消息保留完整 content（含 tool_use 块原样）；一轮内**多个**
#               tool_use 的结果合并进**一条** user 消息，且每个 tool_result 的 tool_use_id 与对应
#               tool_use 匹配（可截获 messages / 或断言回灌结构）。
# TODO(你来补): 记账正确 —— total_input_tokens/total_output_tokens/total_cost_usd 等于各 StepRecord 之和；
#               成本折算与 config.cost_of 一致（含缺表按 0 兜底，_accumulate_usage / step_cost）。
# TODO(你来补): 模型消融 —— Config(model=Config().model_haiku) 无需改 loop 代码即可切；默认
#               config.model == "anthropic/claude-opus-4.8"。
# TODO(你来补): run_tests 透传 —— 断言 guarded_execute 收到 test_timeout==config.run_tests_timeout_s、
#               max_result_chars==config.max_tool_result_chars（用记录型 fake 截获 kwargs）。
# TODO(你来补): enable_retrieval —— 为真时首条 user 消息前置 retrieve_context(...) 文本、为假时完全
#               不触发检索路径（monkeypatch retrieve_context 断言调用/未调用）。
# TODO(集成，bench 跑): 端到端冒烟 —— 真实任务（改坏乘除优先级）观察「定位 → 改 → run_tests → 绿 → 停」，
#               每步 token/成本/工具调用被如实记录。属 eval/run_bench 的集成验证，不在离线单测。
