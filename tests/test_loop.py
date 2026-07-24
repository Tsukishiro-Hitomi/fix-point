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

        def create(self, messages, *, system=None, tools=None, tool_choice=None, stream=None, on_text=None):
            msg = script[self._i]
            self._i += 1
            return msg

        def snapshot(self):
            return Usage(0, 0, self._i, 0.0)

        @property
        def model(self):
            return self._model

    return _FakeLLM


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


def _tool_turn(idx, name="list_dir", tool_input=None, inp=10, out=5):
    return _fake_message(
        content=[_tool_use_block(f"tu_{idx}", name, tool_input or {"path": "."})],
        stop_reason="tool_use", usage=_fake_usage(inp, out),
    )


def _text_turn(text="done", inp=5, out=2):
    return _fake_message(content=[_text_block(text)], stop_reason="end_turn",
                         usage=_fake_usage(inp, out))


def test_run_agent_max_steps(tmp_path, monkeypatch):
    """每轮都带 tool_use → 跑满 max_steps 停。"""
    script = [_tool_turn(0), _tool_turn(1), _tool_turn(2)]
    monkeypatch.setattr("agent.loop.LLMClient", _make_fake_llm_class(script))
    monkeypatch.setattr("agent.loop.guarded_execute",
                        lambda name, ti, wd, **kw: "结果：FAIL（returncode=1）")
    result = run_agent(str(tmp_path), "永远修不好", Config(max_steps=3))
    assert result.stop_reason == "max_steps"
    assert result.num_steps == 3
    assert len(result.steps) == 3


def test_run_agent_budget_exceeded(tmp_path, monkeypatch):
    """累计成本触预算 → 在下次模型调用前停。"""
    script = [_tool_turn(0, inp=1000, out=1000), _text_turn("不该到这")]
    monkeypatch.setattr("agent.loop.LLMClient", _make_fake_llm_class(script))
    monkeypatch.setattr("agent.loop.guarded_execute",
                        lambda name, ti, wd, **kw: "结果：FAIL")
    result = run_agent(str(tmp_path), "费钱任务", Config(cost_budget_usd=0.0001))
    assert result.stop_reason == "budget_exceeded"
    assert result.num_steps == 1  # 第 0 步跑了，第 1 步开头被预算拦下（script[1] 未消费）


def test_run_agent_error_is_caught(tmp_path, monkeypatch):
    """create 抛异常 → stop_reason=='error'、result.error 有简述、不上抛。"""
    class _BoomLLM:
        def __init__(self, config, model=None):
            pass
        def create(self, **kwargs):
            raise RuntimeError("gateway exploded")
        def snapshot(self):
            return Usage(0, 0, 0, 0.0)
        @property
        def model(self):
            return "x"
    monkeypatch.setattr("agent.loop.LLMClient", _BoomLLM)
    result = run_agent(str(tmp_path), "任务", Config())  # 不应抛
    assert result.stop_reason == "error"
    assert result.error and "gateway exploded" in result.error
    assert result.num_steps == 0


def test_history_structure_multiple_tool_uses(tmp_path, monkeypatch):
    """assistant 保留完整 content（tool_use 原样）；一轮多个 tool_use 合并进一条 user 消息，id 对应。"""
    script = [
        _fake_message(
            content=[
                _tool_use_block("tu_a", "read_file", {"path": "a.py"}),
                _tool_use_block("tu_b", "read_file", {"path": "b.py"}),
            ],
            stop_reason="tool_use", usage=_fake_usage(10, 5),
        ),
        _text_turn(),
    ]
    seen = []

    class _RecLLM:
        def __init__(self, config, model=None):
            self._i = 0
        def create(self, messages, **kwargs):
            seen.append(list(messages))  # 快照本轮收到的历史
            msg = script[self._i]; self._i += 1
            return msg
        def snapshot(self):
            return Usage(0, 0, self._i, 0.0)
        @property
        def model(self):
            return "x"

    monkeypatch.setattr("agent.loop.LLMClient", _RecLLM)
    monkeypatch.setattr("agent.loop.guarded_execute",
                        lambda name, ti, wd, **kw: f"结果 for {ti['path']}")
    run_agent(str(tmp_path), "任务", Config())

    hist = seen[1]  # 第二轮 create 收到的历史
    assert hist[0]["role"] == "user"
    assert hist[1]["role"] == "assistant"
    tu_ids = [getattr(b, "id", None) for b in hist[1]["content"]
              if getattr(b, "type", None) == "tool_use"]
    assert tu_ids == ["tu_a", "tu_b"]  # tool_use 块原样保留
    assert hist[2]["role"] == "user"
    results = hist[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_a", "tu_b"]  # 合并进一条、id 一一对应
    assert all(r["type"] == "tool_result" for r in results)


def test_accounting_totals_match_steps(tmp_path, monkeypatch):
    """total_* 等于各 StepRecord 之和，且成本与 config.cost_of 一致。"""
    script = [_tool_turn(0, inp=100, out=20), _text_turn("done", inp=50, out=10)]
    monkeypatch.setattr("agent.loop.LLMClient", _make_fake_llm_class(script))
    monkeypatch.setattr("agent.loop.guarded_execute", lambda *a, **k: "ok")
    cfg = Config()
    result = run_agent(str(tmp_path), "任务", cfg)
    assert result.total_input_tokens == 150   # 100 + 50
    assert result.total_output_tokens == 30   # 20 + 10
    assert result.total_input_tokens == sum(s.input_tokens for s in result.steps)
    assert result.total_output_tokens == sum(s.output_tokens for s in result.steps)
    assert result.total_cost_usd == pytest.approx(sum(s.cost_usd for s in result.steps))
    pin, pout = cfg.price_per_mtok[cfg.model]
    assert result.total_cost_usd == pytest.approx((150 * pin + 30 * pout) / 1_000_000)


def test_model_ablation_switch(tmp_path, monkeypatch):
    """Config(model=haiku) 无需改 loop 即切换；默认是 opus。"""
    assert Config().model == "anthropic/claude-opus-4.8"
    captured = {}

    class _RecModelLLM:
        def __init__(self, config, model=None):
            captured["model"] = model or config.model
        def create(self, **kwargs):
            return _text_turn()
        def snapshot(self):
            return Usage(0, 0, 1, 0.0)
        @property
        def model(self):
            return captured["model"]

    monkeypatch.setattr("agent.loop.LLMClient", _RecModelLLM)
    cfg = Config(model=Config().model_haiku)
    run_agent(str(tmp_path), "任务", cfg)
    assert captured["model"] == cfg.model_haiku


def test_guarded_execute_receives_config_budgets(tmp_path, monkeypatch):
    """loop 把 config.run_tests_timeout_s / max_tool_result_chars 透传给 guarded_execute。"""
    script = [_tool_turn(0, name="run_tests", tool_input={}), _text_turn()]
    monkeypatch.setattr("agent.loop.LLMClient", _make_fake_llm_class(script))
    seen_kwargs = {}

    def _rec_guard(name, tool_input, workdir, **kw):
        seen_kwargs.update(kw)
        return "结果：PASS"

    monkeypatch.setattr("agent.loop.guarded_execute", _rec_guard)
    cfg = Config()
    run_agent(str(tmp_path), "任务", cfg)
    assert seen_kwargs["test_timeout"] == cfg.run_tests_timeout_s
    assert seen_kwargs["max_result_chars"] == cfg.max_tool_result_chars


def test_enable_retrieval_toggle(tmp_path, monkeypatch):
    """enable_retrieval=True → 首条 user 前置 retrieve_context 文本；False → 不触发检索。"""
    calls = {"n": 0}

    def _fake_retrieve(task, workdir, config):
        calls["n"] += 1
        return "RETRIEVED-CONTEXT"

    monkeypatch.setattr("agent.loop.retrieve_context", _fake_retrieve)
    seen = []

    class _RecLLM:
        def __init__(self, config, model=None):
            self._i = 0
        def create(self, messages, **kwargs):
            seen.append(messages[0]["content"])
            msg = _text_turn(); self._i += 1
            return msg
        def snapshot(self):
            return Usage(0, 0, self._i, 0.0)
        @property
        def model(self):
            return "x"

    monkeypatch.setattr("agent.loop.LLMClient", _RecLLM)

    run_agent(str(tmp_path), "TASKTEXT", Config(enable_retrieval=False))
    assert calls["n"] == 0             # 关：不调检索
    assert seen[-1] == "TASKTEXT"      # 首条 user 就是纯任务

    run_agent(str(tmp_path), "TASKTEXT", Config(enable_retrieval=True))
    assert calls["n"] == 1             # 开：调一次
    assert "RETRIEVED-CONTEXT" in seen[-1] and "TASKTEXT" in seen[-1]


# ---------------------------------------------------------------------------
# TODO 待补清单（逐条对齐 DESIGN §8.12 验收标准）
# ---------------------------------------------------------------------------
# TODO(集成，bench 跑): 端到端冒烟 —— 真实任务（改坏乘除优先级）观察「定位 → 改 → run_tests → 绿 → 停」，
#               每步 token/成本/工具调用被如实记录。属 eval/run_bench 的集成验证，不在离线单测。
