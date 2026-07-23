"""tests/test_llm.py —— agent/llm.py 的起步测试集（DESIGN §7.5）。

策略（见脚手架说明）：
- **可运行**的真实断言：``Usage`` 的声明式契约（字段名/顺序/None 语义）——现在就能绿。
- **记账 / 成本数学**的期望：用**假的 anthropic client**（假 ``Message`` / ``usage``）离线驱动，
  写出**具体期望值**的真实断言；但因 ``LLMClient.__init__ / create / snapshot`` 目前是桩
  （``raise NotImplementedError``），这些用例先 ``@pytest.mark.skip`` 标 TODO——**实现后取消
  skip 即成为回归护栏**。
- ``create`` 的**真实网关调用属集成测试**，不在此单测里打真网关（见文末 TODO）。

所有假对象只依赖 ``response.usage.input_tokens/output_tokens``——与 llm 的记账口径一致，
无需真实网络。
"""

from __future__ import annotations

import types

import pytest

from agent.llm import LLMClient, Usage

# 说明：Config 是 config.py（他人负责）里的声明式 dataclass，构造 Config() 即拿默认旋钮。
from agent.config import Config

_TODO = "TODO(实现后取消 skip)：LLMClient.__init__/create/snapshot 目前是桩"


# ---------------------------------------------------------------------------
# 假 anthropic client 基建（假 Message / usage；不触网络）
# ---------------------------------------------------------------------------
class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    """最小 Message 替身：只带记账/收尾所需字段。"""

    def __init__(self, *, content=None, stop_reason="end_turn", usage=None):
        self.content = content if content is not None else []
        self.stop_reason = stop_reason
        self.usage = usage


class _FakeStream:
    """messages.stream(...) 的上下文管理器替身；get_final_message 返回同型 Message。"""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.create_calls = []   # 记录每次 create 的 kwargs（用于「仅调用一次」等边界断言）
        self.stream_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._scripted.pop(0)

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _FakeStream(self._scripted.pop(0))


@pytest.fixture
def install_fake_anthropic(monkeypatch):
    """返回一个安装器：给定脚本化的假 Message 列表，替换 ``agent.llm.anthropic.Anthropic``，
    并设好环境变量以免构造期快速失败。返回被构造出的假 client（可查 _init_kwargs / messages）。"""

    def _install(scripted_messages):
        fake_client = types.SimpleNamespace(
            messages=_FakeMessages(scripted_messages), _init_kwargs=None
        )

        def _factory(**kwargs):
            fake_client._init_kwargs = kwargs
            return fake_client

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gateway.test")
        monkeypatch.setattr("agent.llm.anthropic.Anthropic", _factory)
        return fake_client

    return _install


# ---------------------------------------------------------------------------
# 可运行的真实断言：Usage 声明式契约（DESIGN §7.2）——现在即绿
# ---------------------------------------------------------------------------
def test_usage_fields_contract():
    assert Usage._fields == ("input_tokens", "output_tokens", "calls", "cost_usd")
    u = Usage(157, 223, 3, 0.00636)
    assert (u.input_tokens, u.output_tokens, u.calls, u.cost_usd) == (157, 223, 3, 0.00636)
    # 模型缺表时成本以 None 表达（区分「零成本」与「无价可算」）
    assert Usage(0, 0, 0, None).cost_usd is None


# ---------------------------------------------------------------------------
# 记账 / 成本数学：真实断言，先 skip（实现后取消）——DESIGN §7.5
# ---------------------------------------------------------------------------
def test_construct_wires_sdk_without_secrets(install_fake_anthropic):
    """构造只经 timeout/max_retries 注入 SDK；不手传 api_key/base_url（DESIGN §7.3）。"""
    fake = install_fake_anthropic([])
    cfg = Config()
    LLMClient(cfg)
    assert fake._init_kwargs == {"timeout": cfg.timeout_s, "max_retries": cfg.max_retries}
    assert "api_key" not in fake._init_kwargs
    assert "base_url" not in fake._init_kwargs


def test_token_accumulation_and_calls(install_fake_anthropic):
    """连续 N 次 create 后累计等于各次之和，calls==N；reset 后归零（DESIGN §7.5）。"""
    scripted = [
        _FakeMessage(usage=_FakeUsage(100, 20)),
        _FakeMessage(usage=_FakeUsage(50, 200)),
        _FakeMessage(usage=_FakeUsage(7, 3)),
    ]
    install_fake_anthropic(scripted)
    client = LLMClient(Config())
    for _ in range(3):
        client.create(messages=[{"role": "user", "content": "hi"}], stream=False)

    snap = client.snapshot()
    assert snap.input_tokens == 157
    assert snap.output_tokens == 223
    assert snap.calls == 3

    client.reset()
    zeroed = client.snapshot()
    assert (zeroed.input_tokens, zeroed.output_tokens, zeroed.calls) == (0, 0, 0)


def test_cost_math_default_opus(install_fake_anthropic):
    """成本 = Σ(in/1e6·价in + out/1e6·价out)，按当前 model 价格表（DESIGN §7.5）。"""
    install_fake_anthropic([_FakeMessage(usage=_FakeUsage(157, 223))])
    cfg = Config()
    client = LLMClient(cfg)
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)

    pin, pout = cfg.price_per_mtok[cfg.model]           # 期望值从价格表推导，避免硬编码脆弱数字
    expected = (157 * pin + 223 * pout) / 1_000_000
    assert client.snapshot().cost_usd == pytest.approx(expected)


def test_model_switch_uses_haiku_price(install_fake_anthropic):
    """传 config.model_haiku 即切消融模型，记账用 haiku 价（DESIGN §7.5）。"""
    install_fake_anthropic([_FakeMessage(usage=_FakeUsage(1000, 1000))])
    cfg = Config()
    client = LLMClient(cfg, cfg.model_haiku)
    assert client.model == cfg.model_haiku
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)

    pin, pout = cfg.price_per_mtok[cfg.model_haiku]
    expected = (1000 * pin + 1000 * pout) / 1_000_000
    assert client.snapshot().cost_usd == pytest.approx(expected)


def test_unknown_model_cost_is_none_and_warns(install_fake_anthropic, caplog):
    """模型不在价格表：构造不报错，cost_usd is None 且有 warning、不崩（DESIGN §7.3/§7.5）。"""
    install_fake_anthropic([_FakeMessage(usage=_FakeUsage(10, 10))])
    client = LLMClient(Config(), "anthropic/does-not-exist")
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)
    assert client.snapshot().cost_usd is None
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_usage_none_is_robust(install_fake_anthropic, caplog):
    """usage 为 None 的假响应：不抛异常、只 warning、token 累加器不变（DESIGN §7.5 健壮性）。

    TODO(实现者定夺并钉死)：usage 缺失时 calls 计数是否仍 +1？DESIGN §7.2 说 calls 是
    「成功返回的请求次数」，§7.5 又说「累加器不变」——请实现时明确取舍并在此补断言。
    """
    install_fake_anthropic([_FakeMessage(usage=None)])
    client = LLMClient(Config())
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)  # 不应抛异常
    snap = client.snapshot()
    assert snap.input_tokens == 0
    assert snap.output_tokens == 0
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_create_calls_underlying_once_and_returns_verbatim(install_fake_anthropic):
    """边界纪律：create 只调用一次底层 messages.create，原样返回该 Message
    （不拆包、不改写、不分派 stop_reason、不执行工具）——DESIGN §7.5。"""
    msg = _FakeMessage(stop_reason="tool_use", usage=_FakeUsage(5, 5))
    fake = install_fake_anthropic([msg])
    client = LLMClient(Config())
    returned = client.create(messages=[{"role": "user", "content": "hi"}], stream=False)
    assert len(fake.messages.create_calls) == 1
    assert len(fake.messages.stream_calls) == 0
    assert returned is msg


def test_stream_default_from_config_and_same_type(install_fake_anthropic):
    """stream 缺省取 config.stream；stream=True 走 messages.stream 且返回同型 Message、
    记账一致（DESIGN §7.3/§7.5 流式一致性）。"""
    msg = _FakeMessage(usage=_FakeUsage(5, 5))
    fake = install_fake_anthropic([msg])
    cfg = Config()                                       # 默认 stream=True
    client = LLMClient(cfg)
    returned = client.create(messages=[{"role": "user", "content": "hi"}])  # 不显式传 stream
    assert returned is msg
    assert len(fake.messages.stream_calls) == 1          # 走了流式路径
    assert len(fake.messages.create_calls) == 0
    assert client.snapshot().input_tokens == 5


def test_missing_api_key_fails_fast(monkeypatch):
    """缺 ANTHROPIC_API_KEY 时构造快速失败并给清晰报错（提到该变量 / .env）——DESIGN §7.3。"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(Exception) as ei:
        LLMClient(Config())
    assert "ANTHROPIC_API_KEY" in str(ei.value) or ".env" in str(ei.value)


def test_create_includes_optional_params_only_when_set(install_fake_anthropic):
    """system/tools/tool_choice 仅在非 None 时放入；model 与 max_tokens 恒定传入。"""
    fake = install_fake_anthropic([
        _FakeMessage(usage=_FakeUsage(1, 1)),
        _FakeMessage(usage=_FakeUsage(1, 1)),
    ])
    cfg = Config()
    client = LLMClient(cfg)

    # 只给 messages
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)
    kw = fake.messages.create_calls[0]
    assert kw["model"] == cfg.model            # 恒传
    assert kw["max_tokens"] == cfg.max_tokens  # 恒传
    assert "messages" in kw
    assert "system" not in kw                  # 未给 → 不放入
    assert "tools" not in kw
    assert "tool_choice" not in kw

    # 给了 system/tools/tool_choice
    client.create(
        messages=[{"role": "user", "content": "hi"}],
        system="be brief",
        tools=[{"name": "x"}],
        tool_choice={"type": "auto"},
        stream=False,
    )
    kw2 = fake.messages.create_calls[1]
    assert kw2["system"] == "be brief"
    assert kw2["tools"] == [{"name": "x"}]
    assert kw2["tool_choice"] == {"type": "auto"}


def test_cache_fields_do_not_affect_cost(install_fake_anthropic):
    """usage 带 cache_* 字段时，成本与 token 累计只认 input/output 两项。"""
    usage = _FakeUsage(157, 223)
    usage.cache_creation_input_tokens = 999  # 额外字段，不应参与计价
    usage.cache_read_input_tokens = 888
    install_fake_anthropic([_FakeMessage(usage=usage)])
    cfg = Config()
    client = LLMClient(cfg)
    client.create(messages=[{"role": "user", "content": "hi"}], stream=False)

    pin, pout = cfg.price_per_mtok[cfg.model]
    expected = (157 * pin + 223 * pout) / 1_000_000
    snap = client.snapshot()
    assert snap.cost_usd == pytest.approx(expected)
    assert (snap.input_tokens, snap.output_tokens) == (157, 223)


def test_api_error_propagates_not_swallowed(install_fake_anthropic):
    """底层 messages.create 抛异常时，create 原样上抛、不假装成功，累加器不动。"""
    fake = install_fake_anthropic([])
    client = LLMClient(Config())

    class _Boom(Exception):  # 用哨兵异常，避免依赖具体 anthropic 异常构造签名
        pass

    def boom(**kwargs):
        raise _Boom("gateway down")

    fake.messages.create = boom
    with pytest.raises(_Boom):
        client.create(messages=[{"role": "user", "content": "hi"}], stream=False)
    assert client.snapshot().calls == 0  # 没吞、没假装成功


# ---------------------------------------------------------------------------
# TODO 待补清单（覆盖 DESIGN §7.5 剩余验收项 + 集成边界）
# ---------------------------------------------------------------------------
# TODO(你来补): 重试/超时——不自写 retry 循环，交给 SDK（max_retries/timeout 经 config 注入，
#               已由 test_construct_wires_sdk_without_secrets 覆盖）；源码扫 for/while 的断言略（脆弱、低价值）。
# TODO(集成，不在单测): create 的真实网关调用——设好 ANTHROPIC_API_KEY/BASE_URL 后
#               create(messages=[{"role":"user","content":"ping"}]) 返回带非空 .content/.stop_reason/.usage
#               的 Message；工具穿透（传 tools + 触发 prompt → stop_reason=="tool_use" 且含 tool_use 块）。
#               这些打真网关、须凭据，标记为集成/手动跑，**不进离线单测**。
