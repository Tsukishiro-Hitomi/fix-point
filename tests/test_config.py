"""agent.config 的起步测试。

分两类：
- **声明式**（字段默认值、价格表）在 config.py 已写全，故这里是真实断言、当下即绿，
  钉死 DESIGN.md §8.1 的默认值口径。
- **逻辑**（from_env 映射、cost_of 计算）在 config.py 仍是 NotImplementedError 桩，
  其测试已按「已知输入 -> 期望值」写出但整类 skip；实现后移除 skip 即成红灯规格。
"""

import pytest

from agent.config import Config, cost_of


class TestDefaults:
    """§8.1 默认值口径（跨章唯一，其它模块引用同一数字）。"""

    def test_brain_defaults(self):
        c = Config()
        assert c.model == "anthropic/claude-opus-4.8"
        assert c.model_haiku == "anthropic/claude-haiku-4.5"
        assert c.max_tokens == 8192
        assert c.stream is True
        assert c.timeout_s == 120.0
        assert c.max_retries == 2

    def test_guardrail_defaults(self):
        c = Config()
        assert c.max_steps == 30
        assert c.cost_budget_usd == 0.50
        assert c.run_tests_timeout_s == 60
        assert c.judge_timeout_s == 60

    def test_capability_switches_default_off(self):
        c = Config()
        assert c.enable_retrieval is False
        assert c.self_correction is False

    def test_truncation_budgets(self):
        c = Config()
        assert c.max_tool_result_chars == 8000
        assert c.max_read_lines == 400
        assert c.max_search_hits == 100
        assert c.max_test_output == 4000

    def test_price_table_contents(self):
        c = Config()
        assert c.price_per_mtok["anthropic/claude-opus-4.8"] == (5.0, 25.0)
        assert c.price_per_mtok["anthropic/claude-haiku-4.5"] == (1.0, 5.0)

    def test_price_table_is_per_instance(self):
        # 易错点：dataclass 的可变默认必须用 default_factory，两个实例不能共享同一 dict。
        a, b = Config(), Config()
        assert a.price_per_mtok is not b.price_per_mtok
        a.price_per_mtok["anthropic/tmp"] = (0.0, 0.0)
        assert "anthropic/tmp" not in b.price_per_mtok


class TestCostOf:
    """§8.2 唯一计价函数：(in*pin + out*pout) / 1_000_000。"""

    def test_default_model_known_inputs(self):
        c = Config()  # 默认 model = opus (5.0, 25.0)
        # (1_000_000*5 + 1_000_000*25) / 1e6 = 30.0
        assert cost_of(1_000_000, 1_000_000, c) == pytest.approx(30.0)
        # (200_000*5 + 100_000*25) / 1e6 = 3.5
        assert cost_of(200_000, 100_000, c) == pytest.approx(3.5)

    def test_explicit_model_overrides_config(self):
        c = Config()
        # haiku (1.0, 5.0): (1e6*1 + 1e6*5) / 1e6 = 6.0
        got = cost_of(1_000_000, 1_000_000, c, model="anthropic/claude-haiku-4.5")
        assert got == pytest.approx(6.0)

    def test_zero_tokens(self):
        assert cost_of(0, 0, Config()) == pytest.approx(0.0)

    def test_missing_model_returns_none(self):
        # 缺表 -> None（调用方按 0 展示 + 告警），绝不崩溃。
        assert cost_of(100, 100, Config(), model="anthropic/does-not-exist") is None

class TestFromEnv:
    """§14.4 env -> 字段映射；密钥绝不进 Config。"""

    def test_overrides_mapped_fields(self, monkeypatch):
        monkeypatch.setenv("FIXPOINT_MODEL", "anthropic/claude-haiku-4.5")
        monkeypatch.setenv("MAX_STEPS", "7")
        monkeypatch.setenv("RUN_TESTS_TIMEOUT", "12")
        c = Config.from_env()
        assert c.model == "anthropic/claude-haiku-4.5"
        assert c.max_steps == 7
        assert c.run_tests_timeout_s == 12

    def test_unset_env_keeps_defaults(self, monkeypatch):
        monkeypatch.delenv("FIXPOINT_MODEL", raising=False)
        monkeypatch.delenv("MAX_STEPS", raising=False)
        monkeypatch.delenv("RUN_TESTS_TIMEOUT", raising=False)
        c = Config.from_env()
        assert c.model == "anthropic/claude-opus-4.8"
        assert c.max_steps == 30
        assert c.run_tests_timeout_s == 60

    def test_secrets_not_absorbed(self, monkeypatch):
        # 密钥由 SDK 直接读环境，绝不进 Config（Config 无对应字段、repr 不含真实值）。
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        c = Config.from_env()
        assert not hasattr(c, "api_key")
        assert "sk-should-not-leak" not in repr(c)


# TODO(你来补): 其余待补项
#   - from_env 对非法数值（如 MAX_STEPS="abc"）的处理约定（DESIGN 未明确，需与 §8.1 对齐后补）
#   - cost_of 非默认 model 参数与缺表告警路径在 llm/loop 里的联动（属对应模块测试）
