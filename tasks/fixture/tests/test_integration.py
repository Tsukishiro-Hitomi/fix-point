"""端到端集成测试 —— ``evaluate()`` 从源字符串直达数值，并验证异常原样传播。

纯净态应全绿；当前 fixture 为桩（evaluate 抛 NotImplementedError），故全红。
"""

import pytest

from evaluator import evaluate
from errors import LexError, ParseError, EvalError


def test_simple():
    assert evaluate("1 + 1") == 2


def test_precedence():
    assert evaluate("2 + 3 * 4") == 14


def test_parens():
    assert evaluate("(2 + 3) * 4") == 20


def test_unary_and_parens():
    assert evaluate("-(3 + 1)") == -4


def test_float_result():
    result = evaluate("6 / 4")
    assert result == 1.5
    assert isinstance(result, float)


def test_whitespace():
    assert evaluate("  7  -  2 ") == 5


def test_deep_nesting():
    assert evaluate("((1 + 2) * (3 - 1)) / 2") == 3.0


def test_lex_error_propagates():
    with pytest.raises(LexError):
        evaluate("2 @ 2")


def test_parse_error_propagates():
    with pytest.raises(ParseError):
        evaluate("(1 + 2")


def test_div_zero_propagates():
    with pytest.raises(EvalError):
        evaluate("1 / (2 - 2)")
