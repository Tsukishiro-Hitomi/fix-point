"""parser 层测试 —— 输入用 ``tokenize()`` 构造，断言 AST 结构与语法报错。

覆盖优先级、左结合、括号改写、一元负号，以及各类语法错误。
纯净态应全绿；当前 fixture 为桩（parse 抛 NotImplementedError），故全红。
"""

import pytest

from tokenizer import tokenize
from parser import parse
from errors import ParseError


def test_single_number():
    assert parse(tokenize("5")) == ("num", 5)


def test_addition_node():
    assert parse(tokenize("1 + 2")) == ("binop", "+", ("num", 1), ("num", 2))


def test_precedence_mul_over_add():
    # 加法为根，右子树是乘法（* 优先级高于 +）
    assert parse(tokenize("1 + 2 * 3")) == (
        "binop", "+", ("num", 1), ("binop", "*", ("num", 2), ("num", 3))
    )


def test_precedence_div_over_sub():
    # 减法为根，右子树是除法
    assert parse(tokenize("10 - 8 / 2")) == (
        "binop", "-", ("num", 10), ("binop", "/", ("num", 8), ("num", 2))
    )


def test_left_assoc_sub():
    # "10 - 2 - 3" 左结合 -> ((10 - 2) - 3)
    assert parse(tokenize("10 - 2 - 3")) == (
        "binop", "-", ("binop", "-", ("num", 10), ("num", 2)), ("num", 3)
    )


def test_left_assoc_div():
    # "8 / 4 / 2" 左结合 -> ((8 / 4) / 2)
    assert parse(tokenize("8 / 4 / 2")) == (
        "binop", "/", ("binop", "/", ("num", 8), ("num", 4)), ("num", 2)
    )


def test_parens_override():
    # 括号改写优先级：乘法为根，左子树是加法
    assert parse(tokenize("(1 + 2) * 3")) == (
        "binop", "*", ("binop", "+", ("num", 1), ("num", 2)), ("num", 3)
    )


def test_nested_parens():
    assert parse(tokenize("((1))")) == ("num", 1)


def test_unary_minus():
    assert parse(tokenize("-3")) == ("neg", ("num", 3))


def test_unary_minus_after_op():
    assert parse(tokenize("2 * -3")) == (
        "binop", "*", ("num", 2), ("neg", ("num", 3))
    )


def test_double_unary_minus():
    assert parse(tokenize("--3")) == ("neg", ("neg", ("num", 3)))


def test_missing_closing_paren_raises():
    with pytest.raises(ParseError):
        parse(tokenize("(1 + 2"))


def test_extra_closing_paren_raises():
    with pytest.raises(ParseError):
        parse(tokenize("1 + 2)"))


def test_trailing_tokens_raise():
    with pytest.raises(ParseError):
        parse(tokenize("1 2"))


def test_empty_raises():
    # tokenize("") == [Token("EOF", None)]，交给 parse 应抛 ParseError
    with pytest.raises(ParseError):
        parse(tokenize(""))


def test_double_operator_raises():
    with pytest.raises(ParseError):
        parse(tokenize("1 + * 2"))


def test_unary_plus_not_supported():
    with pytest.raises(ParseError):
        parse(tokenize("+1"))


def test_trailing_operator_raises():
    with pytest.raises(ParseError):
        parse(tokenize("1 +"))
