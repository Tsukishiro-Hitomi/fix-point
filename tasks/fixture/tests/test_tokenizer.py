"""tokenizer 层测试 —— 断言 Token 列表的结构、数值类型与词法报错。

纯净态应全绿；当前 fixture 为桩（tokenize 抛 NotImplementedError），故全红。
"""

import pytest

from tokenizer import tokenize, Token
from errors import LexError


def test_single_integer():
    assert tokenize("42") == [Token("NUMBER", 42), Token("EOF", None)]


def test_multi_digit_number():
    tokens = tokenize("123")
    # "123" 聚合为一个 NUMBER（值 123），不拆成 1/2/3
    assert tokens == [Token("NUMBER", 123), Token("EOF", None)]
    numbers = [t for t in tokens if t.type == "NUMBER"]
    assert len(numbers) == 1
    assert numbers[0].value == 123


def test_float_number():
    tokens = tokenize("3.14")
    assert tokens[0].type == "NUMBER"
    assert tokens[0].value == 3.14
    assert isinstance(tokens[0].value, float)


def test_integer_value_is_int():
    assert isinstance(tokenize("7")[0].value, int)
    assert isinstance(tokenize("7.0")[0].value, float)


def test_operator_tokens():
    tokens = tokenize("+ - * /")
    assert tokens[:4] == [
        Token("PLUS", "+"),
        Token("MINUS", "-"),
        Token("STAR", "*"),
        Token("SLASH", "/"),
    ]
    assert tokens[-1] == Token("EOF", None)


def test_paren_tokens():
    tokens = tokenize("( )")
    assert tokens[:2] == [Token("LPAREN", "("), Token("RPAREN", ")")]


def test_whitespace_ignored():
    tokens = tokenize("  1 \t + \n 2 ")
    assert [t.type for t in tokens] == ["NUMBER", "PLUS", "NUMBER", "EOF"]


def test_eof_appended():
    for src in ["1 + 2", "42", "(3)"]:
        tokens = tokenize(src)
        assert tokens[-1].type == "EOF"
        assert tokens[-1].value is None
        assert [t.type for t in tokens].count("EOF") == 1


def test_empty_source():
    assert tokenize("") == [Token("EOF", None)]


def test_full_sequence():
    assert tokenize("3*(4+2)") == [
        Token("NUMBER", 3),
        Token("STAR", "*"),
        Token("LPAREN", "("),
        Token("NUMBER", 4),
        Token("PLUS", "+"),
        Token("NUMBER", 2),
        Token("RPAREN", ")"),
        Token("EOF", None),
    ]


def test_illegal_char_raises():
    with pytest.raises(LexError):
        tokenize("3 $ 4")


def test_bare_dot_raises():
    with pytest.raises(LexError):
        tokenize(".5")
    with pytest.raises(LexError):
        tokenize("5.")


def test_letters_raise():
    with pytest.raises(LexError):
        tokenize("1e3")
    with pytest.raises(LexError):
        tokenize("abc")
