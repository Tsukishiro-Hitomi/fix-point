"""evaluator 层测试 —— **手工构造 AST**，与 tokenizer/parser 完全解耦。

如此挖空 ``eval_ast`` 某一分支时，只会红到 evaluator 对应用例（+ 依赖它的
integration 用例），parser/tokenizer 全绿。
纯净态应全绿；当前 fixture 为桩（eval_ast 抛 NotImplementedError），故全红。
"""

import pytest

from evaluator import eval_ast
from errors import EvalError


def test_eval_number():
    assert eval_ast(("num", 5)) == 5


def test_eval_addition():
    assert eval_ast(("binop", "+", ("num", 1), ("num", 2))) == 3


def test_eval_subtraction():
    assert eval_ast(("binop", "-", ("num", 10), ("num", 3))) == 7


def test_eval_multiplication():
    assert eval_ast(("binop", "*", ("num", 4), ("num", 5))) == 20


def test_eval_division_is_float():
    result = eval_ast(("binop", "/", ("num", 6), ("num", 2)))
    assert result == 3.0
    assert isinstance(result, float)


def test_eval_division_fractional():
    assert eval_ast(("binop", "/", ("num", 6), ("num", 4))) == 1.5


def test_eval_division_by_zero_raises():
    with pytest.raises(EvalError):
        eval_ast(("binop", "/", ("num", 1), ("num", 0)))


def test_eval_unary_negation():
    assert eval_ast(("neg", ("num", 5))) == -5


def test_eval_nested_tree():
    # 手搭 "1 + 2 * 3" 的树
    tree = ("binop", "+", ("num", 1), ("binop", "*", ("num", 2), ("num", 3)))
    assert eval_ast(tree) == 7


def test_int_arithmetic_stays_int():
    assert isinstance(eval_ast(("binop", "+", ("num", 2), ("num", 3))), int)
