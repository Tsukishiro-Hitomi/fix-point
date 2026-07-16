"""求值（evaluator）+ 一站式入口。

对外公开：
  - ``Number``：``Union[int, float]`` 类型别名（返回值类型）。
  - ``eval_ast(node) -> Number``：对一棵 AST 求值（本文件为脚手架桩，未实现）。
  - ``evaluate(source) -> Number``：串起 tokenize -> parse -> eval_ast（同上，桩）。

语义要点（契约）
----------------
- 除法 ``/`` 恒为 true division，**结果总是 float**（"6 / 2" -> 3.0）。
- 整数间 ``+ - *`` 保持 int（"2 + 3" -> 5，isinstance int）。
- 除以零抛 :class:`errors.EvalError`（不放任 ZeroDivisionError 冒泡）。
"""

from typing import Union

from tokenizer import tokenize  # noqa: F401  (evaluate 实现所需)
from parser import parse, Node  # noqa: F401  (parse 供 evaluate 实现；Node 用于注解)
from errors import EvalError  # noqa: F401  (eval_ast 实现所需)

Number = Union[int, float]


def eval_ast(node: Node) -> Number:
    """对一棵 AST 节点递归求值，返回数值。

    行为契约（按节点形状分派）
    --------------------------
    - ("num", v)          -> 直接返回 ``v``（int 或 float）。
    - ("binop", op, l, r) -> 先递归求 ``l``、``r``：
        * op == "/"：右操作数为 0 时抛 :class:`errors.EvalError`；否则 true
          division，结果为 ``float``。
        * op in {"+","-","*"}：用 Python 同名运算（int op int 仍为 int）。
    - ("neg", x)          -> ``-eval_ast(x)``。

    参数:
        node: 一棵 AST 节点（见 parser 模块的三种形状）。
    返回:
        求值结果（int 或 float）。
    异常:
        errors.EvalError: 除以零时。
    """
    tag = node[0]
    
    if tag == "num":
        return node[1]
    elif tag == "binop":
        op = node[1]
        left = eval_ast(node[2])
        right = eval_ast(node[3])
        if op == "/" and right == 0:
            raise EvalError("除法错误：除数为0")
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return float(left) / right
    else:
        return -eval_ast(node[1])



def evaluate(source: str) -> Number:
    """一站式入口：串起 ``tokenize(source) -> parse(...) -> eval_ast(...)``。

    **不吞异常**：``LexError`` / ``ParseError`` / ``EvalError`` 原样上抛（这是
    下游任务/测试要断言的传播行为）。用 ``eval_ast`` 而非内建 ``eval``。

    参数:
        source: 源字符串（如 "-(1 + 2) * 3"）。
    返回:
        求值结果（int 或 float）。
    异常:
        errors.LexError:   词法期（非法字符）。
        errors.ParseError: 语法期（不符合文法）。
        errors.EvalError:  求值期（除以零）。
    """
    tokens = tokenize(source)
    node = parse(tokens)
    return eval_ast(node)
