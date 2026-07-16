"""语法分析（递归下降 parser）—— 把 Token 列表规约成一棵 AST。

对外公开：
  - ``Node``：AST 节点的类型别名（嵌套 ``tuple``，见下）。
  - ``parse(tokens) -> Node``：解析入口（本文件为脚手架桩，未实现）。

AST 形状（契约，测试直接断言）
------------------------------
用嵌套 tuple 表示（轻量、易断言，不引 dataclass），三种节点：

    数字节点:  ("num", value)              # value 为 int 或 float
    二元运算:  ("binop", op, left, right)  # op ∈ {"+","-","*","/"}；left/right 为 Node
    一元负号:  ("neg", operand)            # operand 为 Node

示例:
    "1 + 2 * 3"  -> ("binop", "+", ("num", 1), ("binop", "*", ("num", 2), ("num", 3)))
    "-(3 + 1)"   -> ("neg", ("binop", "+", ("num", 3), ("num", 1)))

文法（递归下降，直接体现优先级与结合性）
----------------------------------------
    expr    := term (("+" | "-") term)*        # 加减，最低优先级
    term    := factor (("*" | "/") factor)*    # 乘除，高一级
    factor  := "-" factor | primary            # 一元负号（右递归，故 "--3" 合法）
    primary := NUMBER | "(" expr ")"

``(...)*`` 循环天然左结合；一元负号在 factor 层，优先级高于二元运算，且可紧跟
运算符之后（"2 * -3" 合法）。注意：只有一元负号，无一元正号（"+1" 非法）。
"""

from typing import Any, Tuple

from tokenizer import Token
from errors import ParseError  # noqa: F401  (parse 实现所需；此处固定契约依赖)

# AST 节点：三种嵌套 tuple 之一（形状见模块 docstring）。
Node = Tuple[Any, ...]


def parse(tokens: list[Token]) -> Node:
    """把 Token 列表按文法规约成单个根 AST 节点（见模块 docstring 的三种形状）。

    行为契约
    --------
    - 维护一个游标指向当前 token；``expr/term/factor/primary`` 四个内部函数互相
      递归，用 ``(...)*`` 循环消费实现左结合（"10 - 2 - 3" 解析为 ((10-2)-3)）。
    - 优先级：乘除（term）高于加减（expr）；一元负号（factor）高于二元运算。
    - 解析完顶层 ``expr`` 后，当前 token **必须是 EOF**；否则（有残留 token，
      如 "1 2"、"1 + 2)"）抛 :class:`errors.ParseError`。
    - ``primary`` 期望 NUMBER 或 "("，遇到别的（运算符、EOF 等）抛 ParseError；
      "(" 之后必须配到对应的 ")"，否则抛 ParseError。
    - 仅输入 ``[Token("EOF", None)]``（空表达式）时抛 ParseError。
    - 只接受 Token 列表，与 tokenizer 解耦（一站式入口在 ``evaluator.evaluate``）。

    参数:
        tokens: ``tokenize()`` 产出的 Token 列表（末尾含 EOF）。
    返回:
        根 AST 节点（Node）。
    异常:
        errors.ParseError: token 序列不符合文法时（残留 token、括号不匹配、空
            输入、悬空/相邻运算符、不支持的一元正号等）。
    """
    # 游标
    pos = 0
    def peek():
        return tokens[pos]
    def advance():
        nonlocal pos    # 要求函数闭包读外部变量
        t = tokens[pos]
        pos += 1
        return t
    def primary():
        t = peek()
        if t.type == "NUMBER":
            advance()
            return ("num", t.value)
        if t.type == "LPAREN":
            advance()
            node = expr()    
            if peek().type != "RPAREN":     # 括号未闭合
                raise ParseError(f"这里 token.type 应该为 ""RPAREN""，实际为{t.type}")
            advance()
            return node
        raise ParseError(f"这里 token.type 应该为 ""NUMBER"" 或 ""LPAREN""，实际为{t.type}")
    def factor():
        t = peek()
        if t.type == "MINUS":
            advance()
            return ("neg", factor())
        else:
            return primary()
    def term():
        left = factor()
        while peek().type in ("STAR", "SLASH"):
            op = advance().value                       
            right = factor()
            left = ("binop", op, left, right)      
        return left  
    def expr():
        left = term()
        while peek().type in ("PLUS", "MINUS"):
            op = advance().value                      
            right = term()
            left = ("binop", op, left, right)     
        return left

    node = expr()
    if peek().type != "EOF":
        raise ParseError(f"表达式后面还有多余的东西：{peek().type}")
    return node