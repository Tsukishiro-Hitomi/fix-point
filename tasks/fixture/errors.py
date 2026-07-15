"""自定义异常层级 —— 迷你表达式求值器三层共用。

三层（tokenizer / parser / evaluator）分别抛出各自的错误类型，全部继承自
``CalcError``；因此下游只需 ``except CalcError`` 即可一网打尽，同时又能按类型
精确区分错误发生在哪一层（这是任务/测试要断言的行为）。

按契约，实例化时应带上**人类可读的 message**（含出错位置/字符/token），以便
agent 从 pytest 输出里读懂哪里坏了。异常类本身无自定义逻辑，仅继承。
"""


class CalcError(Exception):
    """所有求值器错误的基类；下游 ``except CalcError`` 可捕获全部三层错误。"""


class LexError(CalcError):
    """词法错误：由 ``tokenizer.tokenize`` 遇到非法字符时抛出。"""


class ParseError(CalcError):
    """语法错误：由 ``parser.parse`` 遇到不符合文法的 token 序列时抛出。"""


class EvalError(CalcError):
    """求值错误：由 ``evaluator.eval_ast`` 在求值期检测到（如除以零）时抛出。"""
