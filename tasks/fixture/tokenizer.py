"""词法分析（tokenizer）—— 把源字符串扫描成 Token 列表。

对外公开：
  - ``Token``：``namedtuple("Token", ["type", "value"])``（数据表示，声明式，见下）。
  - ``tokenize(source) -> list[Token]``：扫描入口（本文件为脚手架桩，未实现）。

数据表示（契约，测试直接断言）
------------------------------
``Token.type`` 为以下大写字符串之一：
    "NUMBER" | "PLUS" | "MINUS" | "STAR" | "SLASH" | "LPAREN" | "RPAREN" | "EOF"
``Token.value``：
    - NUMBER        -> 已解析的数值：``int``（如 "12"）或 ``float``（如 "3.14"）。
    - 运算符 / 括号 -> 原始词素字符串："+" "-" "*" "/" "(" ")"。
    - EOF           -> ``None``。
"""

from collections import namedtuple

from errors import LexError  # noqa: F401  (tokenize 实现所需；此处固定契约依赖)

Token = namedtuple("Token", ["type", "value"])


def tokenize(source: str) -> list[Token]:
    r"""把源字符串扫描成 Token 列表，末尾恰好追加一个 ``Token("EOF", None)``。

    行为契约
    --------
    - 从左到右扫描；空白字符（``str.isspace()``，含空格/制表/换行）一律跳过。
    - 连续数字聚合成**一个** NUMBER token（"123" -> 一个值为 123 的 NUMBER，
      不拆成 1/2/3）。字面量形如正则 ``\d+(\.\d+)?``：
        * 恰带一个 ``.`` 且两侧都有数字 -> 解析为 ``float``（"3.14"）。
        * 无 ``.``                      -> 解析为 ``int``（"12"）。
      数值类型由此写入 ``token.value``。
    - ``+ - * / ( )`` 各产出对应类型的单字符 token，``value`` 为其原始词素。
    - 扫描结束后在列表末尾追加 ``Token("EOF", None)``。

    边界与异常
    ----------
    - 空串或纯空白 -> 返回 ``[Token("EOF", None)]``（合法词法，仅一个 EOF）。
    - 前导/尾随点（".5"、"5."）、孤立 ``.``、字母（"abc"，"1e3" 中的 e）、以及
      ``$ @ &`` 等未知字符 -> 抛 :class:`errors.LexError`，
      message 需指明**出错字符与位置**。

    参数:
        source: 待扫描的源字符串。
    返回:
        Token 列表，末尾恰含一个 ``Token("EOF", None)``。
    异常:
        errors.LexError: 遇到非法/未知字符时。
    """
    i = 0
    dic = {"+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", "(": "LPAREN", ")": "RPAREN"}
    result = []
    while i < len(source):
        c = source[i]
        if str.isspace(c):
            i += 1
            continue
        elif c in dic:
            result.append(Token(dic[c], c))
            i += 1
        elif c >= "0" and c <= "9":
            num = ""
            is_float = False
            j = i
            while j < len(source) and source[j] >= "0" and source[j] <= "9":
                num += source[j]
                j += 1
            if j < len(source) and source[j] == ".":
                num += source[j]
                j += 1
                if j == len(source):
                    raise LexError(f"source 第{j - 1}个字符非法：{source[j - 1]}")
                elif source[j] < "0" or source[j] > "9":
                    raise LexError(f"source 第{j}个字符非法：{source[j]}")
                else:
                    while j < len(source) and source[j] >= "0" and source[j] <= "9":
                        num += source[j]
                        j += 1
                num = float(num)
                is_float = True
            if not is_float:
                num = int(num)
            result.append(Token("NUMBER", num))
            i = j
        else:
            raise LexError(f"source 第{i}个字符非法：{c}")
    result.append(Token("EOF", None))
    return result

