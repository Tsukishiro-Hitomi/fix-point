"""把本目录（tasks/fixture/）加入 sys.path，让 tests/ 里的裸 import 解析成功。

DESIGN §4.1：fixture 是被操作的目标代码（扁平模块，非 package，无 __init__.py），
测试用 `import tokenizer` / `import parser` / `import evaluator` / `import errors`
直接引用。pytest 会加载本 conftest 并（在 prepend 导入模式下）把其所在目录插入
sys.path；此处再显式插入一次，确保无论从何处、以何种 import 模式运行 pytest 都成立。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
