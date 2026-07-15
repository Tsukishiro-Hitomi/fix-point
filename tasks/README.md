# tasks/ —— 任务作者指南

任务集是 fixpoint 的「考卷」。每道题是一次受控实验：从纯净 `tasks/fixture/`
（§4 的迷你表达式求值器，pytest 全绿）出发，用一个 `break.patch` 把某个函数
**改坏**或**挖空**，让预先指定的测试变红；agent 的任务是把红改回绿。判定权始终
在 harness 手里——它独立复跑 `pytest`，**绝不采信 agent 自述**。

本文件是给**任务作者**（也就是你）的操作手册，对应 DESIGN 的 §9。

> ⚠️ **两条铁律**
> 1. **纯净 `fixture/` 永不原地改动**——它是唯一真相源。所有「改坏」动作都在
>    临时副本里做，改完只留下一个 `break.patch`。
> 2. **对错只由独立复跑决定**——agent 循环里自己跑的 `run_tests` 只是它的红绿
>    反馈；最终 solved 由 harness 在 agent 停手后用**受保护的原版测试**重跑判定。

---

## 0. 当前状态：break.patch 待补（M0 之后）

本目录现在的三个示例任务 `001_mul_precedence / 002_eval_division_stub /
003_multidigit_number` **只放了 `task.json`，还没有 `break.patch`**。

原因：`break.patch` 是针对 `fixture/` 里**真实实现**生成的 git 补丁——补丁要能
干净地打进纯净 fixture、并让 `target_tests` 精确变红。而 `fixture/` 的库代码
（`tokenizer.py` / `parser.py` / `evaluator.py`）此刻还是桩（`raise
NotImplementedError`）。

**因此：等里程碑 M0 完成（fixture 实现完、`pytest` 全绿）之后**，再按下面
§3 的流程为每个示例任务补出 `break.patch`。在那之前，`task.json` 已经把每道题的
「意图」（`target_tests` 指向的真实测试）钉死，可以先行 review。

---

## 1. 目录布局与契约（§9.1）

```
tasks/
├── fixture/                     # 纯净基座（§4），只此一份，永不原地修改
│   ├── errors.py  tokenizer.py  parser.py  evaluator.py  conftest.py
│   └── tests/     test_tokenizer.py  test_parser.py  test_evaluator.py  test_integration.py
├── 001_mul_precedence/
│   ├── task.json                # 题目元数据
│   └── break.patch              # 把 fixture 改坏/挖空的 git 补丁（M0 后补）
├── 002_eval_division_stub/{task.json, break.patch}
└── 003_multidigit_number/{task.json, break.patch}
```

**契约**：每个任务是 `tasks/<id>/` 下**恰好两个文件**（`task.json` +
`break.patch`）；**目录名必须等于 `task.json` 的 `id`**；bench 按目录名字典序
遍历，故用 `NNN_slug` 前缀保证稳定顺序（`discover_tasks` 会跳过 `fixture/`）。

---

## 2. `task.json` 字段 schema（§9.2）

```json
{
  "id": "001_mul_precedence",
  "title": "Fix operator precedence: '*' '/' must bind tighter than '+' '-'",
  "kind": "fix_bug",
  "description": "Some tests are failing. Run the test suite, find the failing cases, and fix the library code so all tests pass. Do not modify any test file.",
  "target_tests": ["tests/test_parser.py::test_precedence_mul_over_add"]
}
```

| 字段 | 类型 | 必填 | 行为 / 约束 |
|---|---|:--:|---|
| `id` | `str` | ✅ | 全集唯一；**必须等于任务目录名**；用 `NNN_slug`（snake_case）|
| `title` | `str` | ✅ | 一行英文摘要，进记分卡（可以点明修的是什么，它面向人不面向 agent）|
| `kind` | `str` | ✅ | 枚举，只能是 `"fix_bug"` 或 `"implement_stub"`；驱动记分卡按题型分组 |
| `description` | `str` | ✅ | 交给 agent 的自然语言提示，**用英文**；只描述**症状与目标**（「测试在红，让它们变绿且别弄坏别的」），**不得泄露修法**（不说改哪个文件、哪一行）——这是测试驱动求解的核心 |
| `target_tests` | `list[str]` | ✅ | 非空；每项是相对 fixture 根的 pytest node id，形如 `tests/<file>::<func>` |

**`title` vs `description` 的区别**（常见混淆）：`title` 是给**人**看的记分卡摘要，
可以点明 bug 是什么；`description` 是给 **agent** 的提示，必须只讲「有测试在红、
让它们全绿、别改测试」，**绝不能**暴露改哪个文件哪一行。三个示例任务的
`description` 就是同一句通用提示，正是这个原则的体现。

**`target_tests` 语义（关键契约）**：它是这道题「意图钉死的行为」——一组
**纯粹因本次改坏而红、且纯粹因正确修复而绿**的测试。要求：
- 打补丁后每一项都**必须红**（否则造题失败，见 §4 红闸）；
- harness 判分时「`target_tests` 全绿」是 solved 的**必要条件之一**，另一必要
  条件是回归检测（§5）；
- 级联变红的**其它**测试**不必**列进 `target_tests`（由回归规则兜底）；作者应把
  `target_tests` 选成「因果最干净」的那几个（改坏点的**直接**后果）。

**node id 写法**：`tests/<file>::<func>`，例如
`tests/test_parser.py::test_precedence_mul_over_add`。这与 harness `run_pytest`
从 junitxml 还原出的 key（`f"{file}::{name}"`）**逐字符可比**（参数化用例带
`[param]` 后缀，原样写）。引用的必须是 `fixture/tests/` 里**真实存在**的测试函数。

---

## 3. 两类任务（§9.3）

| 题型 | 补丁做了什么 | 红的形态 | 考察 |
|---|---|---|---|
| **`fix_bug`** | 把某函数**改成语义错误**（交换运算符、抹平优先级、去掉循环、返回错类型…），函数照常返回，只是结果 / 结构 / 异常类型错 | 断言不匹配（`assert 9 == 14`、AST 结构不符、`pytest.raises` 抓不到预期异常）| 读懂红点 → 定位「值对但不对」的那处逻辑 → 精准改回 |
| **`implement_stub`** | 把某函数体挖空成 `raise NotImplementedError(...)`（或 `pass` / `return None`）| 目标行为整个缺失，测试拿到 `NotImplementedError` 或错误返回 | 从**契约 + 测试**反推实现，从零把这段写出来 |

两类都**只动库模块**（`tokenizer.py` / `parser.py` / `evaluator.py` /
`errors.py`），**从不动 `tests/`**——这保证「正确的修复」永远存在于库代码里。
本项目要求 `fix_bug` 与 `implement_stub` **各至少 1 题**（§9.8）。

---

## 4. 生成 `break.patch`：路径契约 + 作者流程（§9.4 / §9.5）

> **前置条件**：`fixture/` 已实现完、`pytest` 在 `tasks/fixture/` 下**全绿**
> （里程碑 M0）。补丁是针对真实实现挖的洞，fixture 还是桩时无法生成。

### 4.1 路径契约（务必照做，否则补丁打不进副本）

补丁里的文件路径必须是**相对 fixture 根的顶层名**（`parser.py`、`tokenizer.py`），
**不是** `tasks/fixture/parser.py`。用 `git diff` 生成时默认前缀是
`a/parser.py` / `b/parser.py`，harness 一律用 `git apply -p1 <绝对 patch 路径>`
（`-p1` 剥掉 `a/` `b/` 前缀）。

> **常见坑**：在**项目根**用 `git diff` 生成，路径会变成
> `a/tasks/fixture/parser.py`，`-p1` 剥不干净、打到副本里找不到文件。**所以改坏
> 动作一定要在「fixture 根的副本」里做**（下面第 1 步）。`git apply` 不要求目标
> 是 git 仓库、直接改工作树文件；且默认**严格**（不容 fuzz）——打不干净说明
> fixture 变了、这题须重造，这正是我们要的可复现性。

### 4.2 作者流程（以「乘除优先级」001 为例）

```bash
# 1. 从纯净 fixture 复制到临时工作区（顶层就是 parser.py 等，切勿原地改）
cp -R tasks/fixture /tmp/brew && cd /tmp/brew
git init -q && git add -A && git commit -qm base     # 只为拿到干净的 git diff 基线

# 2. 编辑目标文件把它改坏——只动一个函数，尽量小，让 target 变红、别的尽量少红
#    001 这里：改 parser.py 的 expr（加减层），把 '*' '/' 也拉进循环，抹平优先级

# 3. 生成补丁（顶层路径 a/parser.py，适配 -p1），落进任务目录
git diff > /Users/.../tasks/001_mul_precedence/break.patch

# 4. 临时工作区可丢弃；把 break.patch 落进任务目录，写好/核对 task.json
```

---

## 5. 单任务生命周期与回归检测（§9.6）

harness 对每道题、每次评测跑一遍下面 6 步（第 1/2/3/5 步是 **harness**、第 4 步
是 **agent**）：

```
1. 复制    make_workspace(fixture_dir, patch_path) → <workdir>   # 隔离副本，绝不碰纯净库
2. 打补丁  （make_workspace 内 git apply -p1 <绝对 patch 路径>）
           └─ 打不干净 → SandboxError → 记「任务损坏」(patch_failed)，中止本题
3. sanity  可选：跑 <target_tests> 确认全【红】= 红闸（造题期已自检，运行期可省）
4. agent   run_agent(workdir, task.description, config)：工具全封闭在 <workdir> 内，
           读文件 / 搜索 / 改文件 / 跑 run_tests，看红绿迭代，直到自认完成或触护栏
5. 复判    ① 用纯净 tasks/fixture/tests/（+ conftest.py）覆盖 <workdir>/tests/ —— 反「改测试作弊」
           ② harness 独立复跑全量 pytest（run_pytest），不看 agent 任何自述
           ③ 按回归规则算 solved / not solved，记步数 / token / 成本
6. 清理    cleanup_workspace(<workdir>)（--keep 时保留供调试）
```

第 5 步的**测试还原**是完整性护栏：agent 在第 4 步可以**读**测试（测试驱动求解
本就要读测试理解期望），但最终判分用**受保护的原版测试**——即便它偷偷改了测试
文件也影响不了判分。

**回归检测规则**：设 `BASE` = 纯净 `tasks/fixture/` 全量 `pytest` 结果 = **全绿**
（不变量）。对一次任务运行，第 5 步复判得结果集 `FINAL`，则：

```
solved  ⟺  (target_tests 里每一项在 FINAL 中都通过)
        且  (BASE 中通过的测试，没有任何一个在 FINAL 中变红)   ← 回归检测
```

- 第二个子句就是「**全量测试不得出现新失败**」。因 `BASE` 全绿，它等价于
  「`FINAL` 必须全绿」——所以复判操作上就是「全量跑一遍，看是不是全绿」。
- 任何「在 `BASE` 绿、在 `FINAL` 红」的测试 = **回归**，即使 `target_tests`
  全绿也判 **not solved**（挡住「把目标改绿却弄坏兄弟测试」的伪解）。

---

## 6. 造题自检流程（造完必做，也是本章验收核心 §9.5 / §9.8）

补出 `break.patch` 后，用一份纯净副本按下面三步验证——**红闸通过 + 存在使全绿
的解**才算这题造好：

```bash
# 从纯净 fixture 复制一份专门用来验证
cp -R tasks/fixture /tmp/verify && cd /tmp/verify

# ① 补丁必须干净成功（无 reject / 无 fuzz）
git apply -p1 /Users/.../tasks/001_mul_precedence/break.patch

# ② target_tests 必须全【红】= 红闸通过（这里以 001 的目标测试为例）
pytest tests/test_parser.py::test_precedence_mul_over_add

# ③ 还原补丁后全量必须【全绿】= 存在使全绿的解，且不靠改测试
git apply -R -p1 /Users/.../tasks/001_mul_precedence/break.patch && pytest
```

自检清单（对照 §9.8 验收标准）：

- [ ] 任务目录**恰含** `task.json` + `break.patch`，且 `id` == 目录名。
- [ ] `task.json` 过 schema：五字段齐备；`kind ∈ {fix_bug, implement_stub}`；
      `target_tests` 非空且每项形如 `tests/<file>::<func>`、引用真实测试。
- [ ] `git apply -p1 break.patch` **干净成功**（无 reject / 无 fuzz）。
- [ ] 打补丁后 `pytest <target_tests>` **全红**（红闸）；`git apply -R` 还原后
      全量 `pytest` **全绿**（证明存在使全绿的解、且不靠改测试）。
- [ ] `fix_bug` 与 `implement_stub` **各至少 1 题**。
- [ ] 级联红点集合与设计描述一致（可与 §9.7 的「级联红」清单对照）。
- [ ] 全流程零污染：`tasks/fixture/`（排除 `__pycache__`）在任意次评测后逐字节
      不变（所有动作都在副本 / 系统临时目录里）。

---

## 7. 三个示例任务速览（§9.7）

三题分别命中 fixture 里点名要抽查的三处，且刻意展示**层间级联的不同幅度**。
（补 `break.patch` 时按下面的「改坏点」下刀。）

| id | 题型 | 目标文件 | 改坏点 | target_tests | 级联红（回归兜底，不列 target）|
|---|---|---|---|---|---|
| `001_mul_precedence` | `fix_bug` | `parser.py` | `expr`（加减层）把 `*` `/` 也拉进循环，抹平优先级，`1 + 2 * 3` 被算成 `(1+2)*3=9` | `test_parser.py::test_precedence_mul_over_add` | `test_precedence_div_over_sub`、`test_integration.py::test_precedence`、`::test_deep_nesting`；tokenizer/evaluator 全绿 |
| `002_eval_division_stub` | `implement_stub` | `evaluator.py` | `eval_ast` 的 `op == "/"` 分支挖空成 `raise NotImplementedError`，其余保留 | `test_evaluator.py::test_eval_division_is_float`、`::test_eval_division_fractional`、`::test_eval_division_by_zero_raises` | `test_integration.py::test_float_result`、`::test_deep_nesting`、`::test_div_zero_propagates`；**tokenizer、parser 全绿**（evaluator 层完美隔离）|
| `003_multidigit_number` | `fix_bug` | `tokenizer.py` | 「吃数字」子过程去掉聚合循环、只取当前一位，`"123"` 被拆成 `1 / 2 / 3` | `test_tokenizer.py::test_multi_digit_number` | `::test_single_integer`；并向上级联到 `test_parser.py::test_left_assoc_sub`、`::test_precedence_div_over_sub`（含多位字面量 → 残留 token → ParseError）——**tokenizer 层 bug 会跨层级联**，红点集更大 |

`002`（evaluator 隔离，红点最小）与 `003`（tokenizer 级联，红点最大）刻意作对照，
演练回归规则「全绿才算过」。
