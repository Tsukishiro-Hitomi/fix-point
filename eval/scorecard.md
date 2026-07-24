# fixpoint scorecard

- date: `2026-07-24T14:06:47`  ·  commit: `e2f92f6`
- model: `anthropic/claude-opus-4.8`  ·  retrieval: `False`  ·  self-correction: `False`
- guardrails: max_steps=`30`, cost_budget=`$0.5`, run_tests_timeout=`60s`, judge_timeout=`60s`

## Per-task (baseline)

| task | solved | steps | tokens | cost($) | wall(s) | stop_reason | regressions |
|---|:--:|--:|--:|--:|--:|---|---|
| 001_mul_precedence | ✅ | 6 | 22533 | 0.1247 | 25.9 | model_stop | - |
| 002_eval_division_stub | ✅ | 8 | 31119 | 0.1690 | 29.2 | model_stop | - |
| 003_multidigit_number | ✅ | 7 | 32223 | 0.1722 | 22.6 | model_stop | - |
| 004_unary_minus | ✅ | 6 | 22689 | 0.1211 | 17.0 | model_stop | - |
| 005_eval_negation_stub | ✅ | 6 | 16477 | 0.0912 | 20.0 | model_stop | - |
| 006_eval_subtraction | ✅ | 6 | 20239 | 0.1084 | 17.0 | model_stop | - |
| 007_eval_multiplication | ✅ | 6 | 20944 | 0.1122 | 20.5 | model_stop | - |
| 008_tokenize_float | ✅ | 7 | 19959 | 0.1093 | 20.9 | model_stop | - |
| 009_bare_dot | ✅ | 6 | 20783 | 0.1151 | 23.9 | model_stop | - |
| 010_tokenize_ops | ✅ | 6 | 15592 | 0.0867 | 24.5 | model_stop | - |
| 011_parser_trailing | ✅ | 6 | 21177 | 0.1150 | 29.5 | model_stop | - |
| 012_eval_addition | ✅ | 6 | 21970 | 0.1172 | 18.9 | model_stop | - |

## Summary

- **pass@1 = 12/12 = 100%**
- avg steps: 6.3  ·  avg tokens: 22142  ·  avg cost: $0.1202  ·  total cost: $1.44  ·  avg wall: 22.5s

## Ablations

| variant | model | retrieval | self-corr | pass@1 | avg steps | avg cost($) | total($) |
|---|---|:--:|:--:|:--:|--:|--:|--:|
| baseline | claude-opus-4.8 | False | False | 100% | 6.3 | 0.1202 | 1.44 |
| haiku | claude-haiku-4.5 | False | False | 100% | 6.8 | 0.0360 | 0.43 |

> 小任务集 + 采样随机性下，条件间的小差异可能是噪声；n_attempts=1。
