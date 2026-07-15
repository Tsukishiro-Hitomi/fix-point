# 有意留最小：仓库根 pytest.ini 的 `pythonpath = .` 已让 `from agent.xxx import ...`
# 成立，无需在此改 sys.path。共享 fixture（如假 LLM）后续按需加到这里。
