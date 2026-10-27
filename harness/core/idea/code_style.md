# 思考草稿：面向 Context 的代码组织（尚未实施）

- 对复杂组件优先考虑有清晰继承关系的 Class，而不是堆积无所有者的自由 Function。但 Python 函数型写法也有合理场景，不能一概重构。
- 注释与 Docstring 适当链接到 Knowledge ID（如 `context_id: g.plugin.contract`），而不复制整个科研背景。
- 若单个类职责过多，先评估 Composition、Protocol 与职责拆分，不盲目制造深层继承。
- 这些只是规划想法，尚未批准进入必须通过的 Rules 或 CI。
