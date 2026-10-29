# H04 · WP18 独立验收

**技术结论：FAIL — 嵌套敏感字段泄漏**

- 原始 Agent 分支：`wp18/h01-h06-20261029`，提交 `1d8116f2d8d28177242699ddd7dcc082ecd6d226`。
- 独立技术复验：[CI 40810023608](https://github.com/OpenSAE/SAEBase/actions/runs/40810023608)；**26 passed、4 failed**。六套 Agent 自测分别独立重跑，合计 30 passed。
- 六套源码 SHA、候选节点行号与签名、预期 AST 节点/边由独立构建检查通过。
- 本文件是技术核验，不是人类节点 Approve 或最终科学验收；原始 Agent 分支未修改。

独立反例输入 `{"synthetic": true, "device": {"token": "SECRET-NESTED-123", "home": "/private/home"}, "gpu": null}`，输出的 `fields.device` 原样包含模拟 token 和绝对私有路径。必须递归审查 dict/list、敏感字段名与路径，且不丢 synthetic/unknown 语义。此处是**合成哨兵字符串**，未出现真实用户凭据。独立回归测试 `test_h04_redacts_secrets_nested_in_synthetic_records`。

下一步：要求 Agent 只在本题的 \`harness/tests/wp18/runs/20261029-wp18-01/H04/\` 授权测试目录修复源码和自测；禁止改独立验收用例。重新提交后复测。
