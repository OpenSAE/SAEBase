# H03 · WP18 独立验收

**技术结论：FAIL — 来源路径安全**

- 原始 Agent 分支：`wp18/h01-h06-20261029`，提交 `1d8116f2d8d28177242699ddd7dcc082ecd6d226`。
- 独立技术复验：[CI 40810023608](https://github.com/OpenSAE/SAEBase/actions/runs/40810023608)；**26 passed、4 failed**。六套 Agent 自测分别独立重跑，合计 30 passed。
- 六套源码 SHA、候选节点行号与签名、预期 AST 节点/边由独立构建检查通过。
- 本文件是技术核验，不是人类节点 Approve 或最终科学验收；原始 Agent 分支未修改。

独立反例在节点与术语中同时设置 `source_evidence: ../secret.py:1`。函数返回 `diagnostics: []`，错误接受父目录跳转路径为可定位来源。应拒绝 `..`、绝对路径、非法空段，并保持对正常相对路径和行号的接受。独立回归测试 `test_h03_rejects_traversal_as_source_evidence`。

下一步：要求 Agent 只在本题的 \`harness/tests/wp18/runs/20261029-wp18-01/H03/\` 授权测试目录修复源码和自测；禁止改独立验收用例。重新提交后复测。
