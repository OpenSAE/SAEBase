# WP18 Harness failure attribution · Governance baseline

本文件仅用于 Harness 的持续改进。历史 Agent 原分支 `wp18/h01-h06-20261029` 及独立验收 PR #81 保持原样，**不要求本地 Agent 修改治理实现**。本次治理改动的准则由通用 Contract Registry 预先定义，并由 Work Plan 明确选用。

| 发现 | 旧版 Harness 的可能缺口 | 优先改动 | 复核方式 |
| --- | --- | --- | --- |
| H03 来源路径虽符合 `path:line` 却不具备安全/真实性 | Evidence 只描述“可定位”，没有明确区分语法、边界、存在、版本和 Authority | `CT-EVIDENCE` 预置多层来源要求；`reference_location` 作为统一只读参考验证器 | 新任务中使用不同类型的来源结构，比较 Agent 选择的合同、错误拒绝与来源记录 |
| H04 合成结构中的嵌套敏感字段被遗漏 | 缺少完整输入结构/输出序列化边界约束 | `CT-PRIVACY` + `CT-INTERFACE`；结构化递归脱敏/不支持类型 fail-closed | 新的嵌套审阅报告任务；验证日志与产物，而不是只改某个 `device.token` |
| H05 Parent scope 接受不存在的节点 | Tree/Review 的全局身份不变量没有统一要求先验证筛选目标 | `CT-STRUCT` + `CT-REVIEW`；通用 Tree DFS 不变量 | 使用新图形拓扑和新的范围，确认未知/跨树/环引用均拒绝 |
| H06 英文完整却缺失中文未报错 | 全局双语规则没有可靠进入局部“英文质量”检查任务 | `CT-LOCALE`；双语 required 字段与符号不可变约束 | 新的双语文档类型（非 WP18 原 Overlay），同时检查 zh/en 缺失与源码身份 |

## 归因纪律

以上是由提交及 CI 可观察到的 Harness **设计缺口假设**，不是对 Agent 内部推理的推测，也不证明旧 Agent 违反了它未曾收到的合同。下一轮每个 Run 必须记录已选的 Contract IDs、当次加载的 Context 路径、修改文件、实际测试命令与退出码；独立验收者依据证据区分：

- `CONTRACT_GAP`：通用目录中根本没有相关职责。
- `SELECTION_GAP`：已有职责未在 Work Plan 中选用，或执行侧未能实际获得。
- `IMPLEMENTATION_ERROR`：职责已正确提供而实现仍遗漏。
- `ENFORCEMENT_GAP`：可判定的违规结果未被自动检查拦截。
- `EVALUATOR_SCOPE_GAP`：新增要求并非实施前已冻结的合同。
- `ENV_BLOCKED`：环境、权限或工具导致无法执行。

不应把全部失败推给 Agent，也不应把任意独立测试事后扩展的要求直接算作 Agent 能力不足。后续 Harness 改动由可信维护者负责，测试 Agent 继续遵守当前规范、提供真实产物并暴露失败；通过历史案例不能代替对新任务泛化的核查。

**约束**：通用 Contracts 的版本更新必须独立评审。局部 Work Plan 不得偷偷重定义合同、取消 Always-on 条款或把不通过变成 PASS。权限/审批/科研权威仍由既有独立治理机制控制。
