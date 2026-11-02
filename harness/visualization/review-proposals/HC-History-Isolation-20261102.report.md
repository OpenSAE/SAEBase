# HC-History-Isolation-20261102 · 历史资料版本归档与零依赖治理（Human Check 草案）

- 基线：`d3e495406d32aa15960da615bbfacf2047e67559`；项目：`OpenSAE/SAEBase`。
- 已确认正式软件 Release：v0.2.0, v0.3.0, v0.4.0, v0.5.0, v0.6.0, v0.6.5。
- 检索范围：249 个尚未在 archive/ 的文件（Development Doc 75、Work Plan 71、Work Test 103）。
- 初步分类：PERSISTENT_OR_UNRESOLVED 66、HISTORY_CANDIDATE 79、CURRENT_VERSIONED_PLAN 1、TEST_REQUIRES_LIVE_DEPENDENCY_REVIEW 103；这些数字并不代表已获准迁移。
- 现有 archive_index：10 条；9 条仍有原始兼容副本。

## 拟由人类独立审批的源码/迁移范围
### SC01 Development Doc 历史文件及长期文档隔离

文件数量：75；具体逐文件清单见 `HC-History-Isolation-20261102.manifest.json` 和 `HC-History-Isolation-20261102.inventory.json`。约束：明确文档版本和最终内容；历史报告与当前指南不再相互依赖；保留 RELEASES 公开入口。

### SC02 Work Plan 冻结与历史引用断开

文件数量：71；具体逐文件清单见 `HC-History-Isolation-20261102.manifest.json` 和 `HC-History-Isolation-20261102.inventory.json`。约束：冻结 01–21 原文/表格；当前 WP23 保持独立；现役 Agent 文档不得继承旧执行指令。

### SC03 退出旧测试前的覆盖率与 CI 审查

文件数量：103；具体逐文件清单见 `HC-History-Isolation-20261102.manifest.json` 和 `HC-History-Isolation-20261102.inventory.json`。约束：仅归档已退役版本专属测试；替换仍活跃的通用回归覆盖；避免 pytest 默认收集已归档测试。

### SC04 通用构建、发布和历史验证器解耦

文件数量：9；具体逐文件清单见 `HC-History-Isolation-20261102.manifest.json` 和 `HC-History-Isolation-20261102.inventory.json`。约束：打包不用旧版文件作必需输入；新 release 不读取历史 Work Plan；历史考试用固定 Tag 独立复现；保留当前 API, CI 功能。

### SC05 归档完整性、Release 版本映射与零依赖验证

文件数量：5；具体逐文件清单见 `HC-History-Isolation-20261102.manifest.json` 和 `HC-History-Isolation-20261102.inventory.json`。约束：Release Tag 及 Blob SHA 作为来源证据；拒绝猜测不明确归属的版本；根目录反向扫描未登记档案；移除所有 archive 目录后产品检查仍可通过。

## Atlas Node 影响

五项全部是 **SOURCE REVIEW**：分别挂靠五个真实 Core 节点用于 Atlas 导航。审批只针对具体路径、风险及验收条件，不改变节点的标题、说明、性质、状态、证据或关系。无需虚构 Core/Code 节点变化。

在公开 Pages 上的 Approve / Deny / Unchecked 仅是浏览器草稿，使用 Send report 显式提交才可交给维护者核验。此提案的发布不等于用户已批准任何实际迁移。

## 不变约束
1. 历史原文的 Git Blob SHA 不变；Release Tag 仍是来源证据，不依据 WP 编号猜测版本。
2. 现役软件构建、发布、普通 CI 和 Agent 执行不得依赖 archive 中旧文档；归档完整性自检独立运行。
3. 尚在运行的测试须有等效的活跃回归后才能退役；不得以旧命名作为迁移依据。
4. v0.6.5 之后的新 WP23 不能被误归入已完成版本。
5. 当前状态为 **PENDING HUMAN CHECK**；不代表已批准 PR、源码或发布。

## 验收
- Repo 所有受管理文档和测试精确映射到 KEEP / ARCHIVE / UNRESOLVED，且版本归属有证据。
- 从临时构建树移除 archive 后，独立的现役 CI、构建、发布候选校验照常通过。
- Repo README、长期指南及非历史治理源码内没有旧档案硬依赖、旧计划作为权威任务来源的引用。
- Human Check 逐项批准，必要的 source-only 范围再次进行精确文件 Diff 审批；未批不得改动。
