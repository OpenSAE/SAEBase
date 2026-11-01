# GitHub 工程协作总览

本目录是 Harness 的 GitHub 工程协作资源，独立于 GitHub 原生配置 `.github/`。

| 子目录 | 职责 |
| --- | --- |
| [work_plan/](work_plan/README.md) | Work Plan、任务审批与历史记录 |
| [work_test/](work_test/) | 独立 Harness 测试 |
| [development_doc/](development_doc/README.md) | 开发、API 和发布文档 |
| [assets/](assets/) | 图示与品牌资源 |

## 对话式任务规划

先使用 [SAEBase Delivery Skill](../../.github/skills/saebase-delivery/SKILL.md)
理解目标、判断是否需要源码层面的 Precheck、确定现有模块归属、建立最小任务依赖图。
详细规范见 [任务规划与协作原则](development_doc/agent-delivery.md)
和 [English](development_doc/agent-delivery.en.md)。
此阶段由对话式 GPT 完成，不要求本地 Agent、GPU 或额外的预检脚本。
需要正式实现时继续遵守既有 Work Plan、Contracts 和权限约束。

## Agent 的修改准则

先阅读 [Harness 总览](../README.md)、[仓库架构纲领](../core/BOUNDARIES.md) 和 [通用 Task Contracts](../contracts/README.md)，再按 Work Plan 选择必需文件。先搜已有实现及注册接口，区分通用能力和某个具体 Experiment。

新增脚本和产品测试应归入 `scripts/` 与 `tests/` 一级目录，优先按 `workplan/<id>` 或 `<dataset>/<model>/<task>/<sub_name>` 分类；具体研究也可用 `scripts/experiments/<study>/`、`tests/experiments/<study>/` 组织。这是开放规范，特殊布局需说明归属和复用原因。

PR 必须列出新增与移动文件、原有可复用接口、依赖方向、配置与注册影响、对应测试、证据及未完成验证。新一级模块或改变模块职责必须独立架构审批，获准编辑路径不等于获准新架构。

## Work Plan 规范

所有新工程计划遵循 [Work Plan 编写标准](work_plan/WRITING_STANDARD.md)，包含先验依赖冲突检查、具体文件所有权、可执行命令以及分阶段验收。


## 按版本归档与 Human Check

永久文件命名及历史迁移规则见 [ARCHIVE_STANDARD.md](ARCHIVE_STANDARD.md)，
具体冻结路径、版本和 Git Blob 身份见 [archive_index.json](archive_index.json)。
Development Doc、Work Plan、Work Test 的历史记录按各自所有权保存在
`archive/<version>/`；持续有效的管理规则和当前 CI 测试保持稳定入口。

下一版本准备阶段，先运行 [Archive Preparation](../../.github/workflows/archive-preparation.yml)
产生只读候选，再在 Atlas Human Check 的关联节点中逐项审阅。只有
获批范围才可迁移；归档后经过 [Release Validation](../../.github/workflows/release-validation.yml)
的 SHA 固定验收。
