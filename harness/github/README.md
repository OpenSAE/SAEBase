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

## Harness / GitHub 通用错误的防复发机制（强制）

一旦发现 Harness、GitHub CI、Agent 工作流或相关执行过程中的错误，不管眼前是否容易修复，**必须判断是否可能在其他 PR、Workplan、机器、任务或 Agent 会话中复现**，并提出至少一种适用的通用预防方案。不能仅修复当前报错就声称问题已持久解决。

1. **保留证据与归因：**记录源码 SHA、命令、日志、失败输入、实际负责模块和权限边界；区分仅当前输入异常、共享规范缺失、通用代码缺陷、缺少测试或执行门禁、尚不能确定的情形。
2. **提出通用方案：**明确根因、潜在影响面、最小可复用的校验/配置/工具/流程改动、目标文件、兼容风险、负面回归测试及回滚办法。
3. **较确信时直接防复发：**若根因和修复机制已有可信证据、改动属于已授权的文件与职责范围，应同步解决局部错误和通用缺陷，补充能重现原失败的负面测试，并至少验证一个不同的适用情形。给人类报告两类修复与相应证据。
4. **不确信或越权时请示：**当根因不明、可能改变共有接口/行为/审批、或触及未获批准的文件和资源时，先向人类说明候选方案、利弊、影响范围、验证办法，取得具体授权后再实施；不得通过技术上“很确定”绕过 Human Check、保护目录或科研参数冻结。
5. **明确完成标准：**指出通用防线在哪个权威文件或自动测试生效、如何证明其他场景不会重复同类问题；验证不足时明确写明“防复发尚未验证”，不把当前 CI 变绿等同于永久解决。

此规则是 [AGENTS.md](../../AGENTS.md) 的仓库总控约束，适用于代码、测试、工具、CI、Harness 与 Agent 流程。遇到人类明确指出流程违反规范时，仍需优先提出流程规范化方案，再修复具体问题。

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

## Local Agent 的 Python 格式自动修复

首次使用仅需安装轻量工具和本地钩子（不加载模型/GPU）：

```bash
python -m pip install "ruff==0.15.22" "pre-commit>=4,<5"
pre-commit install
```

本地 Agent 在**只暂存已授权变更**后运行 `python saebase/tools/check_new_python_quality.py --fix`，再核对差异并重新暂存；正常 `git commit` 也会自动格式化已暂存的新增 Python 文件（文件被修改时，Git hook 会要求重新暂存并再提交一次）。既有大文件只有经明确允许才运行 `--include-modified`，避免全文件无关的格式化差异。CI 继续用同一 Ruff 版本只读检查。

**GitHub API 写入不经过本地钩子。** 线上协作 Agent 必须先在受控环境用同版 Ruff 格式化将要上传的 Python 文本，再调用 GitHub 文件写入工具，不能依赖远程 CI 自动代为修改仓库。若工具不可用，则明示阻塞，不允许跳过格式检查或禁用质量门禁。
