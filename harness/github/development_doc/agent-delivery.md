# SAEBase 对话式任务规划与 GitHub 协作原则

本说明供与用户对话的 GPT 或其他规划 Agent 使用。主要入口为
[SAEBase Delivery Skill](../../../.github/skills/saebase-delivery/SKILL.md)。
它主要负责**理解目标、做源代码层面的预检、拆分任务和设计交付顺序**，
不要求安装 Cursor Skill、本地 Agent、GPU 环境或额外的 Precheck 脚本。

既有 `harness/core/BOUNDARIES.md`、[Work Plan 写作规范](../work_plan/WRITING_STANDARD.md)、
Universal Contracts、Work Order v3、Agent Flow 和所有独立审批机制继续有效。
这套规划指南不是新的治理或执行框架。

## 一、GPT 如何规划

首先定义最终的可观察成果，区分通用能力、具体 Dataset/Model/Task、
具体 Experiment 与治理/发布变更。然后只阅读所需的现有代码、配置、
文档和历史 Work Plan，优先复用已有实现。对新增、移动或删除路径，
明确 owner、作用、依赖方向、复用选择和架构审批要求。

规划阶段的 Precheck 是 **desk precheck**，仅依赖当前对话、
已取得的源码与已知证据；它不是在用户机器上执行命令：

| 任务 | 优先检查 |
| --- | --- |
| 文档和 Atlas | 源与生成文件的关系、链接、打包清单、双语和发布入口 |
| 一般 Python 修改 | 模块归属、调用者、Registry/配置、兼容性、已有测试 |
| Dataset / Model / Task | 已有接口、数据与模型身份、任务输入输出、配置传递 |
| 特定 Experiment | 通用能力与实例边界、协议、资源预算、最小 Pilot 设计 |
| Harness / CI / 发布 | 冻结规则、权限、准确 SHA、独立验收与回退 |

查过的事实记为 `VERIFIED_FROM_SOURCE`，合理推断记为 `INFERRED`，
缺失信息记为 `OPEN`，只有真实执行才能确认的事项记为
`REQUIRES_EXECUTION`。源码检查不能冒充安装、测试、真实模型验证或科学结论。

## 二、任务拆分

一个任务至少回答：想交付什么、归谁所有、哪些文件可能改动、
依赖哪个任务、怎样验收、谁有权限执行。

建议先建立最小有向无环依赖图（DAG）：

- 只有文件、职责和审批范围独立时才并行。
- 如果只是开发顺序不同，优先同一个 PR 内分阶段实现。
- 需要独立审批、独立证据或避免同一文件竞争时才拆多个 PR。
- 多 PR 必须标注 base、前置 PR、集成人和最终合并顺序。
- 先完成便宜的结构与依赖判断，再投入大型实现或真实计算。

具体实验应由 `conf/experiment/<study>/` 和
`scripts/experiments/<study>/` 组织；通用 Dataset、Model、Task、
Runtime 的职责保持独立。不要为特定实验提升出新的一级源码目录、
第二套 Registry 或额外的治理系统。

## 三、计划与执行的边界

默认只在**对话中给出计划**，用户要求后才创建或修改 GitHub 文件。
普通规划不必创建 Work Order，也不依赖本地 Agent。执行者可以由用户选择为
GitHub Agent、人工开发者或既有本地 Agent。

需要实际实现时，计划应列出阶段性验收、最小测试、需要的权限、
完成后提交哪些证据。只有在真实环境运行后，才记录命令、退出码、
数据和模型身份、真实硬件及 PASS/FAIL/BLOCKED/NOT_RUN。
涉及受保护治理变更、真实 GPU 计算、远程发布和合并时仍按已有权限规则处理。

## 四、推荐交付格式

1. **最终目标与最小设计**：为什么已有组件足够，哪些新实体确有必要。
2. **Desk Precheck**：源码证据、推断与尚待确认的问题。
3. **任务依赖表**：任务 ID、交付物、文件 owner、前置、并行可能性、验收。
4. **GitHub 支撑工作**：规划文档、配置、测试和 PR 依赖；只做必要改动。
5. **给执行者的简短指令**：准确分支、范围、任务顺序和停止条件。
6. **完成与发布条件**：独立验证、准确 SHA、人工授权、真正需要的实验。

标准是**减少反复返工和不必要的 PR，而不降低研发和科学验收质量**。
