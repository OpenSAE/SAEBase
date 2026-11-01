# Work Plan 编写标准 · 仓库级默认规范

本标准适用于本文件加入后新建或实质修改的 Work Plan。历史冻结计划、试卷及证据保持原文，不以格式迁移改变其可信性。

## 0. 先判明目标、所有权和最小实体

每个 Work Plan 开头必须写明用户要得到的能力、它是**通用能力**还是**特定 Dataset/Model/Task/Experiment 实例**、输入输出、成功证据和不能执行的范围。先定位现有代码和配置。逐个新增或移动路径说明所有者、职责、已有候选复用接口、增加理由、依赖方向和删除/迁移的旧位置。不得为某个实验增加一级源码包、第二套 Registry、配置层、数据抽象或 Runtime。

路径所有权遵循 [Core 架构纲领](../../core/BOUNDARIES.md)；特定研究方案主要由 `conf/experiment/<study>/` 和 `scripts/experiments/<study>/` 组织，通用 API 则归相应 Dataset、Model、Task 或 Runtime。测试置于 `tests/workplan/<id>/` 或实验对应的 Dataset/Model/Task/Sub-name 目录。Scripts 也优先按 `scripts/workplan/<id>.sh` 或 Dataset/Model/Task/Sub-name 分类。布局有据可扩展，不能随意创建根目录。

## 1. Agent 操作入口必须能从干净环境复现

写明仓库 URL、精确目标分支及基线 SHA、工作目录、必要操作系统、Python 版本、可用空间和 CPU/GPU、许可证与认证要求。**不能假定 origin 跟踪所有远程分支。** 单分支 fetch refspec 时明确使用：
```bash
git fetch origin refs/heads/<branch>:refs/remotes/origin/<branch>
git switch --track -c <branch> origin/<branch>   # 仅当本地尚无此分支
git rev-parse HEAD
```
已有同名本地分支则 `git switch <branch>` 并 `git pull --ff-only origin <branch>`。不要重写用户的 fetch refspec 或清理其本地工作树。

## 2. 依赖解决与真实安装是第一道验收

列出用于启动的包、Python/torch/CUDA/driver 条件以及特殊依赖版本；必须在一致环境实测 resolver、`pip check`、关键 import 和代表性功能 smoke。验证每一组 optional extras 与核心库兼容，尤其是传递依赖中的同名版本固定，不能只看依赖顶层名称。**未运行的环境不得宣称 ready**；不能让 Agent 用修改科学指标、降级标准或换模型来绕过依赖冲突。用户报告失败后先保留真实命令、退出码、原始日志、冲突链、已执行/未执行阶段，再由维护者修正源码或隔离不兼容的非核心依赖。

运行生成的 `*.egg-info/`、`build/`、`dist/`、虚拟环境、cache、下载权重和实验结果都是工件，不是仓库顶层源码模块。尽量把临时环境与产物放在 Runtime 所属 `experiment/` 或仓库外受控路径，避免混入提交。任何清理都限定本次生成的、明确可辨识的文件，保留已有用户文件。

## 3. 分阶段执行，尽量缩小首轮测试规模

按照 Source/Branch → Environment → Dependency → Import/Contract → Frozen configuration → Minimal CPU negative tests → 下载/权限检查 → 小样本真实 Pilot → 授权的 Full 执行 → Audit 分阶段。新方法的第一轮默认最小样本和单任务；大规模操作需有独立明确的预算、停止条件、价值依据和用户授权。每阶段写准确入口命令、成功标准、退出状态、独立收据位置和下一阶段的门槛。

下载与执行权限必须分开：GPU 推理、原生训练、受限第三方数据、外部发布分别授权；安装成功不代表权重、SAE hook 或科学结果真实通过。资源不足返回 BLOCKED，不擅自缩小冻结数据或替换权重。

## 4. 结构与证据独立验收

列出每个新增文件的实际源码所有者，维护封闭的 Contract ID 绑定并检查永远启用的合同。验证 Runtime 产物、模型和数据身份、可重复性、恢复、错误证据以及发布门槛。维护精确的 PASS / FAIL / BLOCKED / NOT_RUN 状态及 CI commit SHA；区分代码存在、离线测试、真实模型测量、独立审阅、正式发布。Agent 自测通过不能取代独立验收。

## 5. 每个新 Work Plan 的必备章节

1. 目标与科学/工程边界
2. 现有模块清单及各文件所有权表（新增、复用、迁移、删除）
3. 数据/模型/任务/配置与依赖方向
4. 环境矩阵、兼容依赖和干净机器安装检查
5. 精确 Git fetch、切换、预检、pilot、正式运行命令
6. 资源预算、最小试验、失败与恢复策略
7. 通用 Contract 绑定、自动检查、独立验收与可追踪收据
8. 最终交付路径、工作树卫生、权限和发布限制

发现架构冲突、依赖冲突或缺少执行前提时，Work Plan 应返回修改方案并停止执行，不能把错误交给不知上下文的 Agent 临场猜测解决。

## 6. 新 Work Plan 的控制源与 Atlas 双向审阅

正式起草前与用户讨论目标、科学价值、可复用成果、预期结果表、计算及 B2 交付边界，将已确认内容与未决问题分开记录在 [from_human/](from_human/README.md)。没有确认的信息必须保持 unresolved，不得猜测。起草时先读取 Atlas Core/Knowledge/Context 的真实节点 ID，复用已存在的目标和术语；新节点需独立审阅。

采用 [WORKPLAN_TEMPLATE.md](WORKPLAN_TEMPLATE.md) 中的章节及 `workplan-json` 引用块，在同一份 WP 中关联 `from_human`、Atlas Core/Knowledge/Context、真实 Code 路径、具体 Experiment 配置、具有依赖关系的任务及预定义结果表。具体科研参数仅在 `conf/experiment/<study>/`，不可在计划、Prompt 和脚本中创建数套彼此竞争的数值权威。功能性注释描述 Code 实际职责与对应 WP/Knowledge，不在通用 API 内硬编码单一 WP。

起草后运行 `python -m harness.github.work_plan.lifecycle validate --plan <repo-relative-plan.md>` 和已有 `python -m harness.contracts.registry check --plan <plan>`，核对真实引用、上下游职责、执行 DAG、必要交付表、双语 Context 与 Atlas 来源。起草前后均需审阅关联，不能通过新增抽象节点满足检查。

## 7. Setup、Audit 与已填写 Work Plan

每个实际研究阶段明确 Setup 的样本、环境、网络和 GPU 授权；先测量真实执行速度，再按用户在 WP 确定的预算与阈值选择执行路径。代表性 Pilot 只是已执行组合的证据，不能替代全部模型和来源。计划外失败经通用 Audit 归因并分 local/global/human-decision，科研指标本身仍由所属 Task 的真实实现检查。失败记录与已完成结果不可覆盖。

[DELIVERABLE_STANDARD.md](DELIVERABLE_STANDARD.md) 是每份 WP 的交付规范。主要结果是一份**完整填写的原始 Work Plan 副本**和预先设计的表格，附真实运行、资源、时间、日志、文件哈希、证据和备注；自由形式分析是次要附件。冻结原始文件只读，已填写文档存 `experiment/` 的单次运行目录。B2 是推荐持久化目的地，必须有授权范围、实际校验及脱敏记录；人类服务器上已有凭据由配置读取，不写入仓库。规范检查、执行授权、科学验收和公开发布依然是不同动作。



## 8. 版本命名与历史归档的长期约束

新版本的具体 Work Plan 使用 `<semver>_wpNNN_<word>_<word>.md`
（两个简短英文单词、每个版本独立三位序号）；`WP22` 等原计划 ID
仍保存在计划内容和独立 Index 中，不以归档顺序改变身份。
Release Preparation 前按 [ARCHIVE_STANDARD.md](../ARCHIVE_STANDARD.md)
生成 Human Check 关联审阅方案；仅对审核通过的独立范围进行迁移。
过去已冻结的 WP/考试记录永久豁免未来新增的格式和内容校验。
