# SAEBase 真实工作流执行协议

版本 `1.0`，本地准备日期 `2026-10-30`。本协议说明真实仓库开发与小规模实验怎样执行、怎样保留证据、怎样独立验收。它复用 [catalog.json](catalog.json) 中的 22 个通用 Contract，目录版本仍为 `1.1.0`。每份 Work Plan 预先选择适用 ID，再冻结题目目标、写入范围、输入、预算与验收条件。

本地交付包表示规范和题目已经准备。二十个考生的实际执行、自测结果及独立验收分别记录，均不能从交付包存在推断。

## 1. 身份、权限与工作树

适用 `CT-BASE`、`CT-SCOPE`、`CT-REVIEW`、`CT-TRACE`。

- 一名 agent 对应一份独立工作树。WP21 的二十个 agent 从固定源码基线及各题声明的初始化材料出发，不共享可写代码、实验目录或 Registry 进程状态。
- 题目中冻结的 `allowed_paths` 是最大实现范围。用户明确指派该题时，其真实指令提供该范围内的本地执行依据。读取题目、工作单标为 ready、通过预检均不会产生人工批准，也不允许考生填写代表人类的审核记录。
- 当前用户仅要求准备规范和题目时，执行者完成本地交付准备；真正考试在具体题目被分派后开始。部署、push、PR、release、公开数据上传与权威 Atlas 状态变更各自保持原有权限边界。
- 考生只改自己题目指定的生产代码、配置、回归测试和运行记录。治理目录、冻结题目、裁判代码、历史成绩、`.cursor/`、`.github/` 和 `saebase/runtime/` 的治理语义保持只读；题目确需改变受保护实现时，由治理维护者另立明确范围。
- 本地包的 `.cursor/hooks/wp21_scope.py` 由可信 dispatcher 拥有的 assignment 记录激活，只承认已经存在的实际用户指派，并核对冻结题目、Contract 与协议哈希。仅准备时状态为 `prepared_not_assigned`。Hook 对各主题 acceptance 目录的保护仍然生效。
- Hook 拒绝已指派的合法操作时，保留原命令及拒绝原因，归为执行环境或治理兼容问题。考生不得关闭 Hook、伪造 Agent Flow phase 或改裁判来继续。
- reviewer 使用工作树外的独立目录，自己执行候选代码并保存验收记录。文件 Hook 只提供工具层检查；低权限进程、只读挂载和资源上限由运行环境落实。

## 2. 什么算真实执行

适用 `CT-TRUTH`、`CT-ENV`、`CT-EVIDENCE`、`CT-INTERFACE`。

每题至少执行一个由 SAEBase 当前源码提供、与题目目标直接相关的可观察行为。Task 题通过配置、Registry、Runner 和真实 Task 实现；公开 API 题调用指定生产 API；文件管理题实际操作由 SAEBase 生成的实验产物。实现位置遵守现有模块归属，不在 Harness 中复制一套模型、算法或任务系统代替被测代码。

`task.execute=false`、`configured_only`、Hydra 配置打印、Registry 中出现一个名字只证明配置或注册阶段。模型加载、捕获、编码、干预、训练、评测必须分别提供实际输出证据。`tasks/sae_runtime.py` 返回的 `configured_only` 不能计为实验完成。

小型合成输入、确定性小模型和本地测试 checkpoint 可以用于真实软件接口执行，但必须标为 synthetic。题目要求已训练模型时，需要加载规定的真实权重，并记录模型和 SAE 的来源、版本、实际加载内容哈希、层及 hook。名称相同不等于 checkpoint 相同。缺失权重、许可、依赖或设备记 `ENV_BLOCKED`；替换成 fake model 不能满足原真实权重条件。

独立验收检查 Python 实际导入路径、源码身份和核心调用。覆盖生产模块的本地同名文件、monkeypatch 关键算法、只返回预置结果或只打印 PASS 均不能证明真实集成。测试专用故障注入可以使用明确记录的 patch，但成功路径仍运行题目要求的生产行为。

## 3. 组件新增、修改和删除

适用 `CT-NAMESPACE`、`CT-INTERFACE`、`CT-DATA`、`CT-CONFIG`、`CT-ERROR`。

内建 Dataset、Model、Task 使用 `saebase/registry.py`；插件登记分别由 `datasets/plugins.py`、`models/plugins.py`、`tasks/plugins.py` 负责，执行分派由 `saebase/runner.py` 负责。第三方模型、SAE 与 metric 使用项目已经提供的显式 `saebase.extensions` API 及其 registry；这是现有公开扩展机制，任务不得再创建自己的平行 Registry。新增组件同时说明配置、调用者、兼容检查、样本或输出结构及来源。`saebase.datasets` 继续用于 SAEBase 数据模块，保留第三方 Hugging Face `datasets` 的包名。

新增 Dataset 需要真实读取或生成样本、稳定 sample ID、明确 split、有效数量和错误输入处理。新增 Model 需要真实构造、前向或题目所需能力、输入输出 shape/dtype、设备和身份检查。仅新增 YAML 或向列表追加名称不能完成组件接入。

删除题只处理题目明确列出的实验组件和功能。治理方在每道删除题起点提供可运行的、固定哈希的已安装组件；各题的可执行性不依赖另一名考生先完成新增题。执行者记录删除前的正常调用，然后清理实现、注册、别名、配置和可执行调用者中的残留。新进程中确认删除 ID 明确拒绝，保留 ID 仍可执行；保留历史实验、checkpoint、原始日志及来源记录。没有通用 unregister API 时，源码删除与进程重启就是检查对象，不能直接改私有 Registry 字典冒充生命周期实现。

已发布组件和公开兼容入口的去除不属于实验组件删除。题目要改变已有行为时，先声明受影响输入、旧配置行为、兼容决定和相应回归范围。

## 4. Task、Subtask 与产物层级

适用 `CT-NAMESPACE`、`CT-INTERFACE`、`CT-CONFIG`、`CT-ERROR`。

| 名称 | 含义与要求 |
| --- | --- |
| 七类操作目录 | `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark`、`train`，保留当前目录边界 |
| `task.name` | Registry 查找和 Runner 分派使用的 Task ID；新实验 ID 需题目明确声明，可由既有操作目录中的模块实现 |
| Subtask selector | 同一任务内的实际行为选择；声明支持的值、对应实现、输入输出和未知值错误，可采用任务局部 dispatcher |
| `task.path` | 产物目录层级，首段须等于实际 `task.name`，其余段通过现有安全字符和深度校验 |

新增实验 Task ID 不要求增加第八类顶层科研操作目录。对应配置必须能通过现有统一 Registry 和 Runner 进入实际实现，且不改变已有七类 Task 的含义。

新增或删除 Subtask 要改变可调用行为并验证相应输出。仅增加或删除 `task.path` 的一段目录不等同于新增或删除子任务。路径可以表达子任务名称，但不能代替 selector 的校验、分派和算法。

## 5. 参数契约与配置身份

适用 `CT-CONFIG`、`CT-INTERFACE`、`CT-DETERMINISM`、`CT-ERROR`。

新增参数在编码前声明名称、类型、默认值、有效域、单位、与其他参数的约束、缺省行为和旧配置兼容性。使用 Hydra 合成后的 resolved config 作为实际输入；命令行显式覆盖值优先于配置默认值。禁止在 Task 内无记录地覆盖已经解析的用户参数。任务局部参数不应要求修改通用 Runtime 才能生效。

至少比较两组具有不同预期行为的合法值，并覆盖实际相关的边界及非法输入。验证参数确实到达算法和输出，而不仅停在配置文件。涉及整数、有限数值、范围或枚举的约束，在昂贵计算或写入正式产物前检查。未知参数按冻结契约拒绝或明确诊断。

配置来源与配置内容分别记录。当前 Runtime 的 `metadata/run.json.config_sha256` 对 resolved YAML 文本计算哈希，它保留现有字段语义。另需比较语义等价配置时，使用明确命名的 canonical 配置哈希，对解析后的 JSON 可表示内容排序键序列化，再计算 SHA256。两种哈希不互相冒充；checkpoint、数据和源码身份也各自单列。

## 6. 小规模实验与数值结果

适用 `CT-DATA`、`CT-RESOURCE`、`CT-ENV`、`CT-DETERMINISM`。

题目预先冻结输入范围、有效样本数、token 上限、seed、CPU/GPU、时间、内存及下载预算。预检确认实际环境后才执行。一个数字预算不自动授予额外账户、下载或发布权限；缺条件时保留可复现的阻塞记录。

比较实验保留 baseline、改变项、固定项、原始数值、样本身份和指标定义。指标分母采用实际有效样本，缺失、被过滤或失败样本单独计数。正常结果必须是有限数值；NaN/Inf、空样本、shape 不匹配和不支持组合按冻结契约处理。异常不能变成零值、默认成功或从汇总中静默消失。

同一输入与 seed 的复现比较采用题目声明的容差和样本对齐方式。数值结果用于本题工程或小样本研究目标，不由少量样本推断完整数据集表现或科学结论成立。

## 7. 参数搜索

适用 `CT-CONFIG`、`CT-DATA`、`CT-RESOURCE`、`CT-RECOVERY`、`CT-TRACE`。

开始前冻结参数网格、有效组合规则、seeds、最大 trial 数及数据用途。需要选优的搜索还要冻结优化指标、方向、并列选择规则及验证/确认样本；纯枚举或中断恢复测试明确声明不选优，不临时加入收益目标。每个 trial 有唯一 ID、resolved config、输入与源码身份、实际命令、状态、退出码和产物位置。搜索中的每个完成 trial 都必须运行相应 SAEBase 任务。

trial ledger 覆盖全部计划组合。失败、超时、跳过和未运行 trial 各自保留状态及原因，不能用指标零值代替。报告同时给出计划、实际启动、成功、失败和未运行数量，并检查重复 ID、重复组合与遗漏组合。

选优仅使用预先指定的指标和数据部分。需要确认评估时，锁定被选参数后才运行确认样本，不能再用确认结果调参。恢复搜索只复用身份和产物完整性均匹配的成功 trial；每次新尝试保留原失败记录，避免把覆盖旧日志称为恢复。

## 8. 实验路径、文件管理与恢复

适用 `CT-SCOPE`、`CT-EVIDENCE`、`CT-RECOVERY`、`CT-ERROR`。

当前正式 Runtime 路径为单数 `experiment/`，依据 `saebase/runtime/experiment_paths.py` 和 Hydra 合成值：

`experiment/<dataset>/<model>/<task.path>/<sub_name>_<seed>/`

治理准备过程预建各独立工作树的 `experiment/` 输出根或在题目中指定已存在的受控根。考生使用 Runtime 的路径生成和校验函数；`experiments/` 如作为工作树外容器存在，也不改变其内部 Runtime 路径。Hydra multirun 继续使用当前 `_sweeps/<session>/<job_num>` 规则及真实 MULTIRUN 状态检查。

产物按 `checkpoints`、`metrics`、`tables`、`figures`、`logs`、`metadata`、`reports`、`tmp` 归类，分析组走 `create_analysis_group`。其 `description` 遵守现有 `目标：` 前缀要求，Context ID 通过权威图检查。`harness/experiments/validate.py` 通过仅表示它实际覆盖的分析组结构检查通过。

默认 `experiment.overwrite=false`。重复运行使用新 run 身份。恢复核对源码、配置、输入、seed、模型及 checkpoint 身份；明确哪些阶段重算、哪些产物复用。overwrite 清空旧叶子与 resume 续接状态是不同动作，分别验收。

清理先生成路径清单，只删除题目明确授权的临时对象；历史证据和模型数据保持完整。归档必须逐成员核验相对路径、大小和哈希，解包到新目录，拒绝越界、符号链接逃逸和损坏成员。失败时保留原始运行，不把半写产物标为 completed。

## 9. 真实证据与独立验收

适用 `CT-TRACE`、`CT-EVIDENCE`、`CT-PRIVACY`、`CT-AST`、`CT-CHECK`。

WP21 的 `contract_receipt.json` 使用独立 schema `saebase.wp21.contract_receipt.v1`，继续记录已选 CT、实际阅读路径、生产修改路径、命令和未验证项。旧 `harness/contracts/trace.py` 及旧 acceptance 流程限定 Harness 沙箱编辑，不能用于验收 WP21 的合法生产改动；WP21 使用本轮专用证据核验和外部 reviewer 重放，保持既有 22 个 Contract 内容。另交真实运行证据 manifest，字段名称以冻结题目/工具 schema 为准，至少覆盖下列信息。

| 证据 | 内容 |
| --- | --- |
| 源码身份 | 原始 base commit、初始化材料哈希、实际候选 diff 或源码清单哈希、实际导入模块路径 |
| 输入与权重 | 数据来源、样本数量和内容哈希、split；模型/SAE ID、版本、实际 checkpoint 哈希、层/hook；synthetic 或真实权重标记 |
| 配置与环境 | 实际 resolved config 及其文件哈希；canonical hash 如使用则单列；Python、关键依赖、设备和实际观测字段 |
| 执行记录 | argv、工作目录、开始/结束、退出码、stdout/stderr 路径和哈希；失败及未运行项 |
| 产物清单 | 每项相对路径、类型、字节数、SHA256；对应 run/trial 与输入配置身份 |
| 比较证据 | baseline 与修改后原始结果、独立重算的关键指标、容差、样本对齐及回归范围 |

Manifest 不收录自身的内容哈希，可由独立 checksum 文件记录 manifest 哈希。敏感路径、凭据、token 和无关账户信息不进入交付包。公开相对路径与私有本机日志的处理分别记录。

验收者在干净进程重跑被冻结的公开行为检查，并使用不同具体输入核对相同已声明不变量。检查原始产物与源码 diff，不能信任考生返回的 PASS 字符串。独立输入可以隐藏，评分条件在考前公开冻结。发现新需求先记 `EVALUATOR_SCOPE_GAP`，下一版再变更规范，不对已交考卷追加标准。

治理工具的结构检查、收据检查、预检和考生自测均有各自状态。`DESIGNED`、`PREFLIGHT_VERIFIED`、`SELF_TESTED`、`INDEPENDENTLY_VERIFIED` 描述不同证据阶段；实际工具沿用其声明的等价状态。所有成功和失败记录留存，独立技术验收也不自动成为产品发布或 Atlas 人工批准。

## 10. 失败归因与维护责任

沿用 [FAILURE_ATTRIBUTION.md](FAILURE_ATTRIBUTION.md) 的六类归因，不增加逐题临时 Contract。可以附 `owner`、`stage` 和可复现命令，区分 baseline_product、candidate、governance、environment 与 evaluator。

已有 SAEBase 缺陷由基线复现与源码证据确认；合同缺口和裁判漏检由治理维护者修复；考生只在授权代码范围内完成题目、保存证据并报告缺口。规则修订进入新版本，历史题目和原始失败记录保持可追溯。
