# Dataset 固定集合与自动检查

Dataset 负责来源、输入渲染和选择集合；Model/SAE 负责权重与能力；Task 负责计算与完成条件。统一入口为 SAEBase `conf/config.yaml`，没有额外管理任务。

当前组合包含 IOI、CounterFact、RAVEL、ARCChallenge、MMLUPro、GSM8K、HumanEval、TruthfulQA、HarmBench、XSTest、HotpotQA、BBH。WikiText-103 和 FineWeb-Edu 的活动组件、配置、准备产物与专属下载缓存已移除；历史实验与历史来源锁仍保持原文。

各 Dataset 配置 `sample_count: 500`、`sampling_seed: 42`。自然数据合并所选来源配置的全部原始 split，均匀无放回随机选取500条；不足500时全部保留。BBH 合并全部27个子任务；HotpotQA 选择 distractor 配置的 train 与 validation。split 作为来源标签保存，默认不用于后续任务筛选。IOI 沿用生成器和模型正确答案过滤，扩大候选数量直到取得500条合格输入，达到500条即停止剩余候选的模型前向，再按固定种子排列保留集合；达到明确候选预算仍不足则失败，不静默缩量。CounterFact 为自然数据，不再按模型答案正确与否筛选本轮观察集合。

`datasets/selection.py` 负责保存与读取固定集合。自然数据保存在 `data/<Dataset>/_shared/selections/<配置摘要>/`；IOI 按模型保存。`records.jsonl` 与 manifest 记录来源版本、种子、实际数量和校验摘要。正式任务使用完整固定集合；precheck 独立抽查最多32条输入，数量由 task.check_sample_count 控制。当前从可用集合中确定性抽样，不要求检查数量与正式任务一致。修改样本数量或选择种子会产生新集合。旧 v1 数据不冒充新集合。自然数据来源采用 HF 缓存；RAVEL 使用 city 实体与可渲染的 RAVEL 单实体模板，保留实体与模板各自 split；HarmBench 合并纯文本 validation/test，包括带上下文的条目。

`saebase.runner.run_task` 在执行任何具体 Task 前调用 Task 所有的 `tasks/precheck/gate.py`。仅配置不执行计算；precheck 自身不递归。检查失败保存报告并阻止目标任务。capture/intervene/generate 核对实际选择能力、数据文件完整性与最多32条输入的真实 SAE 编解码；其他任务使用既有、有界的任务预检查协议，未具备有效协议时受阻。检查不等同于已完成正式科研任务，也不保证每条输入的模型上下文或外部解释均可用。报告位于目标运行目录 `tmp/automatic_precheck/metadata/precheck_result.json`，由任务结果引用。

capture 空 `task.prompt` 时遍历 Dataset 的完整选定集合；显式非空 prompt 保留单输入模式。每条输入最后 token 的特征、激活、Neuronpedia 解释及错误写入 `reports/capture_samples.jsonl`，进度与完整性摘要分别记录。部分条目失败仍保存其他条目；无法取得已发布解释时保留测量并记录未完成，不生成解释填补。查询身份包含模型/SAE权重、层位、输入、选择集合和方法版本。

预检查默认隔离进程；可信矩阵在已加载的相同模型/SAE会话内复用实际资源，报告 `hard_kill_isolation=false`，由外层模型进程提供超时。不会因复用而省略真实前向与数据完整性检查。只读会话复用已有 Store 的 VerifiedCheckpointSession，退出时重新检查权重字节。Qwen3.5 使用独立解释器，避免改变其他模型的 Transformers 环境。

运行命令：`python -m tasks.capture.matrix --config conf/experiment/capture_matrix.yaml --tag <本次名称>`。解释器选择读取 `conf/runtime/environments.yaml`。本轮结果目录为 `experiment/central-maintenance/fixed500_matrix_20261113_v2/`；此前首条输入矩阵保留为历史开发记录，不能视为500条结果。每模型预计5514条：9个数据集各500，HumanEval164、HarmBench400、XSTest450。10模型共55140条目标观察，实际完成与受阻数量以账本为准。

软件回归在 `tests/product/test_fixed_dataset_cohort.py`、`test_public_dataset_sources.py`、`test_neuronpedia_capture.py` 和 `tests/precheck/`。Atlas 只读，既存来源所有权检查失败不作为本轮科学计算通过的证据。


续跑由 `saebase.runtime.experiment_results.inspect_run` 统一读取已有 manifest，验证结果对象并调用 Task 的完成规则。只检查 Task 声明的必要文件是否存在及已记录的字节数，不扫描特征账本。capture 区分 `stages.capture` 与 `stages.annotation`：激活全部保存但解释有缺失时返回 `annotations_pending`，调度器跳过模型重算并保留待补说明；未完成、失败、配置变化或记录不完整时允许新尝试。新尝试使用独立目录，原组合记录归档，不覆盖原产物。历史批次缺少分阶段字段时，由 capture 按其既有协议与数量记录解释，不改写历史文件。

计划数量固定来自 Dataset 选定集合；`completion.missing` 表示未提交样本，样本失败与运行错误分别记录。写入失败或迭代异常不会虚增一条失败样本。每个数据集的准备、检查与捕获失败独立保存，其他组合继续；公共模型/SAE 加载失败则逐项保存对应组合受阻或失败原因，继续下一个模型。

显存受限的指定组合可用 `--model <key> --dataset <alias> --offload-unused-layers` 创建独立重试。capture 在支持的 TransformerLens 后端使用 stop_at_layer，仅执行目标 Hook 所需的前缀；显式尾层卸载在预检查和捕获期间将未执行层移至 CPU，退出后恢复设备。模型权重、dtype、输入和层位保持原值；该策略记录在 experiment.capture_memory_policy。数值等价由合成模型逐值检查确认；实际重跑控制样本比较另存记录，适用范围为捕获，不用于完整 logits 或生成。
