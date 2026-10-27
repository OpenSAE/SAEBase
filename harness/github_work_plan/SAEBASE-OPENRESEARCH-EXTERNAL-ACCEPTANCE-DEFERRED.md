# SAEBase × OpenResearch：需要外部条件的验收清单

状态：GPU 第一轮已完成并留存验收报告。**D 节为本地 Agent 的下一轮修复与最小复验清单，执行仍须遵守 Harness 审批和本机预算；A/B/C 中尚无证据的高成本工作维持 DEFERRED。** 关联 [PR #60](https://github.com/OpenSAE/SAEBase/pull/60)；目标分支 `sync-openresearch-harness-20261027`。

## 用途与边界

本清单**只记录当前环境无法给出真实验收证据**的项目。Python 语法、CI、离线单元测试、静态代码审查、Hydra 配置组合、Registry 导入、Atlas 生成与前端脚本检查应当直接在 PR 的 CI 内完成，遇到失败立即修复，不得将可执行的普通代码检查转移到此处。此文档不是已授权的 Agent Flow、GPU 训练计划或发布许可。

验证时应记录代码 commit、命令、环境信息、实际输出路径与结果；`PASS` 只由对应命令及证据支持。不要把导入成功、静态源代码存在或合成测试当成科研结果有效。

## D. GPU 首轮验收后的修复与复验工作（2026-10-27，本地 Agent 下一轮任务）

**当前基线**：PR #60，已记录 GPU 实测的提交 `f7dca13dbc1d06ff06a3efb0780e7c2d27582282`。GPU 验收报告 [harness/github_development_doc/gpu-acceptance-20261027.md](../../harness/github_development_doc/gpu-acceptance-20261027.md)；原始私有证据目录 `/workspace/SAEBase-gpu-acceptance-20261027`。首次实测使用 RTX 5070 12GB、Gemma-2 2B、Gemma Scope SAE。CPU CI（Python 3.11/3.12、Ruff、Hydra、Harness、打包）在此基线通过。后续每次执行应先确认当前分支与 HEAD，并记录实际 commit，不能假定旧 SHA 是新 HEAD。

**本轮目标**：先解决已观察到的产品缺陷，再做最小真实 GPU 复验；生成代码、测试、报告和提交。无需重复所有七类功能的完整运行。不得将正常 Python/CI 修复放进本文件等待 GPU。原始 A/B/C 验收条款仍保留，未获得实测证据的部分不得宣称 PASS。

### D1. P0 — 修复 Hydra 首次运行的日志与实验目录冲突

- 证据：GPU 报告中 Capture 默认路径下的 `train.log` 被 Hydra 先写入，随后 `saebase.runtime.experiment_paths.guard_existing_run` 抛出 `FileExistsError`。此前通过修改 `hydra.job_logging.handlers.file.filename` 绕过。
- 检查 `saebase/runtime/train.py`、`saebase/runtime/experiment_paths.py`、`conf/sae_research.yaml` 及 Hydra 日志初始化时序，修复**首次干净运行**冲突，保留对于有历史产物目录的默认拒绝覆盖机制。不要以统一 `overwrite=true` 或删除历史记录作为修复。
- 回归测试必须覆盖：干净目录默认配置正常创建；同一标识重跑时拒绝覆盖；不同 run 标识互不影响；错误退出保留诊断信息；未设置额外 Hydra CLI 日志重定向时也能运行。
- 最小 GPU 复验：使用已缓存的模型与 SAE，重新从官方 `python -m saebase.runtime.train` 入口运行一次小规模 Capture，不添加临时日志路径绕过。记录命令、退出码及元数据文件。

### D2. P0 — 修复 SAE Lens TrainPlan 数据与精度的配置契约

- 证据：`tasks/train/backend/training.py` 已合入本轮成功的本地 2-step smoke 修正，但当前 `model_from_pretrained_kwargs` 直接写死 `dtype=float16`；`_open_local_training_dataset` 支持 JSON/JSONL 文件，而 `TrainPlan.validate()` 要求 `dataset_path` 必须是目录。
- 保留 Gemma-2 2B 在 12GB GPU 上已验证的低精度加载能力，将模型加载 dtype **显式、受限地配置化**，区分模型加载 dtype 与 SAE 激活/训练 dtype；默认值、支持范围及异常信息必须有明确文档。不能在不同 backend 间静默切换精度。
- 统一 TrainPlan 对目录快照、Hugging Face `save_to_disk` 目录、JSON/JSONL 本地文件的校验与实际加载语义，继续验证 immutable revision、真实文件存在性和权限边界；不得为通过测试取消来源校验。
- 单元测试使用依赖注入或 Mock 检验加载参数传递、路径种类、无效 dtype、拒绝无权限训练，以及兼容旧计划的行为；无需真实 GPU 才能验证的逻辑全部纳入 CI。
- 最小 GPU 复验：在同等或更小预算下用真实模型完成已授权的 2-step TopK SAE smoke，并验证 checkpoint 保存、`SAE.load_from_disk`、shape 与有限数值；记录实际 dtype。不得擅自扩展到层 25 长训练。

### D3. P1 — Circuits 原生图规范化与接口回归

- 当前实测仅证明 `circuit-tracer==0.5.0` 输出**原始** `native_graph.pt`，证据为 118 个节点和 3,636 条边，状态 `native_raw_graph_generated_not_normalized`；不等于完成了标准化图、因果边验证或 feature intervention。
- 为 `tasks/circuits/operation.py` 新增的 `offload`、`dtype`、`batch_size`、`max_feature_nodes`、`lazy_encoder` 透传补测试：参数准确、非法参数拒绝、默认行为兼容、`task.approved=false` 必须拒绝，真实运行的预算/超时限制保留。
- 研究 `tasks/circuits/backend/circuits.py` 已有 typed graph importer 与上游输出格式。如能取得**可信来源、明确版本的前端 JSON**，再调用正式 importer，并验证节点/边、error 与 logit 节点、端点 ID、原始 provenance、保存后 checksum；不要直接加载不可信 `.pt` 或猜测 node ID 与 feature ID 的对应关系。
- 若当前真实产物无法无损转换，报告 `BLOCKED: native graph normalization`，写明缺失的具体上游接口/证据，不允许创建虚构的标准图。真实电路 feature intervention 也应与生成图分别验收。

### D4. P1 — Explain 生成器接入与证据边界

- 当前 `tasks.explain.operation.execute(cfg, generator=...)` 的真实模型调用已成功；普通 Hydra YAML 不能存 Python Callable。这是可用入口限制，并非语义解释已经验证。
- 评估可复用的显式 `generator` 装配方式，使 CLI 能选择已验证的本地/注册模型实现；禁止把任意字符串当作可执行代码、在 YAML 中嵌入对象或隐式请求外部 API。若会破坏现有边界，保持独立 Python 入口并在 README 中给出明确调用方式。
- 添加 CPU 单元测试，覆盖未提供 generator 时显式失败、成功注入、错误输入、留出样本身份、没有语义标签时不产生语义正确率主张。
- GPU 报告中的 `heldout_score=1.0` 是特征激活有无的预测一致性，**不是**自然语言解释的语义准确率。保持两项指标严格区分，未经人工/独立评价不得标为解释质量 PASS。

### D5. P1 — Wheel 与源码安装契约

- 已观察到 v0.5.0 wheel 包含原 SAEBase CLI，但不包含新 `dlbase`。检查 `pyproject.toml` 的 package discovery、CLI scripts 与兼容性要求。
- 目标是 SAEBase 与新增 DLBase 在声明支持的安装方式下行为一致。如果将 DLBase 纳入 wheel，需保持现有公共 CLI/模块不变，并添加**脱离源码目录**的 clean venv 构建安装、CLI、Hydra 配置与最小 Registry 导入测试；注意 Hydra 配置资源也必须随 wheel 分发。
- 如果设计决定新 DLBase 只能作为源检出开发工具，必须在两份 README、安装说明和发布清单中明确，且增加测试防止误宣称 wheel 可运行它。不能无评审改变 v0.5.0 发布面或自动上传 PyPI。

### D6. P2 — 环境与验收报告收尾

- 主环境 `/venv/main` 的 vLLM 安装不完整；已有隔离虚拟环境中的 vLLM 0.27.1 能完成真实 8-token 基线验证。只在独立隔离环境中核对可重复性，未经许可不重装主环境、不更换全局 torch/CUDA。
- 修订 [GPU 验收报告](../../harness/github_development_doc/gpu-acceptance-20261027.md) 中的“代码改动（未提交）”历史措辞：两项改动已经进入 PR；记录对应 commit、测试与复验状态，保留首次实测时的真实 HEAD 和数据，而非覆盖历史信息。
- 报告区分 `runtime smoke passed`、`partial/limited validation`、`scientific acceptance pending`。Circuits 仍有标准化/干预缺口、Explain 仍无独立语义验证、Train 仅为 2-step smoke。
- 只将脱敏的小体积 receipt、数值摘要、哈希与执行命令纳入 Git；不上传 HF token、原始私有 prompts、模型权重、大型 checkpoint 或未经批准的原始日志。

### D7. 全局验证、输出和禁止事项

1. 不引入与 LLM/SAE 无关的图像任务。优先运行 `python -m pytest -q tests/test_no_unrelated_components.py` 和完整原 SAEBase 回归。
2. 运行 `python .cursor/hooks/validate_repo.py --full`、Python 3.11/3.12 CI、Ruff、Hydra 配置组合、轮子构建与 CLI，以及 Harness/Atlas 的 source/semantic checks；对于修复的每个缺陷新增可稳定重放的自动测试。
3. GPU 仅对受此次改动影响的最小输入复验；使用已缓存的模型和权重、明确的测试预算、独立新实验目录。不要反复跑已确认相同结果，也不要扩大训练步数。
4. 保持 `saebase/` 公共 API 与已经通过的七类 LLM Task 功能兼容。Harness Core、`.cursor`、审批 Ledger 与科研状态都不属于普通任务 Agent 可自行更改的范围；遇到治理权限限制，提交精确 patch/提案给负责人审核，不绕过 Hook。
5. 完成后交付 **按 D1–D6 编号的验收矩阵**，每项写明 `PASS / FAIL / BLOCKED`、变更文件、commit、测试命令、真实结果、证据文件路径和剩余风险。把已验证的修复提交到 PR #60 分支，但**不合并 PR、不发布 wheel/PyPI、不自动绿化 Atlas 科研节点**。
6. 对需要新模型下载、B2 写入、额外 GPU 费用、显著长训练或公开部署的步骤，先单独获得明确授权；未授权的项目保持 DEFERRED。

**完成标准**：D1、D2 和相关 CPU 回归/最小 GPU smoke 通过；D3、D4、D5 的可执行代码测试通过并清楚标出尚未科学验收或依赖外部条件的项；PR 的全部自动 CI 成功；报告中没有把部分验收夸大成端到端成功。

---

## A. 实际模型与 SAE 权重的 GPU 兼容性

原因：当前 GitHub 离线 CI 不具备所选的真实 GPU、权重授权与磁盘缓存。

前提：固定 Git commit，明确使用的 Hugging Face revision、SAE checkpoint 哈希、CUDA/PyTorch/TransformerLens/SAE Lens/NNsight 版本，事先批准 GPU 时长与费用。

1. 在真实 Gemma-2 2B 与至少一组 SAE 权重上运行 `capture` 与 `intervene`，验证 hook 名称、层数、SAE 维度、完整 token ID、身份替换后 logits 容差与受控编辑后的差异。验收为可重现的真实数值、源代码路径和完整报告。
2. 对 `generate` 核验正常续写及有控制条件的 feature steering，明确比较不带干预和带干预结果；复核是否复用 SAEBase 现有 `saebase.generate` 后端，而不是两套生成实现并行。检查 vLLM worker/HTTP、KV cache 与采样种子。
3. 对 `circuits`、`explain`、`benchmark`、`train` 分别执行具有明确数据输入的最小真实验收，核查报告、模型授权、计算预算、可复现的产物及安全门控 `task.approved`。训练需要确定模型、数据、步数、优化器与输出位置后单独批准。
4. 比较现有 SAEBase v0.5.0 实现与集成分支在**相同权重、相同输入、相同种子**下的可比指标，设置明确的预期数值容差；有差异则追溯结构改动。

产物：`experiment/` 下独立命名实验目录、解析后的配置、权重/revision/数据指纹、机器快照、JSON 报告、测量的耗时与测试命令。所有科学结论仍需人类审阅。

## B. 真实外部数据加载与资源授权

原因：本次无可供授权的外部数据版本、网络下载额度与独立数据源许可证核查。

1. 对 `datasets/NLP` 与现有 `saebase/data` 的重叠数据集（CounterFact、GreaterThan、IOI、MMLUPro、SimpleCopying）建立名称、split、prompt/text/token、来源 revision 的一致性矩阵。BBH、SubjectVerbAgreement 及其他来源另列。
2. 真实下载时固定 revision，检查第三方 `datasets`、`transformer_lens` 和 `sae_lens` 能在 SAEBase 命名空间中正常导入。用只读数据样本检查 `research_datasets` 路径，避免改变旧 `saebase/data` 产物或已发布结果。
3. 对 Research Lab 原生 Hugging Face streaming、dataset object、tokenizer 接口进行端到端读取测试，检验 schema、shuffle/revision 与缓存行为，确认实验不会静默使用不同数据划分。

产物：每种数据源的 immutable revision、license review、样本量、hash、split、read-only 测试报告、与历史协议差异表。

## C. 实际多硬件环境与发布验收

原因：需要项目授权的部署目标、凭据或物理机器，不能依据本轮只读 GitHub 数据代替实测。

1. 在目标 CPU/GPU 机器运行 `harness/environment/probe.py`，核验磁盘路径、可见设备、CUDA capability、driver、RAM 与部署依赖。
2. 仅经明确授权才执行 Backblaze B2 / rclone 实体传输，验收双端校验和及恢复策略。没有凭据时不测试生产桶。
3. 完成可安装 wheel 与源检出两条路径的实机 CLI 验收。核验 SAEBase 老入口和 DLBase 新入口均能独立执行，无碰撞。
4. Atlas 静态视图应以本 SAEBase 源码、实际 Knowledge/Code/Environment 证据重新生成；公开发布需要用户指定目标网站与权限。人工复核 Knowledge/Core 导入节点的状态、Evidence、跨 Tree 链接与中英文本。原 OpenResearch 图的绿色状态不能直接成为 SAEBase 通过证明。

产物：环境快照、部署流水线日志、源码和页面资产 SHA、权限审批、回滚说明与人类审批记录。

## 进入实测的门槛

- PR #60 的 CPU CI、原 SAEBase 回归、Python 3.11/3.12、构建和静态验证均通过。
- 训练/下载/外部部署前有明确授权、预算、目标及可回滚产物路径。
- 每项实验结果写入 Ledger，明确 passed / failed / blocked，保留失败证据；只能由已记录的实测和人类审批更新相关科研节点状态。

