# SAEBase × OpenResearch：需要外部条件的验收清单

状态：DEFERRED，未授权自动执行。关联 [PR #60](https://github.com/OpenSAE/SAEBase/pull/60)；目标分支 `sync-openresearch-harness-20261027`。

## 用途与边界

本清单**只记录当前环境无法给出真实验收证据**的项目。Python 语法、CI、离线单元测试、静态代码审查、Hydra 配置组合、Registry 导入、Atlas 生成与前端脚本检查应当直接在 PR 的 CI 内完成，遇到失败立即修复，不得将可执行的普通代码检查转移到此处。此文档不是已授权的 Agent Flow、GPU 训练计划或发布许可。

验证时应记录代码 commit、命令、环境信息、实际输出路径与结果；`PASS` 只由对应命令及证据支持。不要把导入成功、静态源代码存在或合成测试当成科研结果有效。

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

1. 对 `research_datasets/NLP` 与现有 `saebase/data` 的重叠数据集（CounterFact、GreaterThan、IOI、MMLUPro、SimpleCopying）建立名称、split、prompt/text/token、来源 revision 的一致性矩阵。BBH、SubjectVerbAgreement 及 CUB-200 另列。
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

