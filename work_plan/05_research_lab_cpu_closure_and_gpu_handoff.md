# Work Plan 05 — Research Lab CPU 阶段收尾、500 条候选语料锁定与 GPU Agent 执行清单

创建日期：2026-10-18  
状态：CPU 基础已合并 main；2026-10-19 GPU agent 已在独立分支提交真实采集与初步分析记录，成果暂存未归档的 experiments/，尚未进入 main。GPU-02 报告 48/48 站点、GPU-04 六站点样例已完成；GPU-05 SAR 0 组、GPU-06 RSS 为待位置审计的部分结果，GPU-09 公开计算仍关闭。本文件 5.2 节是当日加速执行的最新规则。  
母计划：[04 Research Platform Execution Plan](04_research_platform_execution_plan.md)。  
归档位置：work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md。

## 0. 最终目标与已经决定的范围

OpenSAE Research Lab 本轮要交付的是一个真正可用的平台，而不只是 SAEBench 风格的静态表格。五项能力均保留：真实指标浏览、Feature Explorer、交互式干预、SAR/RSS/Localize 科学分析，以及可分享、可追踪、可复现的实验记录。

基础数据由 **10 个公开来源，每源 50 个独立原始 prompt groups，共 500** 构成。每源固定 20 discovery / 30 evaluation，总数 200 / 300。所有输入变体都继承父 group，不增加独立样本数。

GPU 覆盖以现存 18 个 model–SAE 配置、48 个验证站点为基础。平台深度浏览和干预优先 6 站点，标准 SAR 先在 3 站点执行，在线新输入优先一个真实模型。两套额外 6-factor × 64-state controlled designs 专门服务 Localize，不属于这 500 条自然文本。

**严禁**把旧 48 站点上的 64-group Stage D 工程证据称为新 10×50 语料的测量结果。绝不能把合成真值、候选文本、模拟生成或已经规划的作业标记成真实 GPU 结果。

## 1. 已完成且可由 GitHub 直接检查的 CPU 工程

### 1.1 平台和数据层

PR [#38](https://github.com/OpenSAE/SAEBase/pull/38) 已合并，包含：

- Research Lab 网页及 read-only FastAPI：Results、Features、Intervene、Analyze、Datasets、Reports。
- 旧 48 站点真实工程指标的检索和逐行原始报告链接。
- 两种 Feature 浏览索引：小批次 JSON 与可分页的 SQLite，后者按 prompt group 统计出现次数，只接收实际 checkpoint 身份匹配的非合成观测。
- 特征文本示例仅在来源获准公开时展示；标签为空时不编造含义。
- 干预 preflight：只验证已登记的站点与操作，不运行模型；未经认证的 POST /api/jobs 仍返回 503。
- 有状态的 CPU SQLite 作业原型：幂等请求、额度、worker 租约、重试、取消、owner-only 状态；它不保存私有原始 prompt 文本，也**不是**已经可公开使用的 GPU 服务。
- 只接受真实源数据和固定哈希的 SAR/RSS CPU 适配器、原始 Localize 引用，完整候选 continuation 读出与 token 锚点校验。
- 校验哈希的不可变报告、回放与发布结构、CPU Docker 和 HF Docker Space 部署配方。
- 原始 SAEBase CI、Python 3.11/3.12、打包、Docker 启动和数值研究回归；合并检查通过。

PR [#40](https://github.com/OpenSAE/SAEBase/pull/40) 也已合并，准备了：

- 两套完整 6-factor / 64-state bitmask 有序输入。
- 原始 SAEBase ControlledInputDesign 转换和实验来源身份。
- 明确标记为 synthetic 的已知六因子特征真值，并在原始 Localize 上做了 CPU 数值验收。
- GitHub Actions 可下载的两份六因子 JSON；64 个状态属于**一个完整群体**，不能当作 64 个独立 prompt。

### 1.2 本轮 CPU 补充，PR #42

[PR #42](https://github.com/OpenSAE/SAEBase/pull/42) 在以上正式工程基础上进一步完成：

1. **修复 WikiText 的重大 provenance 风险**。WikiText 是文章结构下的行级语料，原来的全行 shuffle 可能把某段落错误归到另一篇文章。现在先按原始顺序识别文章标题，再做确定性组排序。不得重新使用旧 shuffle 结果。
2. **支持 exact upstream revision replay**。HF 下载可指定 40 位固定 Git commit，且校验实际下载解析的 revision；不能静默换 latest。
3. **数据源实际采集与元数据锁**。第一次联网 GitHub Actions CPU 运行冻结了 10×50 个 candidates，随后修正 HellaSwag 标签平衡和增加 near-duplicate 检查，并形成最终 source lock。
4. **HellaSwag 预先固定的标签平衡**：A/B/C/D 数量 13/13/12/12；BoolQ 原有 25 yes / 25 no。
5. **自动近重复检查**：对 500 条实际候选使用 Unicode casefold 五词片段 Jaccard，阈值 0.85；只输出 sample ID 与重合分数，不在公开审计中输出原文。这是 lexical screen，不是语义重复或训练污染证明。
6. **十种任务的 RSS proposal generator**：加入 WikiText/TinyStories 的续写包装，以及 HumanEval 的 Python 注释/空白包装。所有变体都是待审核 proposals；不能默认视为 meaning-preserving。生成时保留完整原始输入文本和变体父组。
7. **完整 22,500 变体 CPU 身份生成**：每条 45 个、每个变换设置 5 selection + 10 held-out；共 7,500 selection、15,000 held-out。每条 proposed text 都参与 SHA-256，不上传实际变体文本。
8. **来源锁复验器**：验证 10 个上游精确 commit、原始导出哈希、50 条候选文件哈希、每条 sample/group identity、200/300 split、500 条元数据整体摘要和 RSS 变体指纹。候选变化即停止。
9. **原始 SAR/RSS 的 CPU 数值 parity**：把 Research Lab 适配器同原始仓库固定 commit 上的真实公式输出逐项比较，包含合成已知交互和 held-out 方向测度。这与生成真实 checkpoint evidence 不同。
10. **数据来源与许可问题单独追踪**：[Issue #41](https://github.com/OpenSAE/SAEBase/issues/41)。未经权限与内容审核的第三方 prompt 不发布到公开 GitHub/HF；任何开源仓库不得直接把它们统一标成 SAEBase 的 Apache-2.0。

### 1.3 冻结结果及验收证据

正式候选锁：apps/research-lab/data/source_lock_v1.json。  
来源锁 schema：opensae.corpus_source_lock.v1。  
原始研究数据类别：candidate_input_only；非模型激活、非校准完成的测试集。

成功的 CPU 审计证据：
- [GitHub Actions Run 39174535881](https://github.com/OpenSAE/SAEBase/actions/runs/39174535881)
- Artifact ID：12154265086，名称 opensae-corpus-cpu-preflight。
- Redacted audit manifest SHA-256：3f044cc31c34c0ea7be4c723921323be90f716183d1276a6dfa58c5a605f7a22。
- Candidate groups：500；discovery 200、evaluation 300。
- 完全相同文本的跨来源 SHA-256 冲突：0。
- 五词片段 Jaccard ≥ 0.85 的候选对：0。**人类语义近重复审核仍未完成。**
- RSS proposals：22,500，selection 7,500，held-out 15,000。
- RSS proposals metadata SHA-256：22fac1b100387ed1a2374dbcb6f8bb593f20a69b4be07d6c63533bc98bbaebf4。
- 所有 source 与 RSS 结果均保留 human_reviewed=false / license_reviewed=false。数据包不包含原始 prompt 文本，不能当成可直接公开展示的语料。
- 来源锁若与再次生成的数据不一致，必须停机排查，不得继续沿用“500条通过”报告。

具体十个来源和锁定的上游 SHA 见后表。真实模型与 SAE 的计算次数：**0**。

## 2. 固定十个数据来源与上游版本

| Dataset ID | HF 来源/配置/原始 split | 固定 HF commit |
|---|---|---|
| wikitext103 | Salesforce/wikitext · wikitext-103-raw-v1 · train | b08601e04326c79dfdd32d625aee71d232d685c3 |
| tinystories | roneneldan/TinyStories · validation | f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 |
| squad | rajpurkar/squad · train | 7b6d24c440a36b6815f21b70d25016731768db1f |
| boolq | google/boolq · validation | 35b264d03638db9f4ce671b711558bf7ff0f80d5 |
| arc_challenge | allenai/ai2_arc · ARC-Challenge · test | 210d026faf9955653af8916fad021475a3f00453 |
| hellaswag | Rowan/hellaswag · validation | 218ec52e09a7e7462a5400043bb9a69a41d06b76 |
| winogrande | allenai/winogrande · winogrande_xl · validation | 01e74176c63542e6b0bcb004dcdea22d94fb67b5 |
| gsm8k | openai/gsm8k · main · test | 740312add88f781978c0658806c59bc2815b9866 |
| humaneval | openai/openai_humaneval · test | 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544 |
| truthfulqa | truthfulqa/truthful_qa · multiple_choice · validation | 741b8276f2d1982aa3d5b832d3ee81ed3b896490 |

每个来源 50 个原始组，不是任意前 50 行。使用固定 seed 20261018；WikiText 按原序文章边界、SQuAD 从 35,000 行候选中取 50 个不同原文文章组，其余多数来源从固定 commit 的 bounded shuffled stream 中抽取。它们不是整个原始 dataset 的严格均匀随机样本。不得在看到模型或 SAR/RSS 结果后重新选择更好看的例子。

官方源内容许可证需逐项审核。特别注意 WikiText 数据卡多处 CC BY-SA 版本表述及历史文档许可，WinoGrande 的 dataset 与 code 采用不同条款。HellaSwag 还需确认上游文件具体许可。详见 Issue #41。

## 3. GPU Agent 开始前如何在 CPU 上恢复并校验候选数据

所有命令均从已包含 PR #42 的 SAEBase commit 执行，不依赖 GitHub Actions 临时 URL 是否过期。

1. 获取 repo，检查 main、work_plan/04 与本文件、data/source_lock_v1.json；冻结并记录本次运行 commit。
2. 在网络通畅、具有源数据读取权限的 CPU 环境建立隔离环境，仅安装 datasets 与 huggingface_hub，不需要模型权重。
3. 运行固定 SHA 的 CPU 审计，然后锁复验：

~~~bash
cd apps/research-lab
python -m pip install datasets huggingface_hub
python -m backend.source_audit \
  --source-lock data/source_lock_v1.json \
  --max-raw-rows 2500 \
  --output /secure-private-workspace/opensae-corpus-audit.json
python -m backend.corpus_lock \
  --audit /secure-private-workspace/opensae-corpus-audit.json \
  --lock data/source_lock_v1.json
~~~

4. 检查输出 sources=10、groups=500、discovery=200、evaluation=300、variant_proposals=22500、gpu_measurements=0、锁摘要完全匹配；并确认没有原文被写入公开日志。
5. 单独用 backend.acquire 与 backend.ingest 重建 full private 500-row dataset，保留逐源的 source export 和 release manifests。原始数据及 HF token 若有，**只放在私有工作区**。
6. 完成每条 prompt 的内容检查和许可/再分发审批；公开显示需要确认，否则继续保持私有且不通过公共 API 发布。请勿通过伪造 human_reviewed=true 或 license_reviewed=true 绕过。
7. 一旦需要更改无效例子、许可证限制导致换行、或改动采样/变体协议，则**必须生成 v2 版本锁并重新审计**。原始锁保持不可变，GPU 作业不得悄悄替换输入。

**边界说明**：CPU 采集候选数据已完成；源数据文字的人工内容审核、部分许可证确认，以及每个真实模型 tokenizer 的访问与精确 token 长度扫描仍需要后续有权限的执行环境，且不需要 GPU 本身。这些是外部授权/人工工作，不是“GPU 已做”的证据。

## 4. 下一步 GPU Agent 执行总原则

执行者：具有真实模型权重、正确 SAE checkpoint、GPU 和持久存储权限的 agent。  
第一优先级：**研究设计和数值语义正确**，其次才是高吞吐和页面漂亮。  
全面执行前先 20 prompts pilot，每数据集两条；所有耗费较大的批次必须先测吞吐、显存和硬盘增长，然后确定上限并提交预算。

建议单卡优先 48GB VRAM，并保留约 500GB 顺序工作盘和独立外部持久化；RTX 5090 32GB 可以试，但必须通过真实 checkpoint/SAE/dtype/batch 的内存 pilot，不能擅自量化后仍沿用原实验身份。一个时间点只装必要模型，避免同时存放全部权重。没有实测吞吐前不能虚构准确费用或运行时长。

不能自动购买或无限扩大计算；设显式 budget、checkpoint、resume 和 fail-closed 错误策略。公开服务不能直接对未认证的任意请求启用 GPU。

## 5. GPU Agent 工作包：按依赖顺序完整执行

### GPU-00：环境与来源 Gate（未通过不得开大批次）

- 记录 checkout commit、所有配置与原始研究源码 SHA-256、PyTorch/transformers/sae-lens/CUDA 环境。
- 成功重建并核验 500 source groups；原始 group 及派生 variants 不跨 discovery/evaluation 分裂。不可将 45 个 variants 作为 45 个独立样本。
- 校验完整 48-site manifest 与真实下载的模型/SAE 权重。历史 48 站点中的 **3 个原始记录缺少完整权重哈希**（pythia410m/mlp_topk/L4、qwen3_1_7b_base/resid_w32k/L7、r1_distill_qwen15b/resid_l20/L20），只允许作为历史工程结果浏览；新实验必须先在实际 loaded checkpoint 上重算/补全身份，不得用空值当真实哈希。
- 核实 Llama/Gemma/Qwen/Mistral 各模型是否需要访问许可、checkpoint tag、dtype、hook、tokenizer/chat template；Mistral 指定站点是 MLP hook，不可改成 residual。
- 逐数据集固定 render/readout、答案完整候选概率、target token 选择与 position policy。输入不得泄露 gold answer；代码不得随意执行。生成和任务评分分开。
- 对 WikiText、SQuAD、代码文本进行模型特定的 token length、truncation、offset/alignment 验收，失败例子不可静默截断后保持旧 identity。

交付：GPU-00/preflight.json，dataset_identity.json，checkpoint_registry_audit.json，分模型 tokenizer_and_position_report.json，完整 source hashes 与通过/失败清单。

### GPU-01：最小真实 pilot（每个来源 2 条，共 20 原始组）

- 先用现有已验证的小模型/SAE 跑一次真实 capture/encode/intervention，再选目标大模型跑一次。
- 校验 identity patch、ablation、scale、全 SAE 重建和普通输出的预期语义；当前 SAEBase 每次 measurement 是一个选定 token 位置，不能宣称支持任意联合多 token patch。
- 同一输入在 clean/edited 时使用相同读出定义，但不预先假定输出相同。保存 exact token IDs、patch timing、label readout、模型/SAE hash 与原始 response。
- 报告 token/s、measurement/s、p95/p99 可选延迟、GPU peak VRAM、host RAM、raw/cache/archived bytes 每组。
- 用 pilot 结果确定 batch、shard、每 job budget、retry、磁盘释放策略。GPU-01 没验收通过前禁止一口气启动全部 SAR 组合。

交付：pilot_20/manifest.json、完整 20 条 ledger、cost_and_capacity.json，所有原始文件哈希，明确 go/no-go。

### GPU-02：48 站点 × 500 原始组的基础激活和 SAE 编码

- 重建新 cohort，保持 Work Plan 04 旧 48-site historical engineering 验收不变。
- 全部 48 个注册站点分别尝试 500 条，理论上 24,000 site–base-group observations，**不是 24,000 个独立自然问题**。
- 利用同一模型前向采集兼容 hook，但必须经新 API 对等验收。真实采集的 all-nonzero SAE codes、token 位置、原始模型响应和 reconstruction 指标写入 sharded stores，不能用网页 top-20 代替科学输入。
- 不把模型/SAE 跨层 code ID 和 activation magnitude 当作共同比较坐标。
- 每站点登记 started/completed/skipped/failed、actual_group_count、source split、耗时、VRAM/bytes、hash 和重试原因。所有失败都明确纳入 denominator 及 coverage ledger。

交付：capture_48x500/、site_coverage.csv、cache_and_code_manifest.json、source_digests.json。禁止用旧 64-group M3/M4 数值冒充新测量。

### GPU-03：6 个主站点的真实 Feature Explorer

六个固定主站点必须从 apps/research-lab/data/coverage.json 精确读取，禁止重新手写 hook。

- 建立真实 feature 活跃索引：每条输入的样例 token、高/低激活对照、至少受控样本内的出现分布。精确记录模型、SAE、layer、feature ID、tokenizer 版本与输入文本权限。
- 目标各站点精选 50 张卡片，总计 300；**如果实际数据不足，报告观测不足**，不能补假解释、虚构激活或把未观察到的 feature 称为 dead feature。
- 全量科研 codes 保存在原始稀疏数据层；网页索引仅存受控 top examples 和分页查询数据。索引文件需校验 SHA-256。
- 人工标签与模型生成候选解释、真实激活证据、干预结果完全分开并保存 source/split/status；禁止用 evaluation 调完解释后又当作独立 held-out。

交付：features/index 或 SQLite DB、feature_cards/、真实示例/对照 provenance、检索 API 端到端测试。

### GPU-04：6 主站点的干预和生成比较

- 使用原始 SAEBase operators：identity、ablate、scale、clamp、add、coalition、full reconstruction。首版编辑位置限制以真实实现为准。
- 对每模型准备至少一组清晰 baseline/edited 对照和 target-readout 记录。不得把随机生成变化当成稳定能力提升。
- 生成先使用已验证 reference full-prefix 语义；未经对等验证不能声称 KV cache 或高吞吐推理已支持。
- 在线自定义计算至少覆盖一个真实模型。必须在认证、预算、私有文本保存、队列、取消、worker 恢复、OOM/failure injection、报告提交一致性测试通过后才允许公开入口启用。
- 身份/安全 Gate 不通过时，网页可继续回放旧真实数据，但 POST /api/jobs 保持关闭。

交付：interventions/ 原始 clean/edited pairs、job worker parity/safety tests、replay reports；至少一个符合条件的真实 online experiment。

### GPU-05：标准 SAR（3 主站点）

- 原始 SAR 方法固定 K=16、M=384、fit=256、held-out=128；必须使用原始公式。每组保存真实 mask、每个 coalition 的 scalar readout、独立 g_full 和精确 fit/held split。
- 主批次：3 站点 × 500 base groups × 384 = **576,000 次逻辑 coalition 测量**；这不是 576,000 个独立数据点。
- 额外在预先固定每来源 5 条、共 50 个基础组上，对 3 站点执行 matched rotation 对照，增加 **57,600** 次；合计计划 **633,600** 次 coalition 测量，另有基线、对照和失败重试的成本。
- 若 active features 少于 16、输出方差不足、设计矩阵秩不足、NaN/undefined rho、GPU 内存不足，记录真实异常和实际 K，不能伪装为正常 K16 数据。
- SAR1/SAR2/Delta2/rho2/rank/ordering/sign/regret 与二阶系数诊断必须保留 original parity 和基线；二阶热图称 fitted interaction，不称已经证明的 complete circuit。
- 按原始 prompt_group 分层/重采样，不能将同一 prompt 的 384 masks 当作独立组。

交付：sar/raw_masks_and_g、sar/per_prompt_metrics、sar/pairwise_diagnostics、sar/rotation_controls、sar/fit_held_provenance、error_ledger。

### GPU-06：RSS（6 主站点）

- 现有 CPU proposal 总数：500×45 = **22,500**。每原始 prompt 加 clean 则为 46 个输入。
- 六个主站点如果每条全部有效，最大 6×500×46 = **138,000 site–input occurrences**；实际模型 forward 数取决于多 hook 共享、失效变体及重试，不等于独立自然 prompt。
- 三套变换 local_format、instruction_wrapper、context_wrapper，每套每组 5 selection 和 10 held-out；10 种来源均有 CPU proposal，但所有 proposal 的 semantic_review 与 tokenizer_alignment 仍待验收，必须由 dataset-aware 规则核对。
- HumanEval 只允许经过独立验证的 Python 注释/空白包装，不能在代码执行环境中执行任意生成内容；WikiText/TinyStories 的续写上下文变更不得被宣称 meaning-preserving。
- 对每模型按实际 tokenizer offsets 重新计算锚点位置。输入相同整数 token index 并不能证明内容对应。不可比较新 wrapper 的结尾与旧 clean 的另一段文本。
- 原始 RSS selector 返回支持集、响应方向、MeanRSS、held-out 持久比例。空支持集记录 undefined，不把 RSS 直接解释成安全风险/下游正确率。
- 统计独立单位始终为 base group；selection variants 只能用于筛选特征，held-out variants 用于检验，不能互换调参。

交付：rss/proposals_reviewed、rss/clean_vs_variants、rss/selection_support、rss/heldout_validation、rss/invalid_variant_ledger、rss/site_coverage。

### GPU-07：Input Localization 真模型 grid（与自然 prompt 分开）

- 从 PR #40 已有的两份 64-state CPU grid 与 source manifest 读取。原始 synthetic known-factor control 只用于正确性验收，不需要 GPU，也不能当 learned SAE 结果。
- 在指定真实小模型上为 real_model_six_factor_design 生成每状态一个唯一 token sample；检查全部 64 个状态都存在，因子 bitmask 0..63 顺序、唯一文本、位置与完整 checkpoint 一致。
- 调用原始 SAE-Input-Localization 对 frozen codes 计算 exact scope curve / 90% budget / assignment。不同 feature 数量比较需按实际可比性约束，不伪造统一质量排名。
- 状态间的差异来自受控因子，不可把 64 个 grid states 当作 bootstrap 的 64 个独立样本。若另做 paired estimator，报告 Monte Carlo seed sensitivity，不称总体置信区间。

交付：localize/full_64_state_manifest、learned_code_grid、exact_curve、synthetic_parity、readout_and_hash_audit。

### GPU-08：CPU 后处理、真实报告与可分享发布

这些步骤本身不需要 GPU，但依赖 GPU-02 到 GPU-07 的真实 measurement 文件。

- 使用已有 backend.methods 调用固定原始 SAR/RSS/Localize；发布经身份/来源/哈希/权限验证的 analysis index，拒绝非有限值伪装。
- 通过 backend.feature_store 构建可分页 feature DB，保留 display-only 与 full codes 的区别。
- 将新的真实指标、任务读出、有效 N、matched comparisons、分组 CI、有限任务覆盖率写到新 cohort 页面。比较必须检查 model/SAE/hook、input release、tokenization、metric、split 和 protocol 兼容性。
- 每份结果生成唯一 run_id、SHA-256、source code commit、model/SAE/tokenizer sha、完整配置和可下载的 reproducer；复跑是新 run，不能覆盖旧报告。
- 独立环境完整下载一次公开 release 并验证，报告失败/缺失项。

交付：research-lab-release-v1/，统一 dataset and model cards、reports.json、feature DB、metric ledger、raw audit map、artifact digests。

### GPU-09：上线验收和外部用户测试

- 只在合法展示数据和授权环境存在时发布正式 HF Space / API。CPU Docker 浏览站点与受限 GPU worker 隔离；HF 不可信临时磁盘不能保存唯一私有作业队列。
- GPU 不在线，公开 Results/Features/Reports 仍可正常访问；在线按钮说明不可用而不是伪 loading。
- 对登录、额度、两 worker 抢占、lease 过期、stale attempt、取消、数据删除/TTL、文件权限、路径 traversal、输入 XSS 进行独立负面测试。
- 至少三个未参与开发的研究者无需维护者指路，完成 browse→feature→intervention→analysis→share 主流程；页面须清楚标记这次实验真正在什么模型/输入上跑过。
- 提供运行手册和回滚手册；所有功能是否完成以实际验收记录为准，不以上线截图、README 文字或 PR 标题为准。

交付：live_service_acceptance.md、production_link、three_user_trials.md、rollback_test.json。

## 5.1 原执行修订的适用关系

2026-10-18 的决定仍有效：继续同一份 release，不因为耗时较长就跳过大型计算，不改原始 SAR/RSS 数学估计器、站点身份、hook 或原始 20/30 split。2026-10-19 用户进一步要求**本日完成有实际结果的全链路工作**，因此执行细节由下面 5.2 节覆盖。旧版中“任何计算方法前提不满足时整批阻塞”的理解不再适用；改为**验证问题 → 选择具有明确科学定义的可执行协议 → 运行 → 如实标注协议与适用范围**。

不允许为加速而虚构测量、擅自复用错位 tokens、掩盖模型身份或公布未获准公开的原始文本。允许降低数值诊断的过度保守门槛、使用另一种有定义的干预协议、缩小单批次规模、单独标注不同 K 的研究结果，以及先发布内部可用的研究报告。

## 5.2 2026-10-19 当日完成修订：避免不必要阻塞，产出真实可复用的实验

**本节为 GPU Agent 最新优先执行条款。先实施并完成当前 release，不新建第二批、不重新采旧 48 站点、不重复已经通过的原始测量。今日交付物为一个可以实际浏览/分析/干预/导出报告的内部 Research Lab release；公开 HF Space 和多用户实时 GPU 只在取得授权及认证后启用，否则作为明确的外部依赖继续关闭。不得将内部完成说成公开上线。**

### A. 当前记录、状态与不能误用的数字

- 固定 release：\`experiments/research-lab/7fb2be8f29a1-20261019T035727Z\`。当前 runner 位于 \`work-plan-05-gpu-handoff\` 分支提交 \`4ef3ba60afa4a87cf3ca4fcc0d59c1e0aaa88a93\`，执行表注明的 \`7fb2be8f...\` 只是 GPU 跑批前的源码身份。**创建一份实际执行源码快照（包含未提交变更），记录 runner SHA 与载入版本；不能只引用旧 main SHA。**
- GPU-00：500 条来源锁通过，200 discovery、300 evaluation，22,500 条 RSS proposals；审计哈希保留不变。原文 human_reviewed/license_reviewed 均 false，未经审批不能发布或自动关闭这项限制。
- GPU-01：RTX 5090 20 条 pilot 报告通过；GPU-02：48 站点各 500 条基准 prompt 的 capture 报告通过。这是**本轮真实采集覆盖**，不是与历史权重逐项一致的证明。
- GPU-03：2,400 个统计卡片（跨 48 站点，各取前 50）属于**无公开文本的 feature 统计索引**，不等于之前六个主站点各 50 张高质量可解释展示卡片；本日进一步准备六主站点精选 300 张，具体证据不足时明确空值。
- GPU-04：六主站点各已有一个原始 logits 干预对照。这是干预可用性测试，不等于六站点真实文本生成比较完成。
- GPU-05：**0 个有效 SAR 标准组**；不可写成 SAR 已完成。阻塞点位于 \`saebase/integrations/sar.py::_check_affine_decode\` 对被编辑 activation 的严格可加性前提，原论文统计函数尚未执行完。这是本日第一科学修复优先级。
- GPU-06：当前显示 **99,200 setting-level 名义输入调用量**，未必等于不同 site–prompt–variant 的去重次数。现有计数器每 setting 计 clean + 15，三 setting 会重复计算 clean；完整逻辑唯一输入的计划上限是 \`6×500×(1+3×15)=138,000\`，而不是每 setting clean 重复的 \`6×500×3×16=144,000\`。**在已存 JSONL 上重新核算有效 unique site/parent/variant，再发布覆盖率**。
- GPU-07：64 状态真实 Pythia L12 采集已完成；全宽字典的直接精确优化出现 192 GiB 内存申请。现有 55 个实际非零 support 特征及合成零列 parity 提供了**有依据的降内存方案**，不应被前面的大内存异常继续阻塞。
- GPU-08：特征存储已有 96,673 个不同特征、1,127,055 条记录；**1,500 条重算身份行与 500 条不一致行没有混入旧身份**。保留这份基线，不覆盖。GPU-09 公开认证与 HF 发布仍需授权。

### B. 今日执行原则：让有定义的结果先产出

1. **每个问题最多做一次小规模诊断；诊断只决定具体方法，不得成为无限循环的 stop gate。** 数值疑点在最多两个真实 prompt 上检查，并保留日志。对原数学估计器不做“静默放宽阈值”；不满足某个方法的前提时，切换到独立命名、科学意义明确的新测量协议，并在结果中记录，而不是整批停产。
2. **先交付小而完整的真实结果，再扩展规模。** 每类功能先形成至少一个可复现真实实例（实际模型/SAE/token/sample、原始 measurements、JSON 报告），随后继续自动批处理全部目标。规模大不再作为跳过理由；按 model/site/shard checkpoint 分批，不因为单批跑不完而扔掉已测有效结果。
3. **允许 scoped pass，不允许 fake pass。** 一站点只有 1 个编辑案例可标记 \`intervention_demo_passed\`，但不能标 \`500_group_population_passed\`；标准 K16 不满足时若 K8 真运行则标 \`SAR_K8_operational_passed\`，不能标 K16；RSS 变体未审完可以标 \`measured_unreviewed\`，不能标 \`meaning_preserving\`。
4. **允许新的 checkpoint 身份，不允许硬比较历史哈希。** 三个旧历史空哈希站点及 Gemma-2-2B IT L12 不一致站点均可用新的、实际加载后计算的 model/SAE SHA，写入 \`new_run_identity\` 命名空间继续实验和展示；不纳入历史 checkpoint match 统计，更不因此整批停工。
5. **立即归档，再继续重任务**。忽略目录不等于结果无效，但 \`experiments/\` 尚未进入 Git；先在受控存储做 snapshot、manifest、SHA-256、执行代码、权重/tokenizer/hash、完整 stdout/stderr、prompt 权限标记，并至少复制到第二个可靠存储位置。没有安全外部上传权限也可先做好可验证的本地 archive，保留原始文件，绝不能为了发包把未经许可的第三方文本公开到 GitHub Release/HF。
6. **GPU 只能用当前空闲资源**。旧 \`ensure_gpu_idle\` 固定等待五分钟只是工程调度偏好，不是科学前提。先查询 \`nvidia-smi\`，无其他任务、无显存争用即可运行；存在其他人的活动任务则不抢占、不杀进程。不要陷入无意义等待循环。

### C. GPU-05：立即解除 SAR 可加性阻塞，真实执行原始统计估计器

**关键区分**：当前 \`_check_affine_decode\` 验证的是“已选单特征编辑造成的 activation 变化能否线性相加”，而 SAR 最终估计的是所定义干预集合引起的标量下游响应函数 \`g(S)\` 的加性残差。严格线性解码不是“测量任何组合干预响应”的必然前提；但若当前 SAE 是非线性解码/归一化，**不能把它和原来的线性解码 feature-contribution 语义混为一谈**。

**新的执行策略必须按以下先后实现**：

- 对至少 2 个真实 prompt 收集 \`h\`、\`edit_features(single)\`、\`edit_features(pair)\`、\`edit_features(full)\`，打印绝对及相对 residual、dtype、layer_norm 状态、\`identity patch\` 一致性、重复测量稳定性。BF16/FP16 舍入引起的偏差单独记账；不要仅凭 1e-4 的 allclose 失败就得出“理论上解码器非线性”的结论。
- 新增显式 \`coalition_protocol = decoder_mediated_residual_preserving_v1\`，使用**现有原始 \`edit_features\`** 完成每个 coalition 的实际编辑，再经 \`adapter.patch_logits\` 获取固定 target-token logit 或 logit margin。保留 clean residual；每个 \`g(S)\` 是实际测量，不是从单特征线性预测补出来。验证真实 baseline/identity/replay、finite logits 与至少一个非恒定 coalition。可在本地 runner 中实现单独函数，或给 \`evaluate_sar\` 增加显式 \`require_affine_decode=False\`（默认继续 True）；**不得猴子补丁、吞掉任意异常、修改原 \`paper.sar\` / \`paper.coalitions\` 数学**。
- 将这种分析命名为 \`SAR (decoder-mediated intervention protocol)\`。它仍可调用原论文相同的 coalition sampler 与 SAR 原始回归/排序估计器，但**解释对象是这个新的组合干预响应函数**；不能声称它直接验证了原来附带严格线性解码前提的实验结论。
- 对满足至少 K16 个非零/可编辑特征的 prompt，保持原目标 \`K=16, M=384, fit=256, held=128\`。若 prompt 的真实活动 support 小于 16，**允许按实际 support 降级**：K8 配 \`M=128/fit=96/held=32\`；K4 配 \`M=64/fit=48/held=16\`；K2 配 \`M=32/fit=24/held=8\`。每种 K 均须满足各自二阶设计矩阵自由度、实际非零支持与 held-out 非泄漏检验。少于 2 或响应方差为零标 \`undefined\` 并继续下一条，**不得零填充假特征以凑 K16**。
- 先在一个主站点的一个 prompt 完整实际生成 masks、\`g\` 向量、\`g_full\`、fit/held、所有原始 metric 和 input/model hash；使用固定原始 SAR 实现做同一组 \`g\` 的独立重复计算 parity。测试通过立刻顺序运行三主站点所有有效组，以单 prompt（最多若干 prompt）为一个 durable checkpoint。即使部分组不支持 K16，也要将真正测到的 K8/K4/K2 统计单独输出，不能让 0 个结果持续存在。
- 输出 \`GPU-05/sar_v2/{site}/{sample_id}.json\`（或等价不可变分片），记录具体 K、M、fit、held、seed、feature IDs、mask、完整响应、期望条件、model/SAE revision、\`decoder_affinity_diagnostic\`、原始 estimator SHA256。K16/K8/K4/K2 各设 cohort 汇总，不在一张无条件平均表里合并。
- 原 runner 的 \`_rotation_control\` 当前仅将相同 16 个 feature IDs 循环移位，**这是置换而非旋转**，不能作为 matched rotation 证据。先停止给该结果贴 \`rotation\` 标签并移出主要测量的 denominator。今天优先完成真实 SAR；对照可改为预先固定种子、同 K 的随机活跃特征集合（标签 \`matched_random_feature_control_v1\`），或另行实现真正的正交基旋转并验证语义。两类对照都不得伪称 matched rotation。未完成的原计划 57,600 个 rotation 测量单独标为 \`rotation_pending\`，不妨碍本日有科学意义的 SAR 主结果。
- **通过条件**：至少一个模型有实际 \`g(S)\` + 原始 SAR 估计器输出；随后按 3×500 执行真实可测的 groups 并显示实际 K 分布、失败/undefined 计数。若最终没有任何非恒定 \`g\`，明确诊断 readout 与干预而不是再返回一个笼统的 “decoder not additive” 后结束。

### D. GPU-06：修复 token/BOS 错位及“99,200”重复计数，继续真实 RSS

- 当前 \`_anchor_token\` 使用 tokenizer 的 \`add_special_tokens=False\` offsets；但 \`models/hf_logits_backend.py::to_tokens\` 在存在 BOS 时主动加 BOS。不能直接把 0-based offset index 当作模型实际位置。为 **HF hook** 模型构造实际完整 \`model.to_tokens(text)\` token IDs，把每个 offset 的子串 tokens 与实际 tokens 做对照，确认 BOS/特殊 token 个数后转换。TransformerLens 同样以最终实际输入 token 序列为准，不硬编码全部模型“+1”。
- 在一个 clean 与一个 wrapper pair 上打印 \`actual tokens\`、\`offsets\`、\`clean_index\`、\`variant_index\`、对应字符区间及 token IDs。输入包裹改变子词分词边界时，允许把严格的同 token span 判定改为**内容锚点末端在真实模型序列中经验证的 readout token**，但必须记录 \`anchor_policy\`、\`position_mapping\` 和端点可能的上下文敏感性；对不可靠的样例只剔除该变体，不整批终止。
- **已有 99,200 名义调用记录不可丢**。从原 JSONL 按 \`(site_id, sample_id, setting)\` 去重，重算成功的 setting 数以及 \`(site_id, sample_id, variant_id)\` 的去重值。若失败来自 BOS，保留旧 measurement 为 \`position_unverified_legacy\`，单独记录，新正确协议重跑受影响的样例。不能把旧错位值默默改名为正确 RSS。
- 现有 resume 判断只要某组有**一个 setting** 就视为 done，应改为**逐 \`(site, sample, setting)\` 的 checkpoint**，并区分 \`completed/invalid/failed/unreviewed\`。一个 setting 成功不能阻止其他两个 setting 重跑。写新分片或按唯一键完成幂等合并，保留旧产物，不覆盖原 JSONL。
- 计划最大 **6×500×46 = 138,000** 个去重的逻辑 site-input 组合；三个 setting 共享一次 clean。已记的 \`99,200\` 数字按旧算法每 setting 计 16，**并不是独立输入数或通过审核的有效测量数**。重新生成 \`rss/coverage_v2.json\`，同时显示 \`n_parent_groups\`、\`settings_complete\`、\`selection_variants\`、\`heldout_variants\`、\`unique_site_input_occurrences\`、\`invalid\`、\`position_recomputed\`。
- 继续用固定原始 \`rss.py::rss\` 和 \`held_out_rate\`，selection/held-out 切分不变，抽样、feature 支持集和方向不从 held-out 反调参数。每个 completed setting 可报告 MeanRSS、support、held-out 方向持久；\`semantic_review=false\` 的结果应明确叫 \`RSS under proposed transformations (not verified meaning-preserving)\`，可内部展示探索结果，不称任务不变性或安全风险。
- **通过条件**：至少六主站点各形成一个完成 5+10 变体、位置核验和原始 RSS 计算的案例；随后续跑原先缺失的 setting 及所有可验证样例。对真实不支持的变体给出稳定 reason，不允许人为造数使覆盖率达到 100%。

### E. GPU-07 / 身份 / Feature Explorer：用已有可信结果继续，不再让附加验证阻塞全链路

- Localize 已有完整 64-state real input codes 与 55 列观察到的非零 support。对**这 64 个状态上严格全零**的字典列，去掉它们不会改变该网格的非零方差贡献；已有 synthetic zero-column parity 应与真实输入的零列检查一起保存。用 55 列 support 直接完成可重算 exact scope curve、预算和 assignment，**不再试图分配完整 192 GiB 稠密中间矩阵**。结果命名 \`real_model_scope_on_observed_support\`，不夸大为整个自然语言分布的行为解释。
- 对 3 个历史无 hash 和 Gemma-2-2B IT L12 历史 hash 不符，用实际加载权重重新登记 \`model_revision + tokenizer_revision + sae_revision + loaded_state_sha256\`，以**新 run 的独立 identity**用于后续分析；历史 identity 对照仍标 \`not_matched\`。不得以四站点不吻合为由删除已成功采集的数据，也不把它们插进旧 hash 的索引。
- GPU-03 从现有全 48 站点 feature 统计里抽出六个主站点的 300 张卡，填实际 \`sample_id\`、feature ID、激活大小、选中 token IDs/位置、可追踪 private sample reference、clean/edited 关联。文本无对外授权，页面显示 \`text unavailable for public release\`，**内部受控访问**可以展示按权限获得的原文；没有权限就不强行发到 HF。
- GPU-04 已完成六站点的 logits/feature-operator 干预；本日再使用至少**一个真实可用模型**完成 baseline 和 edited 的文本生成对比（固定 sampling seed、max_new_tokens、输入和 patch timing）。如果单特征 steering 生成接口做不到稳定拼接，先给本地 operator 工作台显示真实 next-token probability/top-k 与实际 rollout，再明确 \`full_steered_generation_pending\`；不能拿两组模拟文本充当真实生成。
- GPU-08 原有 feature SQLite/JSON store 保持不可变，在新版本/新命名空间纳入拥有**独立、实际哈希**的站点观测（可以同时浏览，不可混作历史 identity）。不重复统计同一 sample/site/token/feature，不把 64 grid states 混入自然文本组。

### F. 当日交付顺序与每项成功定义（必须执行，不要停在文档）

| 优先 | 执行动作 | 最小真实完成标志 | 继续批量的动作 |
|---|---|---|---|
| 0 | **冻结当前 GPU 实验原始证据** | 完整当前 release 的私有 archive、manifest、文件 SHA、执行源码 commit/dirty 状态及第二份备份 | 每项实验完成即增量归档 |
| 1 | **解除 SAR 整批 0 的阻塞** | 1 个真实 \`g(S)\`、K/M/fit/held 与原论文 SAR metric，标明 decoder-mediated protocol | 三主站点逐 prompt 完成；少支持组降 K 后单列 |
| 2 | **RSS token 对齐与恢复续跑** | 6 主站点各 1 个合法 5/10 setting 的 real result，BOS/offset parity 通过 | 重算错位、补漏 setting，输出 v2 unique 计数 |
| 3 | **Localize 已有 55 列曲线归档** | 64-state real scope curve 可重算，support/zero-column parity 通过 | 不扩大不必要的全宽内存申请 |
| 4 | **特征/权重身份与干预页面** | 4 个历史差异站点独立 identity；6 主站点可浏览；至少 1 个真实 edited generation/sample 结果 | 完善 300 张精确证据卡和六站点实例 |
| 5 | **内部研究发布包** | 可在本地 CPU Research Lab 打开来源清楚的 Results/Features/Intervene/Analyze/Reports；原始 run manifest 与 artifact 校验通过 | 继续覆盖剩余 sites/groups、更新总表 |
| 6 | **GitHub 交接** | 只提交代码、测试、去敏汇总、Work Plan 05 与安全 manifest；建 PR 并跑 CI | 准备后续 HF/正式在线服务权限 |

**如果单一模型/样例出现非实质阻塞，就记录并继续其他有效样例；如果重计算可能很贵，用 shard、checkpoint、resume，不准以“超过一小时”直接跳过。不得因无法立刻公开 Space 而阻塞已经能运行的本地平台与内部报告。**

### G. 正式停止条件（只用于保护数据真实性、财产和安全）

**必须停止该项或受影响 shard**：来源锁或 sample 内容不一致；模型/SAE 实际加载身份未知；token 对齐已证明错误而不能修正；干预 identity patch 不成立；实验中断导致结果不可区分重试；fit/held-out 数据泄漏；产生 NaN 仍强行写有限值；代码执行来源不可信；超出用户批准 GPU/存储预算；磁盘不足以保证原始结果持久化；公开服务缺认证与隐私保护。

**不需要整批停止**：historical hash 不一致但新 loaded SHA 可确切登记；某些 feature 没有 K16 support 但可做新 K8/K4 方案；某种变体无法对齐但其他变体合法；原非必须的 rotation control 未完成；没有 public HF 写权限但本地和私有研究报告可完成；全字典稠密求解内存过大但零列等价的 support 版本可算。

**最终对外声明**：报告完整说明每个实验的 \`protocol_id\`、真实完成组数、适用范围和局限。当前目标是**让平台今天拿到真实、有用、可复现的研究数据和端到端工作流**，不是保证所有历史最强协议在今天达到虚假的 100% 覆盖率。

## 6. 交给 GPU Agent 的填报模板

每个 GPU-XX 工作包至少填：

| 字段 | 必填内容 |
|---|---|
| task_id | GPU-00 … GPU-09 |
| status | not_started / blocked / running / passed / failed / partial |
| commit | SAEBase + SAR/RSS/Localize 原始固定源码 commit |
| model_sae_site | model revision, SAE revision, hook, layer, dtype |
| input_release | source_lock manifest SHA 与真实 input ids / split |
| population | 实际 base groups、selected tokens、variations / masks；失败数 |
| commands | 从干净环境执行的完整命令 |
| resources | GPU 类型、VRAM peak、RAM peak、disk bytes、elapsed time、成本 |
| outputs | 原始 artifact path、manifest、whole-file SHA-256、可下载位置 |
| scientific_checks | patch identity parity、原始估计量 parity、independence、CI/undefined |
| decisions | 是否通过进入下一工作包；缺失项和下一位执行者动作 |

严禁在根目录塞散落文件，建议所有新批次位于 runs/research-lab/<release_id>/ 对应工作包目录。任务完成后提供最短可重跑命令和成果哈希，其他 agent 才可接手。

### 本次执行填报

下表只记录这次运行已经写出的产物。没有证据的工作包保持 not_started、partial、failed 或 blocked。

| task_id | status | commit | model_sae_site | input_release | population | commands | resources | outputs | scientific_checks | decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU-00 | passed | 7fb2be8f29a13875c4d60be645771f992dc8c10d | 48-site manifest audited; weights hashed only after load | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | sources=10 groups=500 discovery=200 evaluation=300 variants=22500 gpu_measurements=0 | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run cpu | CPU audit; GPU not required | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-00 | source lock fingerprint, 10x50 split, 22500 RSS proposal hashes, review flags false | continue only if the frozen lock matches |
| GPU-01 | passed | 7fb2be8f29a13875c4d60be645771f992dc8c10d | pythia410m:pythia410m_mlp_topk:L12 then qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14 | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | 20 base groups = 2 per source; variants are not extra samples | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run gpu01 | {"batch_note": "Verified ModelAdapter captures one prompt. A larger batch was not substituted without a parity check.", "batch_size": 1, "go": true, "gpu": "NVIDIA GeForce RTX 5090", "mean_capture_seconds": [0.017654294474050404, 0.01630646684207022], "quantization": "not_used"} | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-01 | identity patch, identity operator, ablate, scale 0.5, full reconstruction, one token position | go |
| GPU-02 | passed | 7fb2be8f29a13875c4d60be645771f992dc8c10d | ["pythia410m:pythia410m_mlp_topk:L4", "pythia410m:pythia410m_mlp_topk:L12", "pythia410m:pythia410m_mlp_topk:L20", "qwen35_2b:qwen35_2b_resid_w32k:L6", "qwen35_2b:qwen35_2b_resid_w32k:L12", "qwen35_2b:qwen35_2b_resid_w32k:L15", "qwen35_2b:qwen35_2b_resid_w32k:L20", "qwen3_1_7b:qwen3_1_7b_ak_resid_l14_65k:L14", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L7", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L18", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L23", "r1_distill_qwen15b:r1_distill_qwen15b_resid_l20:L20", "gemma1b:gemma1b_scope_resid:L7", "gemma1b:gemma1b_scope_resid:L13", "gemma1b:gemma1b_scope_resid:L17", "gemma270m:gemma270m_scope_resid:L5", "gemma270m:gemma270m_scope_resid:L9", "gemma270m:gemma270m_scope_resid:L12", "gemma2_2b_base:gemma2_2b_pt_l25_canonical:L13", "gemma2_2b_base:gemma2_2b_pt_l25_canonical:L25", "gemma2_2b_it:gemma2_2b_it_matryoshka_l12_32k:L12", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L9", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L20", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L31", "gemma3_4b_it:gemma3_4b_it_scope_resid:L9", "gemma3_4b_it:gemma3_4b_it_scope_resid:L17", "gemma3_4b_it:gemma3_4b_it_scope_resid:L22", "gemma3_4b_it:gemma3_4b_it_scope_resid:L29", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L9", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L17", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L29", "llama31_8b:llama31_8b_lxr_8x:L8", "llama31_8b:llama31_8b_lxr_8x:L16", "llama31_8b:llama31_8b_lxr_8x:L21", "llama31_8b:llama31_8b_lxr_8x:L27", "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16", "qwen35_4b:qwen35_4b_matryoshka_l15_65k:L15", "qwen35_9b:qwen35_9b_resid_w64k:L8", "qwen35_9b:qwen35_9b_resid_w64k:L16", "qwen35_9b:qwen35_9b_resid_w64k:L21", "qwen35_9b:qwen35_9b_resid_w64k:L27", "qwen3_8b:qwen3_8b_resid_w64k:L9", "qwen3_8b:qwen3_8b_resid_w64k:L18", "qwen3_8b:qwen3_8b_resid_w64k:L23", "qwen3_8b:qwen3_8b_resid_w64k:L31", "r1_distill_llama8b:r1_distill_llama8b_llamascope_l23:L23"] | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | {"sites_passed": 48, "denominator_sites": 48} | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run gpu02 | null | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-02 | new captures are not Stage D 64-group results; failures stay in the denominator | 48/48 sites have 500 captured groups. Failures and gated download errors stay in the denominator. These captures are not Stage D 64-group results. Recomputed hashes on the three historically empty sites are not historical matches. gemma2_2b_it L12 loaded hash does not match its non-empty historical hash and is not a historical identity. |
| GPU-03 | partial | 7fb2be8f29a13875c4d60be645771f992dc8c10d | ["gemma1b:gemma1b_scope_resid:L13", "gemma1b:gemma1b_scope_resid:L17", "gemma1b:gemma1b_scope_resid:L7", "gemma270m:gemma270m_scope_resid:L12", "gemma270m:gemma270m_scope_resid:L5", "gemma270m:gemma270m_scope_resid:L9", "gemma2_2b_base:gemma2_2b_pt_l25_canonical:L13", "gemma2_2b_base:gemma2_2b_pt_l25_canonical:L25", "gemma2_2b_it:gemma2_2b_it_matryoshka_l12_32k:L12", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L20", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L31", "gemma2_9b_pt:gemma2_9b_pt_scope_resid:L9", "gemma3_4b_it:gemma3_4b_it_scope_resid:L17", "gemma3_4b_it:gemma3_4b_it_scope_resid:L22", "gemma3_4b_it:gemma3_4b_it_scope_resid:L29", "gemma3_4b_it:gemma3_4b_it_scope_resid:L9", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L17", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L29", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L9", "llama31_8b:llama31_8b_lxr_8x:L16", "llama31_8b:llama31_8b_lxr_8x:L21", "llama31_8b:llama31_8b_lxr_8x:L27", "llama31_8b:llama31_8b_lxr_8x:L8", "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16", "pythia410m:pythia410m_mlp_topk:L12", "pythia410m:pythia410m_mlp_topk:L20", "pythia410m:pythia410m_mlp_topk:L4", "qwen35_2b:qwen35_2b_resid_w32k:L12", "qwen35_2b:qwen35_2b_resid_w32k:L15", "qwen35_2b:qwen35_2b_resid_w32k:L20", "qwen35_2b:qwen35_2b_resid_w32k:L6", "qwen35_4b:qwen35_4b_matryoshka_l15_65k:L15", "qwen35_9b:qwen35_9b_resid_w64k:L16", "qwen35_9b:qwen35_9b_resid_w64k:L21", "qwen35_9b:qwen35_9b_resid_w64k:L27", "qwen35_9b:qwen35_9b_resid_w64k:L8", "qwen3_1_7b:qwen3_1_7b_ak_resid_l14_65k:L14", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L18", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L23", "qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L7", "qwen3_8b:qwen3_8b_resid_w64k:L18", "qwen3_8b:qwen3_8b_resid_w64k:L23", "qwen3_8b:qwen3_8b_resid_w64k:L31", "qwen3_8b:qwen3_8b_resid_w64k:L9", "r1_distill_llama8b:r1_distill_llama8b_llamascope_l23:L23", "r1_distill_qwen15b:r1_distill_qwen15b_resid_l20:L20"] | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | cards=2400 target=300 | included in run all | CPU index over saved codes | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-03/feature_cards.json | no invented explanations or dead-feature labels; text hidden until review | cards=2400 from sites with real codes; text hidden; unobserved features are not called dead |
| GPU-04 | passed | 7fb2be8f29a13875c4d60be645771f992dc8c10d | ["qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14", "gemma1b:gemma1b_scope_resid:L13", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22", "gemma3_4b_it:gemma3_4b_it_scope_resid:L22", "llama31_8b:llama31_8b_lxr_8x:L21", "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16"] | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | one evaluation group per loaded deep site; not a 500-group intervention study | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume | one resident model at a time; no quantization | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-04 | identity, ablate, scale, clamp, add, coalition, full reconstruction; single token position | 6/6 deep sites have a real clean/edited pair. Unloaded sites were not filled from another model. |
| GPU-05 | partial | 7fb2be8f29a13875c4d60be645771f992dc8c10d | ["qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22", "llama31_8b:llama31_8b_lxr_8x:L21"] | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | {"main_groups": 0, "planned": 633600, "rotation_groups": 0} | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume | sequential full-prefix patch; no quantization | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-05 | original paper.coalitions and paper.sar; K stays 16; short support is undefined rather than relabeled | SAE decoder is not additive, so coalition groups measured stay 0. Rows with fewer than 16 active features stay undefined and are not labeled K16. Loaded sites are not marked passed. |
| GPU-06 | partial | 7fb2be8f29a13875c4d60be645771f992dc8c10d | ["qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14", "gemma1b:gemma1b_scope_resid:L13", "gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22", "gemma3_4b_it:gemma3_4b_it_scope_resid:L22", "llama31_8b:llama31_8b_lxr_8x:L21", "mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16"] | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | {"occurrences": 99200, "planned": 138000} | python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run resume | hook capture; full-prefix; one model resident | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-06 | original rss() and held_out_rate(); empty support is undefined | Tokenizer-aligned settings were measured. semantic_review stays false. WikiText and TinyStories are not marked meaning-preserving. HumanEval text was not executed. |
| GPU-07 | partial | 7fb2be8f29a13875c4d60be645771f992dc8c10d | synthetic control plus pythia410m L12 when the pilot passed | controlled six-factor grids, not the 500 natural prompts | 64 states, one population | included in run all | 64 short authored prompts on the small site | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-07 | {"real_model_capture": {"failed": [], "independent_natural_prompts": false, "loaded_model_sha256": "f7127cb94a25b63c030d2dd6b2b26c8dd7814eae34ddda2269b068066da22244", "loaded_sae_sha256": "0792d98e4129cadcd2481ccd3c11109d9ecde46cde2e7f0da597ea2cc4b0f6a8", "measured": 64, "n_states_expected": 64, "site_id": "pythia410m:pythia410m_mlp_topk:L12", "status": "passed"}, "real_model_curve": {"error": "MemoryError: Unable to allocate 192. GiB for an array with shape (65517, 393103) and data type int64", "status": "partial"}, "real_model_curve_support": {"budget_at_90pct": 132, "n_features_full_dictionary_not_materialized": true, "n_features_in_support": 55, "site_id": "pythia410m:pythia410m_mlp_topk:L12", "source_type": "real_checkpoint", "states_are_one_controlled_population": true, "status": "passed", "total_variance": 0.279009701245975, "zero_columns_removed_after_synthetic_parity": true}, "synthetic_curve": {"budget": 9, "not_a_learned_sae_result": true, "source_type": "synthetic_logic_not_real_sae", "total_variance": 1.875}, "synthetic_zero_column_parity": {"assignment_match_on_nonzero_columns": true, "budget_full": 9, "budget_match": true, "budget_support": 9, "not_a_learned_sae_result": true, "zero_columns_dropped": 3}} | synthetic curve is not a learned SAE result |
| GPU-08 | partial | 7fb2be8f29a13875c4d60be645771f992dc8c10d | only rows whose loaded hash matches the historical hash | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | {"existing_store": "feature_store_v3", "feature_count": 96673, "immutable_store_not_overwritten": true, "observation_count": 1127055, "public_text_reviewed": false, "recomputed_identity_rows_not_inserted": 1500, "rejected_or_unmatched_rows": 500, "status": "partial"} | included in run all | CPU | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-08 | non-finite activations rejected; unmatched hashes not inserted | not a complete research-lab-release-v1 |
| GPU-09 | blocked | 7fb2be8f29a13875c4d60be645771f992dc8c10d | no public GPU worker | abfe1fb6a72d52a3c97bc63e5c50b9a61e09d84788f95d5bd8ea3db840d084f3 | queue negative checks only | included in run all | CPU | /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/GPU-09/live_service_acceptance.json | {"cpu_queue_checks": {"expired_lease_not_marked_success": true, "idempotent_enqueue": true, "owner_cancel": true, "post_api_jobs_closed": true, "stale_attempt_rejected": true, "stranger_cannot_read_status": true, "two_workers_receive_distinct_jobs": true}, "negative_checks": {"hostile_fields_rejected": true, "no_file_written_outside_release": true, "status_hides_script": true, "status_hides_traversal_path": true, "ttl_reap_does_not_mark_success": true}} | HF Space, three external users, and public online GPU remain blocked |

## 7. CPU 阶段剩余事项：不是 GPU 算法问题

下列工作受**外部授权/人工审查**约束，而非缺少 GPU：

1. 每条真实第三方文本的人工内容审核、任务渲染核对、许可证与归属证明。当前公开仓库只存来源锁与 SHA，不存未获准再分发的原文。
2. 逐模型 tokenizer/checkpoint 下载可能涉及 gated 访问权限。拿到对应权限后，token 数量、位置和模板检查在 CPU 就能完成；禁止假定所有模型使用相同 tokenizer。
3. 向 OpenSAE 组织 Hugging Face Space 发布 CPU Web 应用，需要目标 Space 的写权限及明确备份/回滚。本轮没有这类授权连接，仓库中的 Docker 仅为真实可构建的部署配方，不可声称组织站点已更新。
4. 平台公共在线 GPU 计算涉及身份、私人数据、持续资源配额与主机凭据；无法通过 GitHub CI 绿色代替正式安全验收。

**验收口径**：CPU 程序、完整候选群体、RSS proposals、自动检测、来源锁与已有界面/队列/报告协议都能在无 GPU 下生成和测试。任何尚未审查或必须使用真实 pretrained checkpoints 才能证实的结果保持 blocked；不因追求“全绿”而制造假完成状态。

## 8. GPU Agent 的第一条命令和停止条件

执行者先检查本计划和固定 source lock；在有 HF 读取权限的 CPU/宿主环境复放第 3 节审计。**验证成功再租/启动 GPU 计算**。工作单元从 GPU-00 和 GPU-01 开始，先小样本，测实际吞吐后再批准全量。

后续停止/降级执行规则以第 5.2.G 节为准。已确认的来源锁变化、实际权重身份未知、无法修复的 tokenizer 错位、sample split 泄漏、SAR/RSS 数学估计器 parity 失败、存储/成本超限或未经认证的公开服务，停止对应受影响任务；非必要的线性解码假设、旧历史哈希不一致、部分输入不合法、部分 feature 支持不足等，**选择明示且可验证的替代实验协议继续推进其他合法数据**，不得阻塞整批。

完成标准不是“跑满作业数量”，而是用户可以在 Research Lab 中可靠地**找到特征、查看真实输入、实施受控干预、检验可解释性的限制，并把可追溯的证据分享与复现**。所有结果均须注明模型、数据和实验范围，不能超出科学证据作宣传。
