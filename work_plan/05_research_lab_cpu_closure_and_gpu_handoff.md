# Work Plan 05 — Research Lab CPU 阶段收尾、500 条候选语料锁定与 GPU Agent 执行清单

创建日期：2026-10-18  
状态：CPU 阶段代码、候选数据源锁及 22,500 个 RSS 变体哈希已合并至 main（PR #38、#40、#42）；固定上游版本的 CPU 来源锁复验和正式 CI 通过。原始文本的人工内容/版权审批未完成，GPU 实验未启动。  
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

## 7. CPU 阶段剩余事项：不是 GPU 算法问题

下列工作受**外部授权/人工审查**约束，而非缺少 GPU：

1. 每条真实第三方文本的人工内容审核、任务渲染核对、许可证与归属证明。当前公开仓库只存来源锁与 SHA，不存未获准再分发的原文。
2. 逐模型 tokenizer/checkpoint 下载可能涉及 gated 访问权限。拿到对应权限后，token 数量、位置和模板检查在 CPU 就能完成；禁止假定所有模型使用相同 tokenizer。
3. 向 OpenSAE 组织 Hugging Face Space 发布 CPU Web 应用，需要目标 Space 的写权限及明确备份/回滚。本轮没有这类授权连接，仓库中的 Docker 仅为真实可构建的部署配方，不可声称组织站点已更新。
4. 平台公共在线 GPU 计算涉及身份、私人数据、持续资源配额与主机凭据；无法通过 GitHub CI 绿色代替正式安全验收。

**验收口径**：CPU 程序、完整候选群体、RSS proposals、自动检测、来源锁与已有界面/队列/报告协议都能在无 GPU 下生成和测试。任何尚未审查或必须使用真实 pretrained checkpoints 才能证实的结果保持 blocked；不因追求“全绿”而制造假完成状态。

## 8. GPU Agent 的第一条命令和停止条件

执行者先检查本计划和固定 source lock；在有 HF 读取权限的 CPU/宿主环境复放第 3 节审计。**验证成功再租/启动 GPU 计算**。工作单元从 GPU-00 和 GPU-01 开始，先小样本，测实际吞吐后再批准全量。

任何以下情况立即停止新增高成本任务：source lock 不一致、checkpoint/hash/hook 不匹配、tokenizer 导致错位、source-group 泄漏、原始 SAR/RSS 估计量 parity 不一致、磁盘接近上限、超过明确预算、产生未注明来源的合成结果、或面向公开用户的 GPU 端点未经认证。

完成标准不是“跑满作业数量”，而是用户可以在 Research Lab 中可靠地**找到特征、查看真实输入、实施受控干预、检验可解释性的限制，并把可追溯的证据分享与复现**。所有结果均须注明模型、数据和实验范围，不能超出科学证据作宣传。
