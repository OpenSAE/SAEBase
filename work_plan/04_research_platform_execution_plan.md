# OpenSAE Research Lab 集中开发执行方案

日期：2026-10-18  
状态：执行计划，尚未实施，未启动 GPU 任务。  
归档位置：`SAEBase/work_plan/04_research_platform_execution_plan.md`。  
目标：在保留 SAEBase 科学协议与既有证据的基础上，交付可浏览、可干预、可分析、可分享、可复现的研究平台。

## 0. 本轮固定目标

五项能力全部属于本轮交付范围，不把后两项留作没有验收条件的未来计划。

1. 可探索的实验结果页面。
2. Feature Explorer，真实激活样例、输入搜索与特征详情。
3. 可视化特征干预，包括已计算实验回放和至少一个真实模型的在线作业。
4. SAR 与 RSS 分析工具，另有适用范围明确的 Input Localization 页面。
5. 每次实验可分享、下载、按固定配置重跑并核验来源。

用户指定的基础数据规模是 **10 个来源数据集 × 每个 50 条基础 prompt，共 500 条**。所有 prompt 均有原始来源。变体、组合干预、多个 token 位置及重复运行不算新增独立 prompt。

首版定位为有清楚支持范围的公开研究平台，而非世界模型能力排行榜、完整特征百科或覆盖所有 SAE 的全自动解释器。完整功能与逐项标注的模型覆盖同时成立。

## 1. 目前可复用的基础及必须新增的工作

### 1.1 已核对的基础

截至本次读取，SAEBase 已有模型/SAE 加载、激活与 code 缓存、七种干预条件、Benchmark、分组可靠性分析、SAR/RSS/Localize 集成、外部扩展以及便携报告。当前验证矩阵是 18 个配置、48 个站点，历史实验每站点使用 64 个受控提示组。[S1][S2]

这些历史记录不能改名冒充本轮 500-prompt 实验，也不应重新覆盖。新的数据、输入格式、token 选择、批处理和读出必须形成新协议和新记录。

本轮还确认两个需要明确处理的限制：[S3]

- 现有批量干预每个 measurement 只选择一个 patch token 位置。首版界面按这一能力设计，不承诺任意多 token、多层联合 patch。
- 现有生成是 full-prefix reference 路径，尚不能据此宣传 KV-cache 加速或高吞吐实时服务。

原始 `sar.py` 默认 `K=16, M=384, n_fit=256`，256 个拟合样本拟合 137 个二阶参数；余下 128 个用于 held-out 测量。返回字典主要是摘要指标，不包含可直接绘制热图的二阶系数。[S4]

### 1.2 新增内容

需要新增数据适配器、真实样本索引、token 级浏览数据、可比性规则、在线作业服务、网站、特征解释元数据、分享发布流程及端到端测试。已有 Python 函数不等于已有在线产品。

本次无法重新取得 OpenSAE HF Space 的公开页面内容。因此部署工作必须先读取实际 Space 源码、SDK 类型、环境与持久化状态；不把之前聊天中的页面状态当作本轮已核实事实。迁移前导出旧 Space 和 Dataset 版本并测试回滚。

### 1.3 禁止破坏的边界

- 不改动历史 48-site 文件、原始归档、已发布 tag、原始论文估计量定义。
- 不把 Base 与 Instruct、不同 hook、不同归一化或不同权重视作相同 checkpoint。
- 不因后台库替换而默默改变 patch、生成、读出或缓存身份。
- 前端显示和统计审计不重新实现一套数学定义。
- 不上传第三方权重，不暴露访问 token，不将自定义用户文本默认公开。

## 2. 用户最终看到的产品

主导航建议为 `Explore / Intervene / Analyze / Results / Datasets`。这是用户任务，而不是内部开发编号。

首页用一条真实使用路径说明产品：**选一段文字 → 查看活跃特征 → 改变特征 → 比较输出 → 检查和分享证据**。首屏给可点击示例，不以哈希、CI、模块数量和密集解释文字作为主视觉。

### 2.1 Results：可探索的结果

选择 dataset、model、SAE、layer、protocol、condition、split 后浏览指标和样本。

每张图显示有效样本数、单位、数据版本、误差范围及分母。支持进入单条样本和原始报告。历史验证、本轮语料、用户自定义实验分开显示。

比较必须通过 `comparison_key`：任务读出、数据版本、输入渲染、token/patch 策略和指标定义一致。跨模型的 raw activation 幅度不作直接质量排名；不同模型的同名 token 下标不是相同特征。只在相同模型权重、同一 hook/layer 上做受控双 SAE 比较。

### 2.2 Explore：特征浏览和搜索

至少提供三种入口：按 ID 搜索；按已有解释/人工标签搜索；从已有或新输入找到活跃特征。

特征详情包括完整 checkpoint/site 身份、真实文本片段及 token 高亮、激活频率与分布、最强样例、普通样例、未激活/弱激活对照、不同数据集的覆盖情况、相关干预和 SAR/RSS 报告。频率必须区分 token 频率与 prompt-group 频率。

元数据页可覆盖字典全部 ID，但未观察到的特征显示“在本语料中未观察到”，不叫作 dead feature，不生成虚构解释。每个 token 默认只显示 top-20，页面分页/按需加载。

首版精选目标是 6 个主站点各 50 张经过实际样例核查的卡片，共 300 张；核查的含义是确认样例、token、数值和来源正确，不是证明自然语言标签完全正确。每条标签有 `source`、`author_or_model`、`created_at`、`status`、`label_source_split` 和支持/反例。标签构建默认仅用 discovery；evaluation 样例可在冻结评估后作为有标记的展示证据，不再用于同一版本解释调优。

人工标签、外部导入描述、自动解释、实验结果是不同字段。自动解释必须保留生成器版本和输入示例；观察不足时允许不生成描述。解释不能根据 held-out 结果反复调优后还宣称该集独立。导入 Neuronpedia 描述前必须核对真实 model、SAE revision、hook、归一化与 feature ID；只匹配数字 ID 不可用。

### 2.3 Intervene：干预工作台

左右并排显示基线与干预结果。提供 feature ID 选择、消融、缩放、clamp/add、组合保留，以及独立的 full reconstruction 对照。所有算子直接调用 SAEBase。

三个明确的执行方式：

- **已有实验**：只读取真实存档，可查看已计算的强度档位，不对两个档位之间的生成文字或指标做插值。
- **在线计算**：用户提交参数，后端真实运行，显示排队、运行、失败和取消状态。
- **本地运行**：下载固定实验配置，用研究者自己的机器复现。

首版在线模型建议使用已有验证配置 `gemma3_4b_it` 的 L22 对应 SAE；是否上线取决于新部署的兼容性与负载验收，而非只看历史通过记录。其他模型保留预计算浏览和下载重跑，不假装全部在线。

初始在线约束建议：输入不超过 512 tokens、生成不超过 64 tokens、普通编辑不超过 8 个特征、一个 layer、一个当前 token patch 策略。高级 SAR 作业使用单独限额，不与普通编辑共享“8 特征”的语义。

生成对比既保存文本，也保存 seed、采样方法、停止原因、patch 时序、token 序列与运行环境。same seed 不等于输出路径完全对应；不能把一次生成变化说成稳定的任务改善。

GPU 不在线时仍能浏览全部公开实验，在线按钮说明当前不可提交，不用永久 loading 掩盖无算力。

### 2.4 Analyze：SAR、RSS、Input Localization

详见第 5 节。每个工具先要求有效实验设计，不提供对任意一条文字都声称能计算全部指标的按钮。

### 2.5 可分享与可复现的报告

每个报告包含输入、模型/SAE、被测位置、特征集合、条件、指标、误差、适用范围和可下载配置。公开结果有永久版本地址，复现是创建新运行并与旧结果比对，而不是覆盖旧文件。

分享按钮必须区分“私有链接”和“公开发布”。用户主动确认后才将自定义文本、输出、激活和标签公开。分享与删除行为写审计记录。

## 3. 数据集方案

### 3.1 首批 10 个来源及 50 条采样方案

下表的分层方式是本方案规定的抽样设计，不是数据集官方划分。最终 row ID、revision 和原始文件哈希必须在代码开发开始时冻结。

| # | 数据集 / 来源 | 首版来源 split | 主要用途 | 采样和读出要求 |
|---|---|---|---|---|
| 1 | WikiText-103 raw / `Salesforce/wikitext`, `wikitext-103-raw-v1` | train | 自然文本、实体和语言模式 | 按原始 article 分组，选 50 个不同文档，不抽空行或标题行；保留后续文本作 continuation 读出，不把目标拼进基础输入 |
| 2 | TinyStories / `roneneldan/TinyStories` | validation | 叙事、人物、简单事件 | 50 个不同故事；主题与长度分层；标注其合成语料属性 |
| 3 | SQuAD 1.1 / 官方发布或对应固定 HF 镜像 | train | 上下文阅读、实体与答案定位 | 50 个不同 source article，每篇一问；保留答案所在上下文，不根据模型表现选择窗口 |
| 4 | BoolQ / `google/boolq` | validation | 是非判断、否定表达 | 按 passage 去重，25 true / 25 false；yes/no 完整 tokenization 验证 |
| 5 | ARC-Challenge / `allenai/ai2_arc` | test | 科学知识、多项选择 | 按科目/长度分层，不按模型答对与否筛选；保存选项原文与标签映射 |
| 6 | HellaSwag / `Rowan/hellaswag` | validation | 事件、动作、情景延续 | 50 个不同来源场景，标签平衡；完整候选 continuation 评分与选项字母评分分开命名 |
| 7 | WinoGrande / `allenai/winogrande`, `winogrande_xl` | validation | 指代消解、上下文依赖 | 同源/最小变体合组，只取一个基础 prompt；不使用无标签 test split |
| 8 | GSM8K / `openai/gsm8k`, `main` | test | 算术、数值与条件关系 | 50 题，人工核对数字和最终答案；基础输入仅题目，不含参考解答 |
| 9 | HumanEval / `openai/openai_humaneval` | test | 代码、函数说明、语法 | 50 个独立 task_id，只使用 prompt，不泄露 tests/canonical_solution；在线默认不执行生成代码 |
| 10 | TruthfulQA / `truthfulqa/truthful_qa`, `multiple_choice` | validation | 常见误解与答案偏好 | 保持原 MC1/MC2 定义并独立命名；不把它当作完整安全性数据集 |

来源与任务说明见 [D1]–[D10]。本版以英文为主，因此不宣传跨语言普适性。

### 3.2 采样规则

使用固定 seed `20261018` 和固定来源版本，先形成候选池，再按预先规定的语言、长度、任务标签、文档身份去重和筛选。所有排除条目保留原因。模型输出、SAR、RSS 结果不得影响这 500 条的选取。

跨数据源也检查相同 Wikipedia article、passage 和近重复内容；必要时按冻结规则重选，不能让同一原始文档在两个来源被重复计为独立组。

原始文本保留不变。截取、包裹、渲染通过独立字段保存。默认目标长度 128–512 tokens，但不要为凑这个区间破坏问题；短问题允许保留。跨模型使用相同来源文本边界，不为某个模型单独裁掉答案相关上下文。超限例子按预先定义规则重选，不能运行后静默截断。

50 条按原始 group 分为 `discovery=20` 与 `evaluation=30`，全体共 200/300。source split 与这两个实验 split 是不同字段。WikiText/SQuAD/BoolQ 按来源文档或 passage 合组，不能把同段文本下多个问题当作独立数据。

每条基础 prompt 都接受人工内容核对。全部 row ID、标签、上下文、长度、截取和许可信息冻结后再开始 GPU 计算。

### 3.3 三种“独立”必须分开

- 不同基础 prompt groups：总体统计单位。
- SAR fit 与 held-out coalition：单条 prompt 内拟合/检验单位，不增加总体 N。
- RSS selection 与 held-out variants：单条 prompt 内选特征/验证方向的单位，不增加总体 N。

公开 evaluation 数据只能说明本次实验流程没有用它选特征，不能说明预训练没见过，也不能长期当作反复调参后的未见测试集。对后续正式论文需要新数据或另外预注册的确认实验。

### 3.4 许可与发布

建立每数据源独立的 `licenses/`、`source_manifest` 和 `NOTICE`，保留作者、源 revision、下载时间及逐行来源。不把所有文本重标为 MIT，也不把上游代码许可证当作文本数据许可。

本次核查发现来源标签不同：TinyStories 标为 CDLA-Sharing-1.0，BoolQ 为 CC BY-SA 3.0，ARC 卡片为 CC BY-SA 4.0，WinoGrande 官方说明 code Apache-2.0 / data CC-BY，GSM8K 与 HumanEval 卡片为 MIT，TruthfulQA 为 Apache-2.0。WikiText 的页面 metadata 与正文存在版本表述差异，需要发布前核对具体原始资源；SQuAD 对应下载版本也应单独核实。[D1]–[D10]

这些是来源核查记录，不是完整法律审查。发布关卡要求明确实际文件的显示、修改和再分发条件。未澄清时不能把明文推到公开 Hub；先完成定位和许可，不能用“只发 hash”伪装完整 Feature Explorer 数据已上线。

建议首版新建一个具有清楚说明的聚合数据资产 `OpenSAE/SAEBase-Prompts`，含 10 个 configs 或分目录，而不是为 50 条样本建立 10 个几乎空的仓库。名字为待创建方案。描述为来自十个公开来源的 curated subsets，不宣传十个原创数据集。既有 `SAEBase-Evaluation` 保留历史版本；新结果加明确版本或建立独立 `SAEBase-Experiments`，不覆盖旧记录。

## 4. 模型与计算覆盖

### 4.1 覆盖层级

| 层级 | 目标 | 范围 |
|---|---|---|
| 基础矩阵 | 48 个已有验证站点 × 500 prompts | 激活、稀疏 codes、真实浏览数据、基础重建与适用的对照读出 |
| 深度浏览与干预 | 6 个主站点 × 500 prompts | 完整特征索引、精选卡片、干预档位、RSS；有意义的任务结果展示 |
| 标准 SAR | 3 个主站点 × 500 prompts | K16、384 coalitions 的完整估计，失败/退化例子如实保存 |
| 在线服务 | 至少 1 个真实主模型 | 自定义输入、普通干预、受限 SAR/RSS 作业及完整报告 |
| 双 SAE 对照 | 至少 1 组同模型同站点的两个 checkpoint | 验证公平比较与 paired CI；新增计算量另列 |
| Localize | 两个新增完整 6-factor 受控 grid，每个 64 状态；另可复放原论文 reference grid | 独立于 500 条自然 prompt；至少一个真实小模型工作流 |

建议六个主站点使用已有矩阵精确 ID，不自造 hook：[S2]

- `qwen3_1_7b_base` / `qwen3_1_7b_base_resid_w32k` / L14。
- `gemma1b` / `gemma1b_scope_resid` / L13。
- `gemma3_4b_pt` / `gemma3_4b_pt_scope_resid` / L22。
- `gemma3_4b_it` / `gemma3_4b_it_scope_resid` / L22。
- `llama31_8b` / `llama31_8b_lxr_8x` / L21。
- `mistral7b_instruct` / `mistral7b_instruct_tc_l16_131k` / L16，必须使用登记的 MLP hook，不把它改成 residual hook。

SAR 先使用上述 Qwen 1.7B Base、Gemma 3 4B PT、Llama 3.1 8B 三站点。在线普通生成优先 Gemma 3 4B IT。小模型开发先用 Pythia-410M 或已验证轻量站点。

双 SAE 对照不能只凭模型名字相似挑 Qwen Base 与 Instruct。优先复用历史小模型同层双 checkpoint；在计划冻结时验证真实权重、hook、normalization，指定两个实际 SAE ID。若不在 48-site 主矩阵中，其观测量单列，不偷换 48 的计数。

### 4.2 为什么不直接对 48 站点全部做最重实验

标准 SAR 的逻辑组合测量数：

- 3 × 500 × 384 = 576,000。
- 48 × 500 × 384 = 9,216,000。

这还不包括基线、控制、重试和较长读出。功能可以完整，支持范围可以有层级；不能让界面上的“支持48个站点”被误解为每个站点都拥有所有深度实验。

### 4.3 计算组织

按 model revision 调度，尽量一次装载权重，完成其层位采集、编码与测量后再切换。可尝试同一前向采多个 hook，但作为新增优化验收，不假定当前 API 已支持。不同模型或不同 token/预处理身份不得共享 raw cache。

先做包含十个数据源、每源两条的 20-prompt pilot，在小模型与一个目标大模型各测一次。记录 tokens/sec、activation rows/sec、patch measurements/sec、峰值显存、host RAM、磁盘增长、编码吞吐和下载成本。再生成全任务实际预算。

未配置预算授权时，只允许 preflight 与离线 CPU 检查，不自动购买算力。GPU 机器应先备份和验证现有工作，不自动关停用户已有实例。

## 5. 科学协议与读出

### 5.1 基础表示与行为读出

所有数据均可做激活和重建分析，但不是所有数据都有同一种任务正确率。读出必须分三层：

1. **输出偏好诊断**：在 clean baseline 中固定 top-1/top-2 token 作为 target/reference，记录 logit margin。这适合统一的局部干预研究，不代表题目答对了。选择后不随干预重新选 token。跨模型 target 可能不同，不作为跨模型任务分数。
2. **任务读出**：多选题完整候选得分或经 tokenizer 验证的答案标签概率；SQuAD/GSM8K 的答案生成与官方风格评分单列。多 token 答案必须用完整 continuation 计分，不只读第一个 token。
3. **生成展示**：固定生成协议，展示真实文本，不把 64-token 预览当成完整 HumanEval pass@1 或 GSM8K 正式 benchmark。

在新 `SequenceReadout` 适配器中，明确候选 token 边界、sum/mean log-prob、teacher-forcing、prefix patch 位置和完整 forward 语义。答案计分使用的 teacher-forced tokens 不能混入“未见答案的基础 prompt 激活”记录。

首版 HumanEval 不在在线 API 内执行任意生成代码。需要通过率时仅用独立、禁网、无凭据、有限 CPU/内存/超时的离线执行环境，且使用其原始任务测试规则；普通 subprocess 不算安全隔离。[D9]

### 5.2 SAR

原始估计量和数值约定固定，默认 K16 / M384 / fit256 / held128 / p0.5。[S4]

从 clean activation 依据预注册策略选最多 16 个活跃贡献。少于 16 时记录 `actual_k`；不能补入虚假活跃特征，也不把不同 K 的分数混成无条件排名。保留残差，g_full 独立保存。报告 SAR1、SAR2、Delta2、rho2、held-out rank、ordering/sign error、regret 及有效方差。

用于图形展示的二阶系数需要新增诊断适配器，以原始 `design` 和相同 fit 样本求解；测试其预测与原始估计量完全对应。热图称“二阶拟合系数/组合响应”，不称已发现的完整电路。系数坐标编码与单位随图显示。

不可把 estimated SAR 强行裁剪到 [0,1]，不可把 SAR2 偶尔大于 SAR1、低方差、NaN 或负的估计收益改成漂亮的正数。退化和数值警告有独立状态。

对照包括 identity、full coalition、解析 affine/known-pair synthetic controls。另从每数据源按预先固定规则取 5 个基础 prompt，在三个 SAR 站点增加一个匹配 rotation 对照，共 10×5×3×384 = 57,600 次额外逻辑测量。总计划为 633,600 次 coalition 测量，不含其他前向和控制。该子集报告单列，不暗示全500条都已做 rotation。

单条 prompt 的 fit/held split 不能代替 20/30 个原始 prompt 的组级区分。默认只在 evaluation groups 上做总体推断，discovery 结果可浏览但标注不同用途。

### 5.3 RSS

首版三种明确记录的变换集合：局部格式、指令包装、中性上下文。每集合 5 个操作、每操作 3 个实例，固定一个用于 selection，另外两个用于 held-out，即 5+10。

这是本轮新语料上的变换协议，不冒充原RSS论文全部setting的直接复现；原始selector数学约定保持不变，protocol ID单独记录。

每基础 prompt 最多派生 45 个变体，加 clean 共 46 个输入。6 个主站点的逻辑输入量为 500×46×6 = 138,000；如果多个 hook 共用一个模型 forward，实际请求数可能不同。不能把这 138,000 个观测说成 138,000 条独立问题。

变换由 dataset-aware renderer 产生并保存 exact texts，不在每次点击时重新随机生成。数学不能随意改数字/运算，代码不能破坏缩进/标识符语义，阅读题不能删除证据。schema 检查、答案一致性规则和分层人工审核一起决定变体是否合法。对于无法验证的变体应标 invalid，补充相同预定义操作的有效实例；不得运行后因结果难看而换变体。

token 位置对齐用命名语义锚点及字符区间到 token 的映射，而非相同整数下标。共同答案边界与实际输入内容末端需由每个适配器定义；增加 suffix 后不能比较不同内容的激活。

复用原始 RSS selector，记录响应幅度、recurrence、方向一致性、support、MeanRSS 和 held-out persistence。具体阈值从固定原始实现/配置读取并保存，而不是前端另写常量。跨 SAE 幅度不直接混合。

首版默认不从 MeanRSS 推出“失败风险”。带冻结线性读出的 margin/erosion 与风险分析是单独协议，必须有独立校准、明确标签、有效正负例和 held-out 验证；这些条件未满足时显示不适用。不能拿 TruthfulQA 代替所有安全标签。

### 5.4 Input Localization

50 条任意自然语言 prompt 不构成完整的 2^n 因子 grid。新增两个6-factor受控设计，每个64个状态，作为附加资源，不算第11/12个自然语料来源。

新增的两个6-factor设计各有64状态，至少一个在真实轻量模型上用固定renderer收集，另一个可用已知机制的合成对照并明确标记。另设原论文reference codes复放入口，严格保留原始因子数与grid尺寸，不能把原始256状态随意截为64状态。记录所有因素、组合顺序、样本映射、token位置及code hashes；只对合法完整设计开放原始exact/paired estimator。

页面展示 budget-error curve、给定解释方差目标所需预算和可检验的 scope assignment。不称为任意文本自动词级定位。

grid 状态不是独立自然 prompt groups，不直接套用普通 bootstrap。有限 paired estimator 的随机种子变化单列为 Monte Carlo 敏感性，不包装成总体置信区间。[S5]

### 5.5 可靠性与小样本范围

按原始 prompt_group 重采样；同组 tokens、variants 和 coalitions 一起进出。跨数据集按固定来源分层，默认分数据源报告。只有单位、读出和协议相同，才显示来源等权的描述性宏平均。

双 SAE 使用相同组的 paired difference。预注册主要比较与探索性比较分开；大量特征/条件搜索不当作单个事先指定的显著性检验。图中清楚显示有效 N、缺失原因和协议。

50个样本对非常接近的性能差异识别能力有限。以 Bernoulli p=0.5 的正态近似为量级参考，n=50的95%半宽约13.9个百分点，n=30约17.9个百分点。这是规划算术，不是实际数据的置信区间；实际页面使用适合观测单位的方法。500条适合首版案例库和小规模研究，不等于全面性能认证。

## 6. 存储与性能预算

### 6.1 不把所有激活都变成密集浏览器 JSON

假定500 prompts、平均256 tokens、48 sites，则是6,144,000个site-token observations。

假设每站点平均隐藏宽度4096，FP32原始激活约100.66GB（93.75GiB），尚未计 variants、权重、临时副本和文件元数据。若保存131072维的密集SAE codes，仅这个基准规模就超过3TB。因此必须流式编码并稀疏保存。

假设平均L0=100，int32索引+float32数值的基本非零 payload 约4.915GB（不含 indptr、metadata 等）。给UI的top-20索引+数值约0.983GB，依旧需要分片，不能一页全部下载。这些都是假设推算，必须用 pilot 替换。

科学计算保存完整非零值；网页top-k只是显示索引，不能回流为SAR/RSS/reconstruction的完整输入。raw activation 采用有身份的分片；按模型完成编码、测量、双处校验归档后，才能按保留策略淘汰工作盘缓存。

RSS默认只持久化对齐目标位置的完整 codes，并按需要保留局部展示片段，不为每个变体重复保存所有层所有token的密集激活。需要完整原始数组的实验在计划中单独声明。

### 6.2 硬件建议

离线采集优先一个48GB GPU，降低多种checkpoint的装载压力；32GB设备可在pilot证明真实模型、SAE、dtype和microbatch可运行后选用，不因显存不足默认量化并沿用原结论。

工作盘先按约500GB规划，并配置外部持久化；实际要求取决于顺序下载/清理策略、保留模型数和变体档位，不承诺500GB对所有方案必然够用。所有模型同时常驻不是首版要求。

租赁选择按实测“完成一次有效测量的成本”，不是只比较每小时报价。具体价格和可用实例在执行当天核对，本方案不虚构租金或完成时间。

### 6.3 作业预算

每个批次给出 `estimated_forward_tokens`、`max_measurements`、`max_new_tokens`、`max_wall_seconds`、`max_output_bytes`、worker类别和成本上限。长SAR任务拆成可恢复小单元，不直接绕过当前默认measurement预算。[S3]

设两类预算：离线发布批次与在线用户请求。任何额度增加必须显式批准；不能因“平台成熟化”自动租更多GPU。报告保存实际消耗，用于以后扩容。

## 7. 建议架构

### 7.1 代码组织

首版留在SAEBase主仓库，网站是隔离应用，避免一开始拆出大量需要同步发布的新repo。

```text
SAEBase/
  saebase/                     # 保留公共科研库；兼容现有接口
  apps/research-lab/
    frontend/                  # React + TypeScript；独立依赖锁
    backend/                   # FastAPI；独立 Python 依赖
    worker/                    # 只调用固定版本 SAEBase
    schemas/                   # 导出的 OpenAPI/JSON Schema
  conf/platform/
    datasets_v1.yaml
    coverage_v1.yaml
    protocols_v1.yaml
    budgets_v1.yaml
  scripts/platform/
    build_prompt_release.py
    build_feature_index.py
    publish_artifacts.py
    verify_platform_release.py
  tests/platform/
  docs/research-lab/
  work_plan/platform_v1/
```

这是待新增结构，不代表上述文件现在已存在。实际实现可调整文件名，但模块责任和验收条件不可丢失。

### 7.2 服务分工

- 前端：交互、可视化和轻量过滤，不持有Hub/GPU凭据，不计算一套独立指标。
- CPU API：数据查询、鉴权、参数校验、作业创建、报告发布和下载。
- Postgres：版本与元数据、用户权限、队列和状态。首版可用数据库lease实现单GPU worker队列，不必同时引入Redis、Kubernetes等额外系统。
- 对象存储：不可变的公开实验与私有作业产物。公开release可发布到HF Dataset；在线任务产物用持久bucket/object storage，发布后再进入公开数据资产。
- GPU worker：固定模型/SAE白名单、作业隔离、取消、重试、原子提交；不把推理进程放入Web进程。

HF Space可部署CPU网站/API或作为入口，GPU worker独立。HF官方文档说明Docker Space本地写入可能随重启丢失，必须使用持久化方案；在线队列和报告不能只存在Space临时磁盘。[I1]

### 7.3 核心对象

`DatasetRelease`、`PromptGroup`、`RenderedInput`、`ModelSite`、`SAECheckpoint`、`Feature`、`ExperimentSpec`、`RunAttempt`、`Measurement`、`ReportArtifact`、`ShareRecord`。

`feature_key`至少包含真实模型和SAE revision、site、feature index。prompt保留source row、文本与token版本。推荐把通用基准文本与模型特定的render/tokenization分开保存。

`experiment_fingerprint = hash(canonical spec)`；`run_id`标识一次实际执行尝试；`artifact_hash`标识实际输出。不要仅用config hash作为可变结果目录，导致重跑悄悄覆盖。报告固定到某次run的manifest。

配置必须包括模型/SAE源码身份、权重hash、tokenizer/chat template、dtype、输入tokens、选特征策略、patch语义、采样参数、metric/protocol版本、估计量来源、平台与核心代码commit。资源预算不应改变科学身份，但影响调度与attempt metadata；实际数值行为改变必须生成新协议。

所有hash只证明字节和来源一致，不证明一个解释在科学上必然成立。重现区分byte-level artifact verification与跨设备numerical agreement，不能保证所有硬件逐bit相同。

### 7.4 API与任务状态

公开GET资源包括models、datasets、features、samples、runs、reports和artifacts，支持游标分页与明确排序。POST只接受结构化的预定义实验配置。

任务状态建议：`queued → running → succeeded / failed / cancelled`，另有`blocked`前置条件状态；包含lease、heartbeat、attempt、输出manifest。worker丢失后恢复，不让同一measurement被重复计入统计。

worker根据模型亲和性调度，显存不足可降低microbatch重试；重试不改变seed、输入、精度和任务定义。最多两次自动重试，之后报告原因。用户取消和预算终止都保留已提交measurement的明确状态。

### 7.5 兼容与外部引擎

先复用现有正确性路径，不把引擎重写或interp-engine集成作为本轮全部功能的前置依赖。未来/并行优化必须过logits、activations、SAE codes、identity、ablation、readout、seed和patch-timing的对等测试，单独记录backend provenance。收益没有实测前不写倍数。

## 8. 安全、成本与运行维护

匿名用户可浏览；在线自定义计算需要身份或严格配额。初始建议每用户最多1个排队/运行任务，全站待执行队列16个，具体额度由pilot修正。SAR等重任务另设次数和测量预算。管理员有停止新任务、禁用某模型和回滚release的开关。

自定义prompt默认私有，匿名/临时任务可设置24小时保留；有保存需求的任务显示明确保留政策。公开分享前确认文本、输出和激活元数据都可公开。tokens可能恢复文本，不能称为匿名化。日志不记录密钥或完整私有prompt。

上传仅允许有大小限制的JSON/Parquet等明确schema，拒绝pickle、任意Python、remote code、shell脚本、用户指定任意文件路径和任意下载URL。渲染文本和标签必须做XSS防护；外部链接只允许可信scheme。模型和数据源只在管理员白名单中装载。

监控包括成功率、失败类型、队列长度、worker heartbeat、GPU OOM、各数据集有效覆盖、延迟、磁盘和预算。每日元数据备份；release产物上传后独立重下载和校验；定期做恢复演练。前端版本可快速回退到上一个已验证manifest。

## 9. 工作包与依赖

所有工作包都有代码或配置、测试、运行记录和交接文件；单独文档或截图不算功能交付。

| 包 | 工作 | 依赖 | 验收产物 |
|---|---|---|---|
| 01 | 基线盘点、合同与发布身份 | 无 | 主仓commit/测试状态、HF源码与部署备份、schema v1、遗留问题清单 |
| 02 | 10数据适配器、500 prompts与分组 | 01 | 固定source manifest、人工核对记录、20/30 split、许可与排除记录 |
| 03 | 读出、token/patch映射及预算pilot | 01,02 | 每数据源fixture、20-prompt CPU/真实GPU pilot、预算与比较键 |
| 04 | 激活采集、稀疏索引与新核心测量 | 03 | 48-site×500覆盖台账、分片、hash、恢复验收、缺失清单 |
| 05 | Results与Dataset页面 | 01,03；完整验收需04 | 可筛选真实指标、逐样本钻取、下载、移动端 |
| 06 | Feature Explorer | 04 | all-observed feature索引、token高亮、搜索、300张精选证据卡目标 |
| 07 | API、队列、worker、隐私与持久化 | 01,03 | 在线真实模型、状态/取消/额度、跨重启恢复及权限测试 |
| 08 | 干预工作台和预计算实验 | 06,07 | 全算子与基线、真实回放、在线自定义生成、report入口 |
| 09 | SAR标准批次与分析页 | 03,07 | 原始估计量、633600次计划coalition台账、诊断热图及完整状态 |
| 10 | RSS批次与Localize页 | 02,03,07 | RSS变体/锚点/独立性验证、两类Localize演示及范围标记 |
| 11 | 分享、复现、HF数据发布 | schema从01开始；最终需04–10 | 每类报告固定URL、配置/产物/校验包、重跑差异页、许可通过 |
| 12 | 发布验证、容量/故障测试与上手测试 | 05–11 | 所有五项能力通过、恢复记录、操作手册、外部用户走通 |

最先完成一个垂直切片：一个真实小模型、每来源2条，共20条prompt，贯通“数据 → 激活 → 特征页面 → 真实干预 → SAR/RSS → 报告 → 下载/复现”。它只是开发验收，不替代正式500条。

前端、API和schema可以并行，但必须共用冻结合同与真实fixture。不要先铺满漂亮页面再临时决定数据结构。

### 9.1 Agent职责与合并规则

- 集成负责人：科学合同、schema、依赖边界、合并与发布。
- 数据/科学agent：数据采样、读出、变体、标签及估计量parity。
- GPU agent：采集/编码/干预/批次恢复；不直接改前端，不自改数学定义。
- 后端agent：API、身份、队列、持久化、参数安全和发布。
- 前端agent：真实数据浏览、交互、状态反馈、响应式与可访问性。
- 独立QA：从干净checkout验证、数值对照、故障注入和外部上手。

不同agent不同时改同一个核心schema。升级由集成PR先合并，再更新客户端生成物。每个PR必须标记是否改变科学语义、schema、源码身份或checkpoint。已有CI、打包和原论文回归继续作为必过检查。

每个工作包交接统一包含：准确commit、修改文件、执行命令、依赖版本、数据/模型身份、测试摘要、产物路径与hash、失败与未完成项、下一个agent的入口。不能把计划步骤写成已完成结果。

## 10. 发布门槛

### 10.1 数据与科学

- 十个来源各恰好50条基础prompt，500个独立分组定义清楚；原始source、许可证、文本变换与排除记录齐全。
- 200 discovery / 300 evaluation不重叠；同源内容和所有派生变体不跨组泄漏。
- 48-site基础矩阵完成或明确失败并处理，不以“已注册”代替真实计算通过；发布计数来自台账。
- SAR/RSS原始实现parity通过；低方差、空support、无效分母、缺少任务标签等正确呈现。
- 合法的同模型同站点双SAE配对比较可运行；不同模型比较不制造同一坐标对齐。
- Localize只接受完整受控grid，不误用普通bootstrap。

### 10.2 功能

- 未登录用户能从数据集进入prompt、特征、真实干预结果和分享报告。
- 至少一个真实模型能接收新输入，完成普通干预以及受限的研究作业；不是只有预计算回放。
- 五项功能都可从导航访问，并显示每模型/数据集实际支持状态。
- 导出包含配置、来源、数值、manifest和复现说明。另一个环境能核验存档，并在符合条件的模型环境中重跑。

### 10.3 工程与运行

- 确认目标浏览器和窄屏布局无关键阻断；键盘可操作，token高亮不只依赖颜色。
- 标准网络测试条件下，预计算页面目标p95首屏不超过2.5秒、索引查询目标p95不超过1秒；这些是验收目标，不是现在性能声明。冷启动和GPU运行时间单列，不混入交互响应。
- 浏览器不全量加载激活数组；大列表分页/虚拟化；长图按需取数。
- 模拟重复提交、worker退出、OOM、上传损坏、数据库/Space重启、预算耗尽、取消、无GPU和部分结果恢复，计数无重复且状态真实。
- 权限与私有数据隔离测试通过；缺少授权不返回别人的artifact或prompt。
- 真正从备份恢复一次，并对公开包做清洁环境重下载校验。
- 至少三位未参与开发的研究者在不需维护者现场指导下完成主流程。收集卡点并修复阻断问题。
- 有部署、回滚、开启/暂停在线算力、更新语料和处理数据删除的操作说明。

满足这些条件才将本轮称为完整可用的研究平台版本；500条的覆盖限制仍在dataset card中清楚保留。

## 11. 主要风险与处理

| 风险 | 处理 |
|---|---|
| 全模型全实验乘积失控 | 48基础/6深度/3SAR/1在线，任务计费与预算关卡 |
| 50个样本难支持精细排名 | 显示有效N与CI，分任务报告，不宣传全面能力认证 |
| 罕见feature没有足够样例 | 显示观测不足，不补假解释；记录扩充需要 |
| RSS变体改变原任务 | dataset-aware变换、锚点和答案验证、人工审核 |
| 第一token得分冒充答案正确率 | 引入完整sequence读出，诊断/任务/生成指标分开 |
| Base/IT、hook、dtype混用 | canonical identity、加载和patch验收，比较键校验 |
| 页面好看但无法真实运行 | 首先做真实小模型垂直切片；在线新输入为发布门槛 |
| Spot/租赁GPU或Space丢数据 | 原子产物、lease恢复、独立持久化、备份恢复演练 |
| 自动解释被当成真相 | 标注来源和假设状态，保留反例与实验报告 |
| 多agent重复重构或悄悄改估计量 | schema冻结、责任边界、原始估计量parity、独立QA |
| 公开500条以后反复调参 | 版本/访问用途记录；未来确认实验另取数据 |

## 12. 建议第一步和本轮完成定义

第一步不是租大GPU、批量下载所有权重或全面重写UI。先提交工作包01–03的合同、十个adapter、500条来源清单与pilot，让一条真实完整研究流程可运行，再扩容为发布批次。

本轮完成后的介绍应能够被用户实际操作证实：

> OpenSAE lets researchers explore SAE features, intervene on them, and test what they explain across a curated set of inputs. Each experiment can be inspected, shared and reproduced.

关键资产不是十个数据集名字或几百万个数，而是“来源输入 → 真实特征证据 → 受控干预 → 专门研究分析 → 可核查报告”的完整连接。

## 参考来源

下列链接是本方案的事实依据。任务分层、参数预算、页面结构、数据抽样、排期依赖和验收指标属于本方案建议，而不是已完成结果。

```text
[S1] SAEBase README（本轮GitHub连接读取）
https://github.com/OpenSAE/SAEBase/blob/main/README.md
[S2] Verified Model and SAE Layer Matrix（本轮读取）
https://github.com/OpenSAE/SAEBase/blob/main/docs/verified-model-matrix.md
[S3] Feature Interventions（本轮读取）
https://github.com/OpenSAE/SAEBase/blob/main/docs/feature-interventions.md
[S4] 原始 SAR 估计量（本轮读取）
https://github.com/OpenSAE/SAE-Behavioral-Compositionality/blob/main/sar.py
[S5] SAEBase 已有研究/可靠性/受控输入合同（前述对话已读取；实施时再次固定源码commit）
https://github.com/OpenSAE/SAEBase/blob/main/docs/research-integrations.md
https://github.com/OpenSAE/SAEBase/blob/main/docs/reliability.md
https://github.com/OpenSAE/SAEBase/blob/main/docs/controlled-inputs.md
https://github.com/OpenSAE/SAEBase/blob/main/docs/activation-store.md
[S6] 现有work_plan目录（本轮确认只有01/02/03）
https://github.com/OpenSAE/SAEBase/tree/main/work_plan
[D1] WikiText
https://huggingface.co/datasets/Salesforce/wikitext
[D2] TinyStories
https://huggingface.co/datasets/roneneldan/TinyStories
[D3] SQuAD官方页面
https://rajpurkar.github.io/SQuAD-explorer/
[D4] BoolQ
https://huggingface.co/datasets/google/boolq
[D5] ARC
https://huggingface.co/datasets/allenai/ai2_arc
[D6] HellaSwag
https://huggingface.co/datasets/Rowan/hellaswag
[D7] WinoGrande及官方许可/无标签test说明
https://huggingface.co/datasets/allenai/winogrande
https://github.com/allenai/winogrande/blob/master/README.md
[D8] GSM8K
https://huggingface.co/datasets/openai/gsm8k
[D9] HumanEval及代码执行警示
https://huggingface.co/datasets/openai/openai_humaneval
[D10] TruthfulQA
https://huggingface.co/datasets/truthfulqa/truthful_qa
[I1] HF Docker Spaces：持久化与独立服务能力
https://huggingface.co/docs/hub/spaces-sdks-docker
```
