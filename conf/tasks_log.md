# SAEBase Task Registry

## Capture统一筛选和答案行为模式（2026-11-15）

标准配置新增filter=10和capture_count=null：数字filter对whiten、non_whiten、g_s、SAR统一应用既有SAR筛选，all跳过过滤；capture_count独立控制输出特征／pair数量，默认跟随filter。参考题100道与Dataset实际题数保持独立。mode=g_s按仅删除一项后的标准答案平均log概率排序，附delta_g；mode=sar使用原SAR抽样回归；mode=I穷举并按固定背景真实四格I的最大值排序。具体参数、兼容方式及输出见tasks/capture/README.md。下文旧规则描述为历史配置语义，不覆盖上述新标准默认值。

## 目的和主要功能

说明科研任务的登记入口和当前捕获方法的参数与结果语义。

## 适用范围

用于选择任务及维护方法配置；每项能力的实际支持由所属任务接口决定。

正式 SAE 操作包括 `capture`、`intervene`、`generate`、`circuits`、`explain`、`benchmark` 和 `train`。

执行前检查由 `precheck` 提供，与七类科研操作共同组成八类 Task，在 `tasks/plugins.py` 中通过 `saebase/registry.py` 统一注册，由
`saebase/runner.py` 根据 Hydra `task.name` 分派。
`python -m saebase.runtime.train --cfg job` 仅展示合成配置，不会加载模型或启动计算。
真实实验应先通过任务准入、权重来源、身份、数据划分及执行授权校验。

Capture 支持 `non_whiten`、`whiten` 与 `sar`；特征数量由 `task.top_feature_num` 指定。SAR 需要明确的目标与对照 token、原版估计器路径，并保存子集干预响应与测量协议。

最后一个 token 的 Capture 以选出的 Top N 特征解释是否齐全判断解释完成；其他激活特征的解释覆盖单独记录。补充解释与结果读取沿用同一完成条件，无须重新计算激活。

SAR 可设置 `task.sar.feature_selection=whiten`，在选定数据集上拟合原有逐特征标准化，并对同一批 whiten Top N 特征执行子集干预。`task.sar.top_pairs=5` 使用原版 `design` 和相同拟合组合求出二阶系数，按系数绝对值保存前五对及两端特征解释；系数并非单独的 SAR 分数。完整系数和 ID 保存到原有 SAR 原始数组产物。

SAR 的标准配置启用 `task.sar.feature_filter.enabled=true`，在 Top N 候选选取和干预拟合之前应用 `to_ziye` v2 规则：排除公开解释命中标点、功能词、自然语言语法、通用问句、转述结构或空白排版的特征，以及 Neuronpedia 原生非零激活率达到 `min_native_nonzero_fraction`（默认 0.5）的特征。编程语法具有原规则的例外，主题词与专名按原规则保留。解释与激活率按当前 SAE 来源匹配，跨层特征编号分别判定。缺失标签或空激活率按现有证据判定并记录数量；来源不可用或元数据覆盖不完整时停止该项测量。过滤规则、排除编号、命中原因和来源指纹保存到 `metadata/sar_feature_filter.json` 并纳入测量身份。候选不足时明确报错。独立 `whiten` 与 `non_whiten` 保持原有规则；历史结果保留原始协议，旧显式配置未声明过滤时保持关闭，可显式关闭过滤以复现旧运行。

当前 SAEBase 的配对允许重复端点。`SAR/to_ziye` 的 K32 交付同样按交互系数绝对值排序，并额外贪心跳过已使用端点，选取五对、十个不同特征；两者的展示约束分别保留。

SAR 可启用 `task.sar.cohort_filter.enabled=true`，配合 `feature_selection=whiten` 使用。先拟合原有全提示词元 mean/std，再对全部选定题目的末词元计算有符号 z-score；按 `z > zscore_threshold`（默认1）计数，占全部题目的比例达到 `min_question_fraction`（默认0.5，含边界）时排除该特征。频率排除集合与现有语义和原生密度排除集合取并集，在每题选Top N和SAR测量之前统一应用，single与pair使用同一候选集合。规则、题目身份、全量超阈计数及排除ID保存到 `metadata/sar_cohort_filter.json`，测量身份包含其指纹。该功能保持独立whiten和non_whiten的行为，旧配置默认关闭；候选不足时报告实际缺口。

`task.sar.pair_ranking=signed` 按有符号二阶交互系数降序展示配对；`absolute` 保持按绝对值降序的原有行为。完整45个系数（Top N=10时）仍保留，报告写明实际排序口径。

`task.sar.cohort_filter.reference_sample_count` 单独指定特征选择参考题数，标准配置为100。启用频率过滤后，参考题通过原Dataset选择接口取得，用来计算全提示词元mean/std及参考题末词元的超阈频率；`dataset.sample_count` 继续单独指定实际SAR题数。参考数设为null时沿用SAR选定集合。明确指定的参考数不足时报告缺口，参考选择来源与题目身份随过滤记录保存。参考快照按样本ID和完整prompt对应到SAR题目，顺序变化不会混用快照；目标题未在参考集中时独立捕获。

## Top N 单特征干预与生成

`intervene` 和 `generate` 均支持 `task.feature_selection=capture_top`、`task.top_feature_num=10`。每题使用 capture 的末 token Top N 排名，从同一原始 prompt 分别启动每个特征的独立实验。`dataset.sample_count` 指定题数；`task.prompt` 指定单个问题时仅对该问题执行。旧的 `configured` 单特征模式继续作为默认值。

`task.feature_ranking=non_whiten` 按原始激活绝对值排序；`whiten` 复用 capture 的全 prompt token mean/std 和绝对 z score 排名，参考集合为当前 Dataset 选定的样本。显式问题的 whiten 同样以 Dataset 样本作为参考集合。仅统计 std > 0 的特征；这些模式沿用独立 capture 的规则，SAR 专属过滤仍仅对 SAR 生效。

可用 `task.capture_source=/absolute/path/to/capture/run` 直接复用已保存排名；当前 Dataset 选定的问题（或显式 prompt）按完整 prompt 的 SHA256 对应，检查模型、SAE、层、hook、排序方式及权重指纹。保存的 Top N 不足、问题不匹配或身份不一致时明确报错。复用 SAR 的 Top features 时，应选择与该结果一致的 ranking；其已有过滤选择按保存列表原样沿用。

`intervene` 对每个特征单独替换为 `task.value`（默认0）；`generate` 对每个特征单独应用 `task.intervention` 和 `task.value`，例如 `intervention=ablate` 表示清零，`intervention=scale value=2` 表示放大两倍。生成先固定原问题选出的特征 ID，再在每一步新前缀的末 token 处对该特征执行操作；每个特征从原问题独立生成。为保持单特征语义，capture_top 与自定义 protocol/conditions 分开使用。

正式结果 `metadata/<task>_result.json` 包含逐题逐特征结果；`reports/<task>_features.jsonl` 增量记录排名、特征解释、激活、完整问题、选取身份和干预/生成结果。后续失败前已经写出的行保留在该文件中；可用 `runtime.mode=resume` 恢复，按问题、特征、排序、权重和执行协议的测量身份复用成功行、重试失败行。whiten 统计另存于 `metadata/capture_normalization.*`，结果引用产物的大小和 SHA256。

示例（每题 Top 10 逐个清零）：

```bash
python -m saebase.runtime.train model=gpt2small dataset=ioi dataset.sample_count=1 task=intervene task.execute=true task.feature_selection=capture_top task.top_feature_num=10 task.feature_ranking=non_whiten task.value=0 experiment.sub_name=top10_intervene
python -m saebase.runtime.train model=gpt2small dataset=ioi dataset.sample_count=1 task=generate task.execute=true task.feature_selection=capture_top task.top_feature_num=10 task.feature_ranking=non_whiten task.intervention=ablate task.value=0 task.max_new_tokens=8 experiment.sub_name=top10_generate
```

2026-11-15 默认配置约定：capture 默认 `task.mode=whiten`；intervene 与 generate 在 `capture_top` 模式下默认 `task.feature_ranking=whiten`。显式指定 non_whiten 仍可复现原始激活排名；旧结果保留原协议。

## 主要探究配置的完整干预批次

`conf/experiment/feature_tasks_matrix.yaml` 固定十个模型、十二个数据集和两个任务；Llama 使用当前 L21，其余采用主要探究笔记中明确的历史 SAE 层。各 Dataset 仍请求500题，实际按数据集可用样本池执行。统一 whiten Top10、逐特征清零；generate 使用贪心解码，最多8个新token。

`python -m tasks.intervene.matrix --config conf/experiment/feature_tasks_matrix.yaml --report <报告目录>` 使用正式 Runtime 执行，按模型配置的解释器加载，复用一次模型/SAE覆盖同一阶段的十二个数据集。先完成 intervene 阶段，再执行 generate；失败项第二轮独立重试，成功单元跳过。题目冻结到报告目录的 inputs/；可验证的已有 whiten capture 排名直接复用，其他单元先执行新的 capture。每个单元保留模型、SAE、层、源 capture、运行目录及失败原因。

大批次使用 `task.inline_results=false`，完整逐特征结果保存在正式结果引用的 JSONL 中，避免重复将所有生成轨迹复制到总清单。`tasks.intervene.top_features.read_feature_results(run_dir)` 返回每项测量的最新记录。`metadata/<task>_progress.json` 提供即时进度；`task.continue_on_error=true` 保留失败记录并继续其他独立特征。恢复前检查每个测量的身份，末尾被中断写入的不完整行可修复；中间损坏仍明确报错。外层 Runtime 同时检查配置及源码恢复身份。

`task.reuse_prompt_observations=true` 在同一个原问题的独立特征实验间复用完全相同 token 前缀、SAE位置和实际回写激活的计算结果。该复用不使用 KV 历史，不改变已有逐步干预协议；作用域在每道题结束时释放，最多保留64项。真实 GPT-2 的十个特征已验证干预 logits、全部生成token和逐步指标与独立参考计算完全相同。

Generate 默认 `task.stop_at_context_limit=true`：保持模型原有 prompt tokenization，当生成前缀超过模型原生执行窗口时停止，记录 `context_limit`，保留此前已生成的 token。`max_new_tokens` 为生成上限；达到上下文上限的输出可短于该值。设为 false 可恢复超出窗口时报错的行为。

完整干预批次中新建的 selection capture 关闭 Neuronpedia 完成条件，只负责可验证的数值选特征；已有 source 的解释原样保留。解释覆盖缺失不会阻止没有公开解释的模型执行 intervene / generate。

## SAR 已保存组合中的四状态模式搜索

`python -m tasks.capture.pair_search --run <已有SAR运行目录> --output <新报告目录> --minimum 0.5 --spread 0.05 --gain 0.05` 搜索原选定特征的全部pair。要求00、10、01的平均token概率严格高于minimum、三者最大跨度不超过spread，11相对最高基线的增幅至少为gain，且logprob四格交互为正。概率统一为exp(平均答案token logprob)，差值参数以0到1表示（0.05为5个百分点）。

四状态必须来自其他特征完全相同的背景；重复mask固定取第一次原始测量，并保留全部重复行号。脚本校验NPZ的SHA256，返回严格命中数量、全部基线合格候选及每题最接近候选，明确记录特征ID、解释、背景状态、四个g和概率、原始测量行。`--surrogate` 使用原SAR设计函数和拟合样本，对全部二进制mask进行二阶预测检查，预测与真实测量单独保存。本入口只读取历史实验，不运行模型。

SAR四格搜索默认口径已修正：`--gain` 默认不设置额外概率增幅门槛。基线概率与接近程度仍由 `--minimum`、`--spread` 控制；入选需11高于最高基线、实测I为正，并按实测logprob交互I降序排列。每题展示I最大的合格候选，两行概率分别为平均token概率与答案联合概率，最后一列分别计算各自的11减最高基线。历史显式 `--gain 0.05` 搜索结果保留原阈值含义。

CaptureMatrix 支持 `matrix.dataset_overrides`，用于独立设置本批的实际观测题数等 Dataset 参数；`task.sar.cohort_filter.reference_sample_count` 继续独立控制筛选参考题数量。
