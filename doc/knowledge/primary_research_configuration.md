# 主要探究的配置

记录日期：2026-11-15。用户约定：以后提到“主要探究的配置”，指本笔记中的模型、SAE、数据集及实验约定。

## 模型和 SAE

范围来自已有 whiten / non_whiten capture 的十个模型。下表保留历史捕获层；当前重点 SAR 使用 Llama 3.1 8B 的 **L21**。

| 模型 | 项目配置名 | 权重身份 | SAE 配置键 | 历史捕获层 | 已存 Neuronpedia 解释 |
| --- | --- | --- | --- | --- | --- |
| GPT-2 Small | `gpt2small` | `openai-community/gpt2`（加载名 `gpt2-small`） | `gpt2small_res_jb_l8` | 8 | 有 |
| Gemma 2 2B Base | `gemma2_2b_base` | `google/gemma-2-2b` | `gemma2_2b_pt_l25_canonical` | 25 | 有 |
| Gemma 3 1B PT | `gemma1b` | `google/gemma-3-1b-pt` | `gemma1b_scope_resid` | 13 | 有，部分特征缺失 |
| Gemma 3 4B PT | `gemma3_4b_pt` | `google/gemma-3-4b-pt` | `gemma3_4b_pt_scope_resid` | 17 | 有，部分特征缺失 |
| Gemma 3 12B IT | `gemma3_12b_it` | `google/gemma-3-12b-it` | `gemma3_12b_it_scope_resid` | 24 | 无 |
| Llama 3.1 8B Base | `llama31_8b` | `meta-llama/Llama-3.1-8B` | `llama31_8b_lxr_8x` | 16；当前默认 **21** | 有，L16、L21 均有 |
| Qwen3 1.7B Base | `qwen3_1_7b_base` | `Qwen/Qwen3-1.7B-Base` | `qwen3_1_7b_base_resid_w32k_l0_100` | 14 | 无 |
| Qwen3 8B Base | `qwen3_8b` | `Qwen/Qwen3-8B-Base` | `qwen3_8b_resid_w64k` | 18 | 无 |
| Qwen3.5 9B Base | `qwen35_9b` | `Qwen/Qwen3.5-9B-Base` | `qwen35_9b_resid_w64k` | 16 | 无 |
| DeepSeek-R1-Distill-Llama-8B | `r1_distill_llama8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `r1_distill_llama8b_llamascope_l23` | 23 | 有 |

“有/无”指这些 SAE 和已保存结果中的解释覆盖情况。需要有解释的模型时，从本表选取标“有”的六个模型。Qwen3.5 9B 的历史结果为 non_whiten 有样本、whiten 无有效样本。

历史层以批次显式配置为准。例如 Gemma 3 4B PT 历史为 L17，当前 SAE 通用预设为 L29，复现时须显式指定 L17。Qwen3 1.7B 的历史 SAE 键已不在当前 SAE 配置目录的具名预设中；模型仍登记此 profile，当前具名预设为 `qwen3_1_7b_base_resid_w32k`。复现时应核对实际 SAE 身份。

## 数据集

| 数据集 | 项目配置名 | 子配置（已指定时） | 已完成本轮 100 题筛选、10 题 SAR |
| --- | --- | --- | --- |
| HotpotQA | `hotpotqa` | `distractor` | 是 |
| BBH | `bbh` | — | 是 |
| IOI | `ioi` | — | — |
| RAVEL | `ravel` | `city` | — |
| CounterFact | `counterfact` | — | — |
| ARC-Challenge | `arc_challenge` | `ARC-Challenge` | 是 |
| MMLU-Pro | `mmlu_pro` | — | 是 |
| GSM8K | `gsm8k` | `main` | 是 |
| HumanEval | `humaneval` | `openai_humaneval` | — |
| TruthfulQA | `truthfulqa` | `multiple_choice` | 是 |
| HarmBench | `harmbench` | — | — |
| XSTest | `xstest` | — | — |

历史 capture 使用 `non_whiten`、`whiten` 两种模式，种子 42，每个数据集请求 500 条。实际固定样本池中 HumanEval 为 164 条、HarmBench 为 400 条、XSTest 为 450 条，其余各 500 条，共 5514 条；具体运行是否成功以保存结果为准。

## 当前 SAR 约定

- 模型 `llama31_8b`，SAE `llama31_8b_lxr_8x`，显式层 **21**，种子 42。
- 每个数据集用 **100 题**作为标准化及特征频率筛选依据，另设参数控制此数量；实际 SAR 只跑其中 **10 题**。z score 的 mean/std 来自参考题的全部 prompt token；频率来自每题 prompt 最后一个 token。
- SAR 前先应用 `to_ziye` 对齐的语义与原生激活密度过滤，再排除在参考题中 **z > 1 的题数占比 ≥ 50%** 的特征。阈值和比例可配置；这些过滤仅对 SAR 生效。
- 过滤后按 whiten 得分选择单特征 Top 10。SAR pair 按交互系数**原值降序**取 Top 5，允许不同 pair 共用端点。
- SAR 使用 384 个 coalition、256 个拟合样本、128 个留出样本，读出为 `answer_mean_logprob`，答案执行后端 `auto`。
- 汇报逐题包含完整原问题及选项、中文翻译、数据集标准答案、Top 10 single feature（中文解释、raw activation、z score、参考题频率），以及 Top 5 SAR pair（两端中文解释和带符号交互值），并提供 single 与 pair 的对比。

## 配置及结果依据

Intervene 和 generate 可用 `task.feature_selection=capture_top task.top_feature_num=10`，对每题 capture 选出的十个特征分别执行独立实验；capture 默认 `task.mode=whiten`；跟随选特征的任务默认 `task.feature_ranking=whiten`，也可显式指定 `non_whiten`，`task.capture_source` 可复用已有 capture 的排名。参数和示例见 [任务接口说明](../../conf/tasks_log.md#top-n-单特征干预与生成)。

- [历史模型和数据集矩阵](../../conf/experiment/capture_matrix.yaml)。其中 Llama 的 L16 是历史配置；当前默认 L21 见 [模型目录](../../conf/model/catalog.yaml) 和 [SAE 目录](../../conf/sae/catalog.yaml)。数据集预设见 [数据集目录](../../conf/dataset/catalog.yaml)。
- [历史 capture 解释覆盖记录](../../experiment/central-maintenance/top10_explanation_coverage_20261114/coverage.json)。
- [五个新增 QA 数据集的批次配置](../../experiment/central-maintenance/sar_fiveqa_ref100_20261114/matrix.yaml)及[完整结果](../../experiment/central-maintenance/sar_fiveqa_ref100_20261114/five_datasets_full_report_zh.md)。
- [HotpotQA 的 100 题参考、10 题 SAR 完整结果](../../experiment/HotpotQA/llama31_8b/capture/hotpot10_L21_ref100_freq50_sar_20261114_42/reports/full_questions_singles_pairs_zh.md)。

## 当前完整干预批次

2026-11-15 启动主要探究配置的完整 intervene / generate 批次：十个模型、十二个数据集，每题 whiten Top10 独立清零，generate 最多8个新token；Llama 使用当前 L21。配置见 [完整批次](../../conf/experiment/feature_tasks_matrix.yaml)，[实时进度及结果入口](../../experiment/IOI/gpt2small/intervene/_tmp/primary_whiten_top10_20261115/matrix/progress_zh.md)。后台 tmux 会话为 `primary-feature-tasks-20261115`；进度与完整运行目录以实时状态文件为准。

## 当前五个自然QA的500题Capture批次

用户将此前两个模型、100题的Capture批次作废，改为以下80组：

- 数据集顺序：`gsm8k`、`mmlu_pro`、`arc_challenge`、`truthfulqa`、`bbh`；每个取500题，不足则取全部。
- 模型顺序：`llama31_8b` / `llama31_8b_lxr_8x` / L21；`qwen35_4b` / `qwen35_4b_matryoshka_l15_65k` / L15；`gpt2small` / `gpt2small_res_jb_l8` / L8；`gemma3_4b_pt` / `gemma3_4b_pt_scope_resid` / L17。
- 每个数据集、模型下依次执行 `whiten`、`non_whiten`、`g_s`、`I`；此处 `I` 为1024状态穷举，回归 `sar` 不在本批。
- 每组观测500题、筛选参考题500；`filter=10`、`capture_count=10`、seed=42；先完成全部Precheck，再启动tmux。
- 计算沿用现有Capture；固定题目在模型与模式间共享。公开解释有缺失时保留实际覆盖状态。

[批次配置](../../experiment/maintenance/20261116_fiveqa_four_models_capture500/matrix.yaml)、[实时进度与结果入口](../../experiment/GSM8K/llama31_8b/capture/_tmp/fiveqa4models_500_20261116/progress_zh.md)。
