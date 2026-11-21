# The configuration of the main inquiry

Date of record: 2026-11-15. User engagement: Later reference to the “principal search configuration” refers to models, SAEs, data sets and experimental engagements in this note.

## Models and SAE

The range is derived from 10 models that already exist. The following table retains the historical capture layer; the current focus SAR uses Llama 3.1 8B ** L21**.

| Model | Project Configuration Name | Weighted identity | SAE Configuration Key | Historical capture layer | Saved Neuronpedia Interpretation |
| --- | --- | --- | --- | --- | --- |
| GPT-2 Small | `gpt2small` | `openai-community/gpt2` (loading name `gpt2-small`) | `gpt2small_res_jb_l8` | 8 | Yeah. |
| Gemma 2 2B Base | `gemma2_2b_base` | `google/gemma-2-2b` | `gemma2_2b_pt_l25_canonical` | 25 | Yeah. |
| Gemma 3 1B PT | `gemma1b` | `google/gemma-3-1b-pt` | `gemma1b_scope_resid` | 13 | Yes, some of the features are missing |
| Gemma 3 4B PT | `gemma3_4b_pt` | `google/gemma-3-4b-pt` | `gemma3_4b_pt_scope_resid` | 17 | Yes, some of the features are missing |
| Gemma 3 12B IT | `gemma3_12b_it` | `google/gemma-3-12b-it` | `gemma3_12b_it_scope_resid` | 24 | None |
| Llama 3.1 8B Base | `llama31_8b` | `meta-llama/Llama-3.1-8B` | `llama31_8b_lxr_8x` | 16;current default **21** | Yes, L16, L21 both. |
| Qwen3 1.7B Base | `qwen3_1_7b_base` | `Qwen/Qwen3-1.7B-Base` | `qwen3_1_7b_base_resid_w32k_l0_100` | 14 | None |
| Qwen3 8B Base | `qwen3_8b` | `Qwen/Qwen3-8B-Base` | `qwen3_8b_resid_w64k` | 18 | None |
| Qwen3.5 9B Base | `qwen35_9b` | `Qwen/Qwen3.5-9B-Base` | `qwen35_9b_resid_w64k` | 16 | None |
| DeepSeek-R1-Distill-Llama-8B | `r1_distill_llama8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `r1_distill_llama8b_llamascope_l23` | 23 | Yeah. |

"Yes/No" refers to the interpretation coverage of these SAEs and saved results. When an explanatory model is required, the six models marked "Yes" are selected from this table. The historical results of Qwen 3.5 9B are non_whiten with a sample and whiten without a valid sample.

The historical layer is based on a batch-based configuration. For example, Gemma 3,4B PT history is L17, the current SAE generic preset is L29, and L17 must be specified visibly when it is repeated. The Qwen3 1.7B historical SAE key is no longer in the name preset of the current SAE configuration directory; the model is still registered with this profile and the current name preset is `qwen3_1_7b_base_resid_w32k`. The actual SAE identity should be checked for the recurrence.

## Data Set

| Data Set | Project Configuration Name | Subconfiguration (when specified) | 100th Question Filter, 10th SAR |
| --- | --- | --- | --- |
| HotpotQA | `hotpotqa` | `distractor` | Yes. |
| BBH | `bbh` | — | Yes. |
| IOI | `ioi` | — | — |
| RAVEL | `ravel` | `city` | — |
| CounterFact | `counterfact` | — | — |
| ARC-Challenge | `arc_challenge` | `ARC-Challenge` | Yes. |
| MMLU-Pro | `mmlu_pro` | — | Yes. |
| GSM8K | `gsm8k` | `main` | Yes. |
| HumanEval | `humaneval` | `openai_humaneval` | — |
| TruthfulQA | `truthfulqa` | `multiple_choice` | Yes. |
| HarmBench | `harmbench` | — | — |
| XSTest | `xstest` | — | — |

History capture uses two models, `non_whiten`, `whiten`, seeds 42, each data set requests 500 entries. The actual fixed sample pool of HumanEval is 164, HarmBench is 400, XST is 450, and the remaining 500 are 500 articles, totalling 5514; whether or not the specific operation is successful is based on the preservation result.

## Current SAR engagement

- Model `llama31_8b`, SAE `llama31_8b_lxr_8x`, visible layer **21**, Seed 42
- Each data set uses **100 ** as the basis for the standardization and characterization frequency screening, with a separate parameter to control this quantity; the actual SAR runs only **10 **. z score mean/std from all references token; frequency from the last token of each topic.
- SAR Apply First `to_ziye` Alignment semantics and primary activated density filters, which are then excluded from reference questions **z > 1 Percentage of topics ≥ 50%** . Thresholds and ratios can be configured; these filters are only right SAR Entry into force.
- Following filtering, press whiten score to select single feature Top 10. SAR pair takes Top 5 in order of original ** interactive factor ** and allows different pair peer-to-ends.
- SAR uses 384 proposed, 256 proposed and 128 reserved samples and reads `answer_mean_logprob` at the back end of the answer `auto`.
- The report consists of full original questions and options, Chinese translation, standard answers to data sets, Top 10 single answer (Chinese interpretation, Raw action, z score, reference frequency) and Top 5 SAR pair (Chinese interpretation and symbol interaction values) and provides a comparison of single with pair.

## Configure and Basis of Results

Interventions can be `task.intervene.feature_selection=capture_top task.intervene.top_feature_num=10` to generate an independent experiment for each of the ten features selected by `task.generate.feature_selection=capture_top task.generate.top_feature_num=10`; Capture default `task.mode=whiten`; tasks following the selected feature default `task.feature_ranking=whiten`, or may be specified in a visible way by re-numbering `non_whiten`, `task.capture_source` with existing Capture. Parameters and examples can be found in [mission interface description](../code/maintenance/task_log.md#top-n-single_feature_intervention_and_generation)).

- [Historical model and dataset matrix](../../conf/tasks/capture/task.yaml).of which Llama It's... L16 is the historical configuration;current default L21 See [Model Directory](../../conf/models/catalog.yaml) and [SAE Contents](../../conf/base/interfaces/sae_profiles.yaml).See data set presets [Catalogue of data sets](../../conf/datasets/catalog.yaml).
- [History capture Explain Overriding Records](../../experiment/batches/capture/top10_explanation_coverage/coverage.json).
- [Five new QA Batch Configuration of Data Sets](../../experiment/batches/capture/sar_qa_comparison/matrix.yaml)and[Full result](../../experiment/batches/capture/sar_qa_comparison/five_datasets_full_report_zh.md).
- [HotpotQA It's... 100 Subject reference, 10 Question SAR Full result](../../experiment/HotpotQA/llama31_8b/capture/reference_additivity/reports/full_questions_singles_pairs_zh.md).

## Current full intervention batch

2026-11-15 Starts the full intevene / complete batch of the configuration: ten models, twelve data sets, each whten Top10 stand-alone zero, primerate up to eight new token; Llama uses the current L21. See the [full batch](../../conf/tasks/intervene/task.yaml), [real time progress and result portal](../../experiment/batches/intervene/whiten_feature_effects/reports/session/matrix/progress_zh.md). Backstage tmux session is `primary-feature-tasks-20261115`; progress and complete running directory is based on real-time status files.

## 500 Capture batches of five natural QAs in history (suspended)

For the previous two models, Capture batches for 100 questions, the user replace the following 80 groups:

- The order of the data set: `gsm8k`, `mmlu_pro`, `arc_challenge`, `truthfulqa`, `bbh`; each taking 500 questions, if not all.
- Model order: `llama31_8b`/ `llama31_8b_lxr_8x`/L21; `qwen35_4b`/ `qwen35_4b_matryoshka_l15_65k`/L15; `gpt2small`/ `gpt2small_res_jb_l8`/L8; `gemma3_4b_pt`/ `gemma3_4b_pt_scope_resid`/L17.
- Individual datasets, models executed sequentially `whiten`,`non_whiten`,`g_s`,`I`;Here. `I` Yes. 1024 We're out of shape. Come back. `sar` Not in this instalment.
- 500 questions per group, screening reference 500; `filter=10`, `capture_count=10`,seed=42; complete all Precheck and then start tmux.
- The calculation follows the existing Capture; the fixed title is shared between the model and the mode. The actual cover is maintained when the public explanation is missing.

[Batch Formula](../../scripts/capture_recipes/natural_qa_full_cohort.yaml) · [Formula Use](../../scripts/capture_recipes/README.md) · [Progress and results before pause](../../experiment/batches/capture/qa_capture_pilot/full_cohort_pilot/reports/session/progress_zh.md).


## 2026-11-18100 Capture and Directional Backup Records (completed)

Current scope of implementation is five data sets×100 Question×Four Models×Four configurations, total 80 Group: The order of data sets and models follows the previous section; the four configurations are as follows: `whiten`,`non_whiten`,`g_s`,`I`.Each group of observations 100 Questions, screening references 100 Question,`filter=10`,`capture_count=10`,seed=42.Original 500 Quote is still suspended..

The public model ' s numerical strategy has been completed to stabilize the average, repeated with vLLLM, with the original model formula, SAE position and definition of the answer rating; the different bats continue to be selected by Precheck and require full consistency in the rating. The current identity is `saebase.vllm_batch_invariant.mean.v6`, Qwen also maintains a stable GDN cumulation with `saebase.vllm_batch_invariant.qwen_gdn_scan.mean.v6`. The user ' s most recent request is to continue to save time: the successful results of the old v4/v5 have been used, and only five BBH pre-screens have failed (500 implementations) and five items, totalling 505 times; the new result is used v6, each original version and source is retained, and the original result is not marked as v6.

[Current Batch Configuration](../../experiment/batches/capture/normalization_fix/matrix.yaml),[Formal repairs and acceptance instructions](../../experiment/batches/capture/normalization_fix/Formal rehabilitation statement.md),[Current batch progress](../../experiment/batches/capture/qa_capture_comparison/stable_mean/reports/session/progress_zh.md).

Current run-off entrance: [Leave list only](../../experiment/batches/capture/normalization_fix/repair_manifest.json), [State of play-off](../../experiment/batches/capture/bbh_capture_repair/reports/session/status.json)].


## Current 500 Capture amplification (reference 100)

Each data set has been expanded from 100 to 500 questions, with no change: four models/SAE/levels, four Capture, filler=10, Capture_count=10, Seed=42, and 100 questions for standardization and frequency screening reference groups. Five data sets are available for 500 questions, and the original 100 questions are aligned with the answers, with 8,000 previously completed, and only 400 questions x 5 data sets x 4 models x 4 configurations = 32000.

[Configure](../../experiment/batches/capture/qa_expansion/expansion_inputs/matrix.yaml),[Repeat title and new scope](../../experiment/batches/capture/qa_expansion/expansion_inputs/expansion_manifest.json),[Start statement](../../experiment/batches/capture/qa_expansion/expansion_inputs/Capability Operations Description.md),[Real time progress](../../experiment/batches/capture/qa_expansion/reports/session/progress_zh.md).Use after startup of precheck tmux Session saebase-capture500-expand-20261118;By Data Set→Model→Configure Execution.


## MMLLU-Pro / Llama: I by g(S)

This time only the data set/model is currently 500 questions and the original results of the three other configurations are followed; reference 100 questions, L21, filter10/capture_count10 and soe 42 remain unchanged. I adds the selection parameter `task.sar.i_feature_selection=g_s`, scoring a maximum of 10 from all non-zero features after filtering to the standard answers after a separate zero. Users clearly choose to relax the filter full at less than 10, using `task.sar.i_g_s_shortfall=relax_filters`: first-frequency filtering, followed by semantic/density filtering, if necessary, to record the actual rules. I filtering uses a standard answer, which is recorded in the behavioral identity.

[Four Configuration Contrast Sources](../../experiment/MMLUPro/llama31_8b/capture/score_ranked_interactions/launch_evidence/reports/session/comparison_sources.json),[New I Run progress](../../experiment/MMLUPro/llama31_8b/capture/score_ranked_interactions/launch_evidence/reports/session/progress_zh.md).tmux:saebase-mmlu-i-gs.


## MMLLU-Pro / Llama:g(S) from Whiten 100 candidate Top10

Current 500 Question, Llama 3.1 8B,L21,References 100 and seed42 Hold on. Just... g_s Model adopted `task.capture.g_s.candidate_count=100`:Press Whiten Absolutely. z-score The primary, and then the zero-sum answer. g(S),Outputs 10 One.. Whiten,Non-whiten And before that, by all non-zero features g(S)Candidature New I direct reuse;old g_s The result is preserved. Original answers are updated Capture Record, with existing response caches entered.

## Current MMLU-Pro / Llama: I Unified with SAR

2026-11-19 Users rephrase: I and SAR have been semantically filtered to exclude 50% of the original non-zero-activation ratio and ⁇ 50% of the reference > > ⁇ 1 ⁇ , and have taken 10 candidates in absolute ⁇ -score; I no longer opts for g(S) and does not relax the filter. The density orientation is clearly confirmed by the user, following the original rule that excludes universal activation features. I public default selection is changed to whiten, and historical g_s configurations retain their original meaning. This time using the 100th question selected by soeed 42 from the 500 questions, the reference question is the same as 100. I run out of 1024, SAR uses a full I rating of 384 sets, 256 is set in line with the original paper sample of 128, recalculated with a common proposed function, and the additional model is zero; the source group Hashi retains the reordering of the active and filtering records directly from the Non-whiten round, and g (S) uses the Whiten 100 candidate to avoid historical standardization.

[Implementation record](../code/maintenance/sar_method_comparison/execution.md) · [Configure and Original Evidence Archive](../../assets/history_experiment/maintenance_evidence/files/sar_method_comparison.tar.gz) · [Saved Status](../../assets/history_experiment/maintenance_evidence/files/sar_method_comparison/status.json).

## Current MMLU-Pro / Llama: SAR uses 32 Whiten candidates

Users then requested that only SAR be reruned and the number of candidates be changed to 32. The title remains the 100 above, with reference to 100 questions, and the rating of filters, models, SAEs, layers, seeds and standard answers remains unchanged. 32 candidates have 496 pairs, 529 regression parameters; the current round uses 1536 sample combinations, of which 1024 are proposed, 512 are set aside, p = 0.5, and output pair numbers are maintained at 10.

[Implementation record](../code/maintenance/sar_candidate_count/execution.md) · [effective configuration](../../assets/history_experiment/maintenance_evidence/files/sar_candidate_count/config.yaml) · [Saved Status](../../assets/history_experiment/maintenance_evidence/files/sar_candidate_count/status.json).The official product and completion list is located at [SAR 32 Can not open message](../../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/).

[Original Start Script and Pre-Check Evidence](../../assets/history_experiment/maintenance_evidence/files/whiten_candidate_scores.tar.gz) · [Four Configuration Sources](../../experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates/launch_evidence/reports/session/comparison_sources.json) · [Run Status](../../experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates/launch_evidence/reports/session/status.json).Saves the official result on `experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates`.
