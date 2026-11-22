# Primary Research Configuration

Recorded on 2026-11-15. By agreement with the user, “primary research configuration” refers to the models, SAEs, datasets, and experimental conventions in this note.

## Models and SAEs

This scope comes from the ten models with existing whiten or non_whiten capture results. The table retains the historical capture layers. The current SAR focus uses **L21** of Llama 3.1 8B.

| Model | Project configuration name | Weight identity | SAE configuration key | Historical capture layer | Saved Neuronpedia explanations |
| --- | --- | --- | --- | --- | --- |
| GPT-2 Small | `gpt2small` | `openai-community/gpt2` (loaded as `gpt2-small`) | `gpt2small_res_jb_l8` | 8 | Available |
| Gemma 2 2B Base | `gemma2_2b_base` | `google/gemma-2-2b` | `gemma2_2b_pt_l25_canonical` | 25 | Available |
| Gemma 3 1B PT | `gemma1b` | `google/gemma-3-1b-pt` | `gemma1b_scope_resid` | 13 | Available, with some features missing |
| Gemma 3 4B PT | `gemma3_4b_pt` | `google/gemma-3-4b-pt` | `gemma3_4b_pt_scope_resid` | 17 | Available, with some features missing |
| Gemma 3 12B IT | `gemma3_12b_it` | `google/gemma-3-12b-it` | `gemma3_12b_it_scope_resid` | 24 | Unavailable |
| Llama 3.1 8B Base | `llama31_8b` | `meta-llama/Llama-3.1-8B` | `llama31_8b_lxr_8x` | 16, with current default **21** | Available for both L16 and L21 |
| Qwen3 1.7B Base | `qwen3_1_7b_base` | `Qwen/Qwen3-1.7B-Base` | `qwen3_1_7b_base_resid_w32k_l0_100` | 14 | Unavailable |
| Qwen3 8B Base | `qwen3_8b` | `Qwen/Qwen3-8B-Base` | `qwen3_8b_resid_w64k` | 18 | Unavailable |
| Qwen3.5 9B Base | `qwen35_9b` | `Qwen/Qwen3.5-9B-Base` | `qwen35_9b_resid_w64k` | 16 | Unavailable |
| DeepSeek-R1-Distill-Llama-8B | `r1_distill_llama8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `r1_distill_llama8b_llamascope_l23` | 23 | Available |

“Available” and “Unavailable” refer to explanation coverage for these SAEs and the saved results. When explanations are required, choose one of the six models marked available. The historical Qwen3.5 9B non_whiten results have samples, while whiten has no valid samples.

Use the layers explicitly specified in the historical batch. For example, Gemma 3 4B PT historically used L17, while its current general SAE preset uses L29. To reproduce the historical run, specify L17 explicitly. The historical Qwen3 1.7B SAE key is no longer a named preset in the current SAE catalog. The model still registers that profile. The current named preset is `qwen3_1_7b_base_resid_w32k`. Verify the actual SAE identity when reproducing the run.

## Datasets

| Dataset | Project configuration name | Subconfiguration where specified | Completed this round's 100-question filtering and 10-question SAR |
| --- | --- | --- | --- |
| HotpotQA | `hotpotqa` | `distractor` | Yes |
| BBH | `bbh` | — | Yes |
| IOI | `ioi` | — | — |
| RAVEL | `ravel` | `city` | — |
| CounterFact | `counterfact` | — | — |
| ARC-Challenge | `arc_challenge` | `ARC-Challenge` | Yes |
| MMLU-Pro | `mmlu_pro` | — | Yes |
| GSM8K | `gsm8k` | `main` | Yes |
| HumanEval | `humaneval` | `openai_humaneval` | — |
| TruthfulQA | `truthfulqa` | `multiple_choice` | Yes |
| HarmBench | `harmbench` | — | — |
| XSTest | `xstest` | — | — |

Historical captures used `non_whiten` and `whiten` with seed 42 and requested 500 examples per dataset. The fixed sample pools actually contained 164 examples for HumanEval, 400 for HarmBench, 450 for XSTest, and 500 for each remaining dataset, for 5514 in total. Consult saved results to determine whether an individual run succeeded.

## Current SAR Conventions

- Use model `llama31_8b`, SAE `llama31_8b_lxr_8x`, explicit layer **21**, and seed 42.
- Use **100 questions** per dataset as the reference for standardization and feature frequency filtering, with a separate parameter controlling this count. Run SAR on only **10** of those questions. Compute the mean and standard deviation for z-scores over all prompt tokens of the reference questions. Compute frequency from the final prompt token of each question.
- Before SAR, apply the semantic and native activation density filters aligned with `to_ziye`. Then exclude features for which **at least 50% of reference questions have z > 1**. The threshold and proportion are configurable. These filters apply only to SAR.
- After filtering, select the top 10 individual features by whiten score. Select the top five SAR pairs by **descending raw interaction coefficient**. Different pairs may share endpoints.
- SAR uses 384 coalitions, with 256 fitting samples and 128 held-out samples. The readout is `answer_mean_logprob`, and the answer execution backend is `auto`.
- For each question, report the full original question and options, a Chinese translation, the dataset's reference answer, the top 10 single features with Chinese explanations, raw activations, z-scores, and reference-question frequency, and the top five SAR pairs with Chinese explanations for both endpoints and signed interaction values. Include a comparison of single features and pairs.

## Configuration and Result References

Use `task.intervene.feature_selection=capture_top task.intervene.top_feature_num=10` for intervention and `task.generate.feature_selection=capture_top task.generate.top_feature_num=10` for generation. Each program runs a separate experiment on each of the ten features selected by capture for the question. Capture defaults to `task.mode=whiten`. Tasks that use the selected features default to `task.feature_ranking=whiten`, with `non_whiten` available explicitly. Use `task.capture_source` to reuse an existing capture ranking. See [task interfaces](../code/maintenance/task_log.md#top-n-%E5%8D%95%E7%89%B9%E5%BE%81%E5%B9%B2%E9%A2%84%E4%B8%8E%E7%94%9F%E6%88%90) for parameters and examples.

- [Historical model and dataset matrix](../../conf/tasks/capture/task.yaml). Llama L16 is historical. See the [model catalog](../../conf/models/catalog.yaml) and [SAE catalog](../../conf/base/interfaces/sae_profiles.yaml) for the current L21 default. See the [dataset catalog](../../conf/datasets/catalog.yaml) for dataset presets.
- [Historical capture explanation coverage](../../experiment/batches/capture/top10_explanation_coverage/coverage.json).
- [Batch configuration for five additional QA datasets](../../experiment/batches/capture/sar_qa_comparison/matrix.yaml) and [full results](../../experiment/batches/capture/sar_qa_comparison/five_datasets_full_report_zh.md).
- [Full HotpotQA results with 100 reference questions and 10 SAR questions](../../experiment/HotpotQA/llama31_8b/capture/reference_additivity/reports/full_questions_singles_pairs_zh.md).

## Current Full Intervention Batch

The full intervene and generate batch for the primary research configuration began on 2026-11-15. It covers ten models and twelve datasets. The top 10 Whiten features per question are individually zeroed, and Generate produces at most 8 new tokens. Llama uses the current L21. See the [full batch configuration](../../conf/tasks/intervene/task.yaml) and [live progress and results entry point](../../experiment/batches/intervene/whiten_feature_effects/reports/session/matrix/progress_zh.md). The background tmux session is `primary-feature-tasks-20261115`. The live status file gives progress and the full run directory.

## Historical 500-Question Capture Batch for Five Natural QA Datasets (Paused)

The user discarded the previous two-model, 100-question Capture batch and replaced it with these 80 combinations:

- Dataset order: `gsm8k`, `mmlu_pro`, `arc_challenge`, `truthfulqa`, and `bbh`. Take 500 questions from each, or all available questions if fewer.
- Model order: `llama31_8b` / `llama31_8b_lxr_8x` / L21, `qwen35_4b` / `qwen35_4b_matryoshka_l15_65k` / L15, `gpt2small` / `gpt2small_res_jb_l8` / L8, and `gemma3_4b_pt` / `gemma3_4b_pt_scope_resid` / L17.
- For each dataset and model, run `whiten`, `non_whiten`, `g_s`, and `I` in that order. Here `I` exhausts 1024 states. Regression-based `sar` is outside this batch.
- Each combination observes 500 questions and uses 500 filtering reference questions, with `filter=10`, `capture_count=10`, and seed=42. Complete all Prechecks before starting tmux.
- Use the existing Capture computation. Share fixed questions across models and modes. When public explanations are missing, retain the actual coverage status.

[Batch recipe](../../scripts/capture_recipes/natural_qa_full_cohort.yaml) · [Recipe instructions](../../scripts/capture_recipes/README.md) · [Progress and results before the pause](../../experiment/batches/capture/qa_capture_pilot/full_cohort_pilot/reports/session/progress_zh.md)


## 2026-11-18 100-Question Capture and Targeted Reruns (Completed)

The current scope is five datasets × 100 questions × four models × four configurations, for 80 combinations. The dataset and model order follows the preceding section. The four configurations are `whiten`, `non_whiten`, `g_s`, and `I`. Each combination observes 100 questions and uses 100 filtering reference questions, with `filter=10`, `capture_count=10`, and seed=42. The older 500-question batch remains paused.

The shared model numerical policy now includes stable averaging using the vLLM implementation. It preserves the original model formulas, SAE sites, and answer-score definitions. Precheck still measures batch sizes before selection and requires exact agreement in scores. The current identity is `saebase.vllm_batch_invariant.mean.v6`. Qwen also retains stable GDN accumulation with identity `saebase.vllm_batch_invariant.qwen_gdn_scan.mean.v6`. At the user's request to save time, reuse the successful v4/v5 results. Rerun only the five BBH groups blocked at precheck (500 executions) and five individually failed questions, for 505 executions. Use v6 for the new results, keeping each result's original version and provenance. The full rerun has stopped.

[Current batch configuration](../../experiment/batches/capture/normalization_fix/matrix.yaml), [Formal fix and acceptance notes](../../experiment/batches/capture/normalization_fix/%E6%AD%A3%E5%BC%8F%E4%BF%AE%E5%A4%8D%E8%AF%B4%E6%98%8E.md), [Current batch progress](../../experiment/batches/capture/qa_capture_comparison/stable_mean/reports/session/progress_zh.md).

Current recovery entry points: [manifest of missing items only](../../experiment/batches/capture/normalization_fix/repair_manifest.json), [rerun status](../../experiment/batches/capture/bbh_capture_repair/reports/session/status.json).


## Current 500-Question Capture Expansion (100 Reference Questions)

Expand each dataset from 100 to 500 questions. Keep the four models, SAEs, layers and Capture modes, filter=10, capture_count=10, seed=42, and 100-question standardization and frequency-filtering reference cohort. All five datasets have 500 available questions. The prompts and answers of the original 100 questions were verified to match. Reuse the 8000 completed executions and add only 400 questions × 5 datasets × 4 models × 4 configurations = 32000 executions.

[Configuration](../../experiment/batches/capture/qa_expansion/expansion_inputs/matrix.yaml), [Reused questions and new scope](../../experiment/batches/capture/qa_expansion/expansion_inputs/expansion_manifest.json), [Launch instructions](../../experiment/batches/capture/qa_expansion/expansion_inputs/%E6%89%A9%E5%AE%B9%E8%BF%90%E8%A1%8C%E8%AF%B4%E6%98%8E.md), [Live progress](../../experiment/batches/capture/qa_expansion/reports/session/progress_zh.md). After launch precheck passes, use tmux session saebase-capture500-expand-20261118. Run in dataset → model → configuration order.


## MMLU-Pro / Llama: Use g(S) to Select I Candidates

This run covers only the current 500 questions for this dataset and model. Reuse the original results for the other three configurations. Keep the 100 references, L21, filter10/capture_count10, and seed42. With the new I selection setting `task.sar.i_feature_selection=g_s`, rank all surviving nonzero features by reference-answer score after individually zeroing each feature, select the highest-scoring 10, and exhaust 1024 states. The user explicitly chose to relax filters when fewer than 10 remain, using `task.sar.i_g_s_shortfall=relax_filters`. Relax the frequency filter first, then the semantic and density filters if needed. Record the actual rules per question. I selection uses reference answers, and that fact is recorded in the behavioral identity. The prior default null and selection rules of the other modes retain their meanings.

[Sources for the four configurations](../../experiment/MMLUPro/llama31_8b/capture/score_ranked_interactions/launch_evidence/reports/session/comparison_sources.json), [New I run progress](../../experiment/MMLUPro/llama31_8b/capture/score_ranked_interactions/launch_evidence/reports/session/progress_zh.md). tmux: saebase-mmlu-i-gs.


## MMLU-Pro / Llama: Select g(S) Top 10 from 100 Whiten Candidates

Keep the current 500 questions, Llama 3.1 8B, L21, 100 reference questions, and seed42. Only g_s mode uses `task.capture.g_s.candidate_count=100`. First select by absolute Whiten z-score, then zero each feature in turn, compute g(S) for the reference answer, and output 10 features. Reuse Whiten, Non-whiten, and the newer I results that selected from all nonzero features using g(S). Retain the earlier g_s results. Record the original answer with the new Capture and reuse cached answers for identical inputs.

## Current MMLU-Pro / Llama: Shared Whiten Candidates for I and SAR

On 2026-11-19, the user changed the rule. Both I and SAR first apply semantic filtering, exclude features with native nonzero activation proportion ≥50% and features for which the share of reference questions with z>1 is ≥50%, then select 10 candidates by absolute z-score. I no longer selects by g(S) or relaxes filters. The user explicitly confirmed the density direction, preserving the rule that excludes widely active features. The public default for I is now whiten. Historical explicit g_s configurations retain their meaning. This run uses the standard 100 questions selected from the existing 500 with seed42, with the same 100 reference questions. I exhausts 1024 states. SAR samples 384 groups from the same full I scoring according to the original paper, with 256 for fitting and 128 held out. It recomputes using the shared fitting function, requires 0 additional model forward passes, and saves the source array hash. In the comparison report, rerank Whiten and Non-whiten using the activations and filtering records saved by this run. For g(S), reuse the results from the 100 Whiten candidates to avoid mixing historical standardization.

[Execution record](../code/maintenance/sar_method_comparison/execution.md) · [Configuration and original evidence archive](../../assets/history_experiment/maintenance_evidence/files/sar_method_comparison.tar.gz) · [Saved status](../../assets/history_experiment/maintenance_evidence/files/sar_method_comparison/status.json).

## Current MMLU-Pro / Llama: SAR with 32 Whiten Candidates

The user subsequently requested rerunning only SAR with 32 candidates. Keep the same 100 questions and 100 reference questions, as well as the filters, model, SAE, layer, seed, and reference-answer scoring. The 32 candidates yield 496 pairs and 529 regression parameters. Use 1536 sampled combinations, with 1024 for fitting and 512 held out, and p=0.5. Continue to output 10 pairs. Use the shared native batch scoring directly, without reusing responses from the 10-candidate I run. I retains the 10-candidate result.

[Execution record](../code/maintenance/sar_candidate_count/execution.md) · [Effective configuration](../../assets/history_experiment/maintenance_evidence/files/sar_candidate_count/config.yaml) · [Saved status](../../assets/history_experiment/maintenance_evidence/files/sar_candidate_count/status.json). The production artifacts and completion list are in the [SAR 32-candidate run directory](../../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/).

[Original launch scripts and precheck evidence](../../assets/history_experiment/maintenance_evidence/files/whiten_candidate_scores.tar.gz) · [Sources for the four configurations](../../experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates/launch_evidence/reports/session/comparison_sources.json) · [Run status](../../experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates/launch_evidence/reports/session/status.json). The production results are stored in `experiment/MMLUPro/llama31_8b/capture/whiten_score_candidates`.
