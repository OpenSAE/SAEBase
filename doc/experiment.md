# Experiment Reports

## Purpose and Main Functions

Record the research questions, conditions, results, and scope of actual experiments.

## Scope

Use this document to look up results and write new experiment reports. Run files are stored under `experiment/<dataset>/<model>/<task>/<experiment_name>/<optional_execution_role>/`. For an individual run, consult `metadata/manifest.json` for the effective configuration, status, precheck receipt, and results. For batches that span datasets or models, store progress, the member list, and combined results under `experiment/batches/<task>/<experiment_name>/<optional_batch_role>/`. A batch and its members share the experiment name. When a capture specifies a method, both use the same method directory. Build the member path from its actual identifiers. Per-question results are always in that directory's `reports/capture_samples.jsonl`, and run conditions are in `metadata/manifest.json`. Repeated executions for the same purpose share a named directory. Read the current result from `metadata/manifest.json` and use `saebase.runtime.experiment_results.list_executions` to look up past configurations and artifacts.

Use SAEBase's `saebase.runtime.experiment_results.read_run` to validate and read native runs. Formal tasks can write project-specific folders such as `metrics/` beneath their run, using the selected dataset and model keys. EverResearch's structure check follows Git's source scope and skips these ignored results. Its explicit experiment checker applies to records written in the EverResearch format.

In an SAEBase checkout, configuration-only commands such as `saebase task=capture task.execute=false` produce a separate `executions/<execution_id>/` directory for each invocation when using a built-in dataset and a single built-in model. Its `metadata/manifest.json` records the effective configuration and `reports/task_result.json`. Use `read_run` with an execution directory to read that execution or with the study directory to read the latest one. Use `list_executions` to list all invocations. See [archived local research runs](../assets/history_experiment/local_runs/README.md) for original measurements and the two restored executions.

## BBH Capture Configuration Precheck

1. **Purpose and criteria.** Check the BBH capture configuration and actual run conditions on Qwen3.5-4B using the precheck manifest and performance receipt.
2. **Run and conditions.** The run directory was named `fiveqa4models_100_meanv6_20261118_42`. The [effective configuration](../assets/history_experiment/local_runs/files/experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/config.yaml) and [manifest](../assets/history_experiment/local_runs/files/experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/manifest.json) record the settings and actual environment. The manifest identifies Python 3.12.15 and PyTorch 2.11.0+cu128.
3. **Status and artifacts.** The manifest status is `executed`. The same directory contains the [precheck results](../assets/history_experiment/local_runs/files/experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/precheck_result.json) and the [performance report](../assets/history_experiment/local_runs/files/experiment/BBH/qwen35_4b/precheck/qa_comparison/reports/performance_precheck.md), which records the conditions and results of the bounded performance precheck, including the batch sizes tested.
4. **Conclusion and use.** The records allow readers to review the run conditions and checks for this combination.
5. **Limitations and follow-up.** This record covers that precheck. Consult the relevant run records separately for full capture and behavioral conclusions.

## Comparison of Four Capture Methods for Natural QA (100 Questions per Dataset)

1. **Purpose and criteria.** Compare four feature selection methods based on native activation, standardized activation, changes in behavioral scores, and exhaustive interactions. Check whether answer scores remain consistent across batch sizes.
2. **Run and conditions.** The study included five datasets, four models, and four methods, for 80 combinations. Both the observation and reference samples contained 100 questions. The candidate and output counts were both 10, and the random seed was 42.
3. **Status and artifacts.** The original repair record shows that successful results were saved under v4/v5. The v6 reruns filled gaps for five BBH groups and five individual questions, for 505 executions in total. Provenance was recorded by version.
4. **Conclusion and use.** The records support comparing candidate features with the questions, models, layers, and reference cohorts held fixed. They also identify which evidence predates or follows the numerical fix.
5. **Limitations and follow-up.** The results combine reused and rerun versions. For comparisons within a version, filter responses by their recorded provenance. The [research configurations and rerun records](knowledge/primary_research_configuration.md) record the main research combinations and completed capture batches. Use them to check layer sites, question counts, and rerun coverage, including the four capture methods in the 100-question comparison.

## Expanded Capture for 500 Natural QA Questions

1. **Purpose and criteria.** Expand observation from 100 to 500 questions to see whether more questions support the original observations with the reference cohort held fixed.
2. **Run and conditions.** Use the same four models, four methods, candidate and output counts of 10, and seed 42 as in the 100-question study. The original 100 questions remain the reference for standardization and frequency filtering.
3. **Status and artifacts.** A total of 40,000 numerical observations were verified and published. This includes 8,000 reused observations from the 100-question experiment and its reruns, plus 32,000 new observations. Of the new observations, 5,418 retained a blockage marker. Both the values and explanation status are present in the published table. [Publication manifest](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation/tree/848a1ddbd8d757f4dab648678ba7b7d08afdd3c4/data/capture_qa500)
4. **Conclusion and use.** For example, open the per-question results in the member directory for BBH, GPT-2 Small, experiment name `qa_expansion`, and method `whiten`. The batch whiten summary records progress for each combination. Reused questions are registered by question ID. [New per-question results](../assets/history_experiment/local_runs/files/experiment/BBH/gpt2small/capture/qa_expansion/whiten/reports/capture_samples.jsonl) · [Whiten summary](../assets/history_experiment/local_runs/files/experiment/batches/capture/qa_expansion/whiten/reports/matrix/summary.json)
5. **Limitations and follow-up.** Reused and new observations use their respective recorded numerical versions. To filter by whether a public explanation is complete, use the fields in each observation row. [Batch provenance and effective configurations](knowledge/primary_research_configuration.md)

## Comparison of Feature Additivity Across Five Natural QA Datasets

1. **Purpose and criteria.** Determine whether the change in answer score from intervening on two features jointly can be explained by adding the effects of the individual interventions.
2. **Run and conditions.** Use fixed reference questions from five datasets and the selected model, layer, and SAE. Compare results by question, feature, and intervention condition. Use the residual of the combined prediction to represent the non-additive component.
3. **Status and artifacts.** The report records a reference cohort of 100 questions and a selection of 10 questions for SAR analysis using the layer 21 SAE for Llama 3.1 8B. For one GSM8K question, the report gives +0.01230 as the interaction coefficient of the highest-ranked feature pair. Interpret conclusions for each dataset and question within their respective coverage.
4. **Conclusion and use.** Use these results to identify feature pairs that warrant individual review and to form hypotheses for further interventions.
5. **Limitations and follow-up.** Matching feature IDs or score changes alone cannot establish a shared mechanism across models. See the [report](../assets/history_experiment/local_runs/files/experiment/batches/capture/sar_qa_comparison/five_datasets_full_report_zh.md) for actual captures and pairwise interactions on five QA datasets, including the questions, reference answers, and feature-pair coefficients for GSM8K.

## 1% Density Filtering for 32-Candidate SAR on MMLU-Pro

A historical analysis with a 5% threshold preserves the distribution plot, per-feature densities, and quantiles for Llama Scope L21 32K. [Archive of the 5% threshold analysis](../assets/history_experiment/sar_density/README.md)

The same batch of 100 questions was rerun using Llama 3.1 8B Base and Llama Scope L21 32K. The density exclusion threshold was strictly greater than 1%. For each question, 32 candidates were still selected by absolute Whiten z-score. SAR was fitted with 1,024 masks and validated with another 512 held-out masks. [Comparison-run manifest](../assets/history_experiment/local_runs/files/experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/metadata/manifest.json) · [Rerun manifest and environment](../assets/history_experiment/local_runs/files/experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/metadata/manifest.json)

The two-question automatic performance precheck passed. The full rerun completed 100/100 questions with 0 failures. The density rule excluded 399 features. After applying the combined filters, 26,857 features remained eligible for Whiten ranking. Compared with the original threshold of ≥50%, the candidate sets changed for all 100 questions, with a total of 2,415 candidate positions replaced. Compare regression coefficients in the context of the candidates and intervention combinations used for each run. [Precheck report](../assets/history_experiment/local_runs/files/experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/metadata/automatic_precheck/reports/performance_precheck.md) · [Comparison report](../assets/history_experiment/local_runs/files/experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/reports/density_comparison.md)

## 0.1% Density Filtering for 32-Candidate SAR on MMLU-Pro

1. **Purpose and criteria.** Compare how a stricter feature density threshold affects coverage of SAR results.
2. **Run and conditions.** Reuse the 100 questions, Llama 3.1 8B Base, Llama Scope layer 21 32K SAE, and regression configuration from the 1% rerun. Strictly exclude features with density above 0.1%. The snapshot contained 11,222 features that met the exclusion condition.
3. **Status and artifacts.** The run evaluated all 100 questions. SAR was defined for 76 questions and undefined for 24 because the behavioral scores had zero variance across feature combinations. There were 0 failures. The [run manifest](../assets/history_experiment/local_runs/files/experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density0p1pct/metadata/manifest.json) records processing status, the per-question ledger, and the configuration so the status of each question can be checked.
4. **Conclusion and use.** In the same batch, the 1% threshold yielded 100 defined results and the 0.1% threshold yielded 76. To compare regressions across thresholds, align the questions with defined results on both sides and report the coverage change separately.
5. **Limitations and follow-up.** These counts apply to this model, site, and question cohort. SAR is undefined for questions with zero variance.

## Five Capture Methods with 0.1% Density Filtering

Run Non-whiten, Whiten, g(S), I, and SAR on the same 10 correctly answered MMLU-Pro questions, retaining the original 100 reference questions. All five methods use the same density, semantic, and cross-question exclusion list. It excludes 12,051 features in total, 11,222 of which match the strictly greater than 0.1% density rule.

Each of the first four methods produced 10 complete results. SAR produced 8 complete results and 2 undefined results because of zero variance. Two other SAR results had held-out scoring variance below `1e-30`, as marked under their respective questions in the report. For all methods, candidates for 4 questions had zero native activation throughout. The comparison table includes native activations so these cases can be identified. [Per-question comparison of five methods and run manifest](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/comparison.md) · [Filter verification](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/verification.json)

## Comparison of Density Thresholds with Shared Active Candidates

Compare exclusion thresholds of strictly greater than 0.1% and strictly greater than 0.5% on the same 10 correctly answered MMLU-Pro questions and the same 100 reference questions. All five capture methods filter for nonzero activation on the question and share up to 10 candidates ranked by absolute z-score.

At 0.1%, 4 questions had no candidates and 1 had only one candidate. At 0.5%, the density rule matched 1,160 features, and the combined rules excluded 2,379. Each question retained 3 to 10 candidates. All five methods completed 10/10 questions with identical candidates and ordering per question. Interpret interaction coefficients in the context of the candidates and measurement method for each run. [Per-question results at 0.1%](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/active_only/comparison.md) · [Per-question results and run manifest at 0.5%](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/active_density0p5pct/comparison.md)

A rerun was also started on the same questions with a threshold of strictly greater than 1% and at most 20 shared candidates. A preliminary check with saved activations found 4 to 12 eligible candidates for each question, all within the existing exhaustive-I budget. Refer to the run manifests and batch summary for the final results. [1% launch record](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/active_density1pct/launch.json)

## Comparison of Four Capture Methods on Correctly Answered HotpotQA Questions

Use HotpotQA distractor, Llama 3.1 8B Base, and Llama Scope L21 32K. The reference set contains 100 questions. The first 40 questions were examined in run order. After normalization, the model's first answer line exactly matched the reference answer in 13 cases. Select the first 10 of those questions with eligible candidates. The run excludes features whose nonzero native activation density is strictly greater than 1%, retains only features with nonzero activation at the end of the question, and selects up to 20 by absolute z-score.

Non-whiten, Whiten, g(S), and SAR each completed 10/10 questions with the same candidates and order for each question. Each question retained 11–20 candidates. SAR used 1,536 masks, with 1,024 for fitting and 512 held out. This batch used four configurations and excluded exhaustive I. [Per-question Top 3 report](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/hotpotqa/correct_questions_top3.md) · [Per-question Top 20 report and run manifest](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/hotpotqa/correct_questions_top20.md) · [Question selection record](../assets/history_experiment/local_runs/files/experiment/batches/capture/density_comparison/hotpotqa/selection.json)

The same 10 correctly answered questions also ran through SAR with 50 unfiltered candidates. Density, semantic, and cross-question frequency filters were disabled. A fixed set of 50 candidates per question was selected by Whiten rank, allowing zero native activation. Each question used 3,072 sampled masks, with 2,048 for fitting and 1,024 held out. All 10/10 questions completed and all second-order designs reached full rank of 1,276. On an RTX 5090, mean time per question was 256.0 seconds and total time, including precheck and loading, was 3,215.6 seconds. Candidate counts and sampling budgets both changed. Interpret each run's coefficients in the context of its fitted variables. [Per-question Top 3 without filtering](../assets/history_experiment/local_runs/files/experiment/HotpotQA/llama31_8b/capture/sar/density_comparison/unfiltered_top50/reports/correct_questions_top3.md) · [Measured runtime](../assets/history_experiment/local_runs/files/experiment/HotpotQA/llama31_8b/capture/sar/density_comparison/unfiltered_top50/reports/performance.md)

The same 50 candidates per question from the unfiltered SAR run were also used to compare Non-whiten, Whiten, and g(S). The first two methods reranked the candidates using activations and reference statistics saved by SAR. g(S) completed all 10 questions. One question, “So Free Am I,” ran out of GPU memory during automatic batch scoring and passed when rerun separately at a fixed batch size of 2. All three methods were verified per question to use the same candidate set and ordering as SAR. [Per-question Top 3 for the three methods](../assets/history_experiment/local_runs/files/experiment/HotpotQA/llama31_8b/capture/g_s/density_comparison/unfiltered_top50/reports/other_modes_top3.md)

## Recorded Runs

When reading manifests created before the migration, `saebase.runtime.experiment_results.read_run` locates existing task results and automatic precheck receipts by the current run directory. The manifest retains the original run record, and the result files retain their original contents.

[Recorded runs](experiment/recorded_runs.md) lists 131 groups of records with local evidence by experiment name. It records the conditions, status, and configuration or manifest by experiment and task so the scope of the original execution can be checked. Formal research reports explain what each comparison establishes.

## Real-Checkpoint Model and SAE Site Compatibility and Evaluation

1. **Purpose and criteria.** Verify that the selected model layers can load compatible checkpoints and complete evaluation and stability checks.
2. **Run and conditions.** The project records list 18 model–SAE combinations, with model sizes from 270 million to 9 billion parameters. Evaluation used 64 held-out prompt groups.
3. **Status and artifacts.** The records show that 48 selected sites passed real-checkpoint compatibility checks and completed evaluation and reliability checks. The [verification matrix](code/documentation/source/development/verified-model-matrix.md) lists measured model and SAE sites and can be used to select combinations with compatibility evidence, such as whether a chosen layer completed the held-out evaluation.
4. **Conclusion and use.** The records identify which configurations and layers have evidence to support further research choices.
5. **Limitations and follow-up.** These results apply to the versions, layers, and held-out prompt groups in the matrix. Consult separate evidence for additional models and other sites.


## Single-Question Feature Capture for the Naming Check

1. **Purpose and criteria.** Verify automatic naming through the official launch entry point and confirm that the run name agrees with the dataset, model, method, and artifacts.
2. **Run and conditions.** Capture one IOI input on CPU using the cached GPT-2 Small and layer 8 sparse autoencoder, with ranking by native activation. The launch used the earlier naming rule, omitted the experiment name, and explicitly set the run identifier to `seed_42_name_check`.
3. **Status and artifacts.** Precheck passed and capture completed for one input. The run configuration records the generated name `capture_non_whiten`. The manifest records the sampling ledger, performance report, and precheck receipt. [Run manifest](../assets/history_experiment/local_runs/files/experiment/IOI/gpt2small/capture/non_whiten/metadata/manifest.json)
4. **Conclusion and use.** This run verifies that the entry point uses the same name in the effective configuration and the saved path.
5. **Limitations and follow-up.** This is a one-sample run check covering the actual artifacts and directory lookup.
