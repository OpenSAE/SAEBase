# Experimental report

## Purpose

(c) Recording of practical experimental problems, conditions, results and their scope of application.

## Scope

The operating document is stored at `experiment/<dataset>/<model>/<task>/<experiment_name>/<optional_role>/`. One run from `metadata/manifest.json` to the configuration, status, pre-checking back and results. The batch progress, list of members and joint results of the cross-data set or model are shown at `experiment/batches/<task>/<experiment_name>/<optional_batch_role>/`. The batch and members share the name of the experiment. The catch with the method is in the same method directory. The member directory is `reports/capture_samples.jsonl` by physical identification, and the performance result is fixed at `metadata/manifest.json`. The common name directory for repeated execution for the same purpose. The current results are read from `metadata/manifest.json`, and successive configurations and products can be consulted at `saebase.runtime.experiment_results.list_executions`.

## BBH Capture Configuration Precheck

1. ** Purpose and decision **. Check the BBH capture configuration and actual operating conditions on Qwen3.5-4B, based on pre-check lists and performance back confirmations.
2. **Operation and Conditions**.Run directory name `fiveqa4models_100_meanv6_20261118_42`. [effective configuration](../experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/config.yaml)with[List](../experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/manifest.json)Saves options and the actual environment. Python 3.12.15,PyTorch 2.11.0+cu128.
3. ** Status and product **. List status is `executed`, with [pre-test results](../experiment/BBH/qwen35_4b/precheck/qa_comparison/metadata/precheck_result.json) and [performance report](../experiment/BBH/qwen35_4b/precheck/qa_comparison/reports/performance_precheck.md)](recording of the performance conditions and results of this boundary precheck, e.g., to see which batches have been tried).
4. **Conclusions and uses**.May review the operational conditions of this portfolio and check items.
5. ** Restrictions and follow-up **.

## Comparison of the four capture methods for natural questions and answers (100 questions per data set)

1. **Purpose and judgement**.Compare primary activity, standardized activation, behavioral changes and an exhaustive selection of four characteristics, while observing consistency in the scores of answers under different batches.
2. **Operation and Conditions**.Five data sets, four models and four methods, total 80 Group. Both observations and reference samples 100 Question, number of candidates and outputs set to 10,Random Feeds 42.
3. ** state and product **. The original repair record shows that the successful results were saved at v4/v5, using v6 to complement BBH group 5 and five singles missing, 505 times.
4. ** Conclusions and uses **.
5. **Limitations and follow-up**.A mix of reuses and offsets is available. You need to screen the results by the source of the results in the version.[Research configuration and backtracking records](knowledge/primary_research_configuration.md)(Recording of major research packages and captured batches that have been implemented, suitable for cross-checking layers, questions and ranges, e.g., check 100 Four capture methods for comparison).

## Natural questions and answers 500 Questions capture amplification

1. ** Purpose and sentence **. Extend the observation range from 100 to 500 questions to check if more topics support the original observation under the fixed reference group.
2. **Operation and Conditions**.with 100 The experiment shared four models, four methods, candidates and outputs. 10,Seeds 42.We'll use the original. 100 as a reference for standardization and frequency screening.
3. The ** status and product ** have been checked and 40,000 numeric observations have been published. Of these, 8,000 repeat 100 subject experiments and runbacks, and 32,000 add amplifications. The new section has 5,418 retention of blocking marks, and values and interpretation status are on the release form. [published list](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation/tree/848a1ddbd8d757f4dab648678ba7b7d08afdd3c4/data/capture_qa500)]
4. ** Conclusions and uses **. For example, the issue-by-issue results of the member directory are opened directly by BBH, GPT-2 Small, experimental name `qa_expansion` and method `whiten`. The batch albinization summary records the progress of the combinations, re-numbering the title by title id. [Adds the issue-by-issue result](../experiment/BBH/gpt2small/capture/qa_expansion/whiten/reports/capture_samples.jsonl)] [Balinization summary](../experiment/batches/capture/qa_expansion/whiten/reports/matrix/summary.json)]
5. **Limitations and follow-up**.Reuses and expands using the respective numerical versions recorded. When you have completed the screening with a public explanation, the field of the observation line is used..[Source by batch and actual configuration](knowledge/primary_research_configuration.md)

## Qualitative comparison of the five natural question and answer data sets

1. ** Purpose and sentence **. When examining joint intervention of one characteristic and another, the change in the answer fraction can be explained by a separate intervention.
2. ** Runs and conditions**. Runs with the selected model, layer and SAE according to the fixed reference of the five data sets. Results represent the non-additional portion using the balance of the combination forecast against the title, characteristics and intervention conditions.
3. ** Status and product **. The report records an analysis of SAR with 100 entitled Reference Groups and 10 selected questions, using Llama 3.1 8B level 21 SAE. The first feature of the ranking given in the report is an example of GSM 8K with an interactive coefficient of +0.01230. The conclusions of the different data sets and topics are explained by their respective coverage.
4. ** Conclusions and uses **.
5. **Limitations and follow-up**.The same mechanism for cross-modeling cannot be inferred from the same or fractional change in the feature number.[Report](../experiment/batches/capture/sar_qa_comparison/five_datasets_full_report_zh.md)(Document the actual capture and pairing of five question-and-answer data sets, e.g., view GSM8K Question, reference answers and characterization logarithm).

## 1% density filter for MMLLU-Pro 32 candidate SAR

An additional 5% historical screening analysis of the threshold, which retains the distribution, feature by feature density and fraction of Llama Spope L21 32K. [5%] Threshold analysis archive](../assets/history_experiment/sar_density/README.md)

Same one. 100 It's for the dojo. Llama 3.1 8B Base and Llama Scope L21 32K Run again. Density exclusion threshold is set to be more strict than 1%.Every single one of them is still on. Whiten Absolutely. z-score Filter 32 Candidatures 1,024 We're gonna need a mask. SAR,And with another one. 512 Team stay for validation..[Check-list](../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/metadata/manifest.json) · [Rerun List and Environment](../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/metadata/manifest.json)

The two automatic performance pre-checks were passed and the formal runback completed 100/100 , 0 failed. The density rule excluded 399 features, with 26,857 eligible to participate in the Whiten rankings after a combination of filtering. Compared to the original threshold of 50 per cent for ⁇ , the pool of 100 candidates was changed to replace 2,415 candidate positions. The regression factor should be compared with each candidate and intervention group. [Previous inspection report](../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/metadata/automatic_precheck/reports/performance_precheck.md)] [Compact report](../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/reports/density_comparison.md)]

## 0.1% density filter for MMLLU-Pro 32 candidate SAR

1. ** Purpose and sentence **. More stringent characterization density thresholds affect SAR outcome coverage.
2. ** Runs and conditions**. Follow the 1% runback 100 questions, Llama 3.1 8B Base, Llama Spope 21st Floor 32K SAE and regression configuration. Strictly excludes features with a density exceeding 0.1%, with 11,222 features in the snapshot meeting the exclusion conditions.
3. **Status and products**.Run all assessed 100 Question, 76 It's defined. SAR Value, 24 The topic is not defined because of the difference in the number of behavioral scores of zero under the different combinations, 0 Question Failed.[Runlist](../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density0p1pct/metadata/manifest.json)Record processing status, issue-by-issue book and configuration, from which the status of the issues can be reviewed.
4. **Conclusions and uses**.Under the same title, 1% Thresholds obtained 100 A defined result., 0.1% Thresholds obtained 76 individual. When cross-threshold comparisons of regression results, the results should be aligned to each side of the theme, with a separate column covering the change.
5. ** Limitations and follow-up **. This set of numbers applies to the model, bits and subject groups. The Zero-Equal Zone SAR statistics are not defined.

## 0.1% Five capture methods under density filter

Runs Non-whiten, Whiten, g(S), I and SAR for the same set of 10 MMLU-Pro questions and follows the original 100-track reference. The same density, semantics and cross-cutting exclusion list is used in five ways, excluding 12,051 features, of which 11,222 were strictly > 0.1%.

The first four are different. 10 Full result. SAR Yeah. 8 Full result and 2 Zero difference undefined result. Two more SAR Less than the scoring margin `1e-30`,The report is marked under the corresponding headings. 4 There's no candidate for original activation of the subject, but it's kept for identification..[Question-by-Question 5 Check and Run List](../experiment/batches/capture/density_comparison/comparison.md) · [Filter Check](../experiment/batches/capture/density_comparison/verification.json)

## Density threshold against shared active candidate

In the same series of 10 MMLU-Pro answers, in the same set of 100 reference questions, there are two exclusion thresholds that are more strict than 0.1% and more than 0.5%. The five capture methods are screened on the basis of the non-zero-activated theme and the absolute z-score receives up to 10 common candidates.

Under 0.1% there are no candidates for 4 and only 1 for 1. Under 0.5%, the density rule has 1,160 features, excluding a total of 2,379. For each of the remaining 3 to 10 candidates, each of the five ways to complete the 10/10 question, each of which is consistent in order. The interaction factor should be read in conjunction with each candidate and measurement. [0.1%] The issue-by-issue outcome](../experiment/batches/capture/density_comparison/active_only/comparison.md) [0.5%] and the operational list](../experiment/batches/capture/density_comparison/active_density0p5pct/comparison.md)

The same set of questions has been started with a strict threshold of more than 1 per cent and a maximum of 20 common candidates. The pre-checking of each set of questions with a saved activation of between 4 and 12 eligible candidates can be carried out within the existing I budget. The official results are based on the list of moded operations and a summary of batches.[1 per cent] The start-up record](../experiment/batches/capture/density_comparison/active_density1pct/launch.json)

## HotpotQA Four Capture Contrasts for Matching

Use HotpotQA distractor, Llama 3.1 8B Base and Llama Spope L21 32K. The reference collection contains 100 questions, with the first 40 questions in the run order, 13 questions in the first line of the model and the first 10 questions with a qualified candidate after the standard answer has been standardized. The official candidate excludes the non-zero-activated density of the original population strictly above 1%, retaining only the non-zero-activated and taking up to 20 at the end of the question in absolute z-score.

Non-whiten, Whiten, g(S) and SAR each completed 10/10 topics, each with a consistent list of candidates and sequences, each of which was reserved for 11-20. SAR uses the 1,536 mask, 1024 grouping, 512 set aside. Four configurations are used in this instalment and I is not included. [Top 3 report](../experiment/batches/capture/density_comparison/hotpotqa/correct_questions_top3.md)] [Top 20] issue-by-issue report and operational list](../experiment/batches/capture/density_comparison/hotpotqa/correct_questions_top20.md)](../experiment/batches/capture/density_comparison/hotpotqa/selection.json) [option record <x03]

Same one. 10 The answer to the question is unfiltered. 50 Candidates SAR.Close density, semantics and cross-question frequency filter, press Whiten Fixed selection of rankings 50 Candidates, allow Zero Native Activation. 3,072 Group Mask, 2,048 Grouping, 1,024 Stay in group.. 10/10 It's complete. It's full of second-stage designs. 1,276.RTX 5090 On average 256.0 Seconds, including pre-check and load totals 3,215.6 sec. The size of the candidate and the budget for the sampling changes at the same time. The two-wheel coefficients should be explained in conjunction with the respective proposed variables..[No Filter-by-Place Top 3](../experiment/HotpotQA/llama31_8b/capture/sar/density_comparison/unfiltered_top50/reports/correct_questions_top3.md) · [Time taken from actual measurements](../experiment/HotpotQA/llama31_8b/capture/sar/density_comparison/unfiltered_top50/reports/performance.md)

The same group of 50 candidates without filtering SAR is also used for the comparison of Non-whiten, Whiten and g(S). The first two are reordered by the activation and reference statistics saved by SAR. g(S) completes 10 questions, of which So Free Am I has an automatic batch rating that is deficient and a fixed batch of 2 has been successfully measured separately. On a question-by-issue basis, three methods are checked with SAR using the same candidate pool and sequence. [Top 3](../experiment/HotpotQA/llama31_8b/capture/g_s/density_comparison/unfiltered_top50/reports/other_modes_top3.md) in three methods]

## Recorded Runs

When you read the list created before the migration, `saebase.runtime.experiment_results.read_run` locates the existing task results in the current active directory and automatically pre-checks back. The list maintains the original running record and the outcome document itself remains as it is.

[Recorded Runs](experiment/recorded_runs.md)(Register stored configurations, operating conditions and sources of results by experiment and task, for review of the scope of implementation at the time) by laboratory name 131 The official study explains the conclusions of the actual comparison..

## Real model compatible and assessed with SAE bits

1. **Purpose and judgement**.Verify whether the selected model layer can be loaded at the matching point and complete the assessment and stability check.
2. ** Runs and conditions**. The project information lists 18 models combined with SAEs, with models ranging from 270 million to 9 billion q.
3. ** Status and product **. 48 selected sites have been recorded as having completed the assessment and reliability check through physical checkpoint compatibility. [Certification matrix](code/documentation/source/development/verified-model-matrix.md) (aggregating measured models and SAE points, suitable for selecting a combination of compatible evidence, e.g., to check whether the selected layer has been completed and left for evaluation]
4. ** Conclusions and uses **. Recorded configurations and layers can be found as a basis for further study of options.
5. ** Limits and follow-up **. This result applies to the version given in the matrix, the layer and the prompting group.


## Queue characterization in naming check

1. **Purpose and judgement**.Validate automatic naming from the official start-up entry and confirm that the operational name corresponds to the data set, model, method and product.
2. ** Runs and conditions**. Captures an IOI input on CPU with a cached GPT-2 Small and an 8-layer Sludge Encoder, sorting it by origin. At that time, the start-up uses the old name rule, omits the name of the experiment and displays the running mark `seed_42_name_check`.
3. **Status and products**.Pre-check passed. Capture complete. 1 bar. The name that automatically generates the running configuration record `capture_non_whiten`.Checklist registration sample book, performance report and pre-checkback.[This operational list](../experiment/IOI/gpt2small/capture/non_whiten/metadata/manifest.json)
4. ** Conclusions and uses **. This validation portal can use the same name as the valid configuration and landing address.
5. ** Limits and follow-up **. This is a running inspection of a sample, covering the actual product and the catalog.
