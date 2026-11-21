# SAEBase Task Registry

## Current capture parameters

The current command is based on the [configuration parameter](../../conf.md#433-capture-conftaskscapturetaskyaml). The page indicates the number of candidates, the number of outputs, the common screening and the unique set-up of subtasks. For example, the number of proposed samples is returned using `task.capture.sar.n_fit` and the number of candidates is increased by `task.capture.i.feature_selection`. The following records keep the configuration names and scientific conditions used at each stage.

## Capture Unified Screening and Answer Behaviour Model (2026-11-15)

The standard configuration adds a new filer = 10 and Capture_count=null: the digital filer returns using the original SAR sample; Mode=I is rounded and ranked according to the maximum values of the real four graphs of the fixed background. See tasks/capture/README.md for specific parameters, compatibility and output. The old rule below describes the historical configuration as meaning, without covering the new standard default values.

## Number of directories dedicated to maintenance

Collapse the public adaptor, the back end of the research platform, Capture rating and export, RSS duplicate scripts and product tests by duty. SAR ' s 50-track history questions are archived by data set, maintaining the original title identification; running and reporting the routing of the common subject. After this process, 6 places of over-limits were removed and the direct subdirectories of the AR ' s history catalogue were reduced from 53 to 6. Document by document, 51,200 files of 50 questions were reviewed by the official Capture and the independent verification script. Installing packages and configuration checks passed. Two SAR completed-status tests returned to the old catalogue `incomplete`, which is the same decision.

## Purpose

Describes the parameters and semantics of the results of the registration entry and current capture methods for the scientific mission.

## Scope

The actual support for each capacity is determined by the mission interface.

Official SAE operations include `capture`, `intervene`, `generate`, `circuits`, `explain`, `benchmark` and `train`.

Pre-execution inspections are provided by `precheck` and are organized in eight categories, together with seven scientific operations, Task, which is registered in `tasks/plugins.py` by `base/application/registry.py`.
`base/application/runner.py` Based on Hydra `task.name` Assigned.
`python -m saebase.runtime.train --cfg job` displays only synthetic configurations and does not load models or start calculations.
Real experiments should first be verified through mission access, the source of weights, identity, data disaggregation and execution authorization.

Capture supports `non_whiten`, `whiten` and `sar`; the number of features is specified by `task.top_feature_num`. SAR needs to have clear targets against token, original estimate path, and to preserve subset intervention response and measurement protocols.

The last token Capture judge whether the selected Top N character interpretation is complete; the other activated features are interpreted over a separate record. The additional explanation follows the same completion condition as the result readout and does not need to be recalculated.

SAR can set up `task.sar.feature_selection=whiten` to standardize the original characteristics on the selected data set and execute a subset of interventions for the same white Top N characteristics. `task.sar.top_pairs=5` Use the original version `design` and the same combination to find a second-order coefficient to preserve the first five pairs and two-end characterizations in absolute values; coefficients are not separate SAR fractions. Complete coefficients and IDs are saved to the original SAR original group product.

SAR Standard Configuration Enabled `task.sar.feature_filter.enabled=true`,Yes. Top N Candidate selection and intervention application prior to integration `to_ziye` v2 (a) Rules: exclude features of public interpretation of the command points, functional terms, natural grammaticals, general question phrases, truncated structures or blank layouts, and Neuronpedia Primary non-zero activation rate achieved. `min_native_nonzero_fraction`(Default 0.5). The programming syntax has an exception to the original rule, and the subject word and the name are retained according to the original rule. The interpretation and activation rate is the current one. SAE Source matches, and cross-layer characterizations are determined separately. The missing label or empty activation rate determines and records the quantity according to the available evidence; the measurement is stopped when the source is not available or metadata overwhelm is incomplete. Filter rules, exclusion numbers, hit cause and source fingerprints are saved to `metadata/sar_feature_filter.json` And include an identity measurement. `whiten` with `non_whiten` Keeps the previous rule; historical results keep the original protocol, keeping the old visible configuration closed when the filter is not declared, and allows the visible filter to repeat the old running.

The current SAEBase pair allows repeatends. The K32 delivery at `SAR/to_ziye` is also sorted by the absolute value of the interactive coefficient, with an additional greedy heart skipping the used endpoint to select five pairs and ten different features; each of these display constraints is maintained separately.

SAR can enable `task.sar.cohort_filter.enabled=true` to be used in conjunction with `feature_selection=whiten`. It is proposed to first align the full reminder mean/std with the last words of all selected topics, and then calculate the symbol z-score; by `z > zscore_threshold` (default 1) the feature is excluded when the proportion of the entire topic reaches `min_question_fraction` (default 0.5, including boundaries). The frequency excludes the collection from the existing semantic and primary density, and to apply it uniformly to the Top N and SAR measurements, using the same candidate pool as Pair. The rules, title identity, full threshold count, and the exclusion of ID are saved to `metadata/sar_cohort_filter.json` to measure that identity to include its fingerprints. The function maintains the independent whitten and non_whiten behaviour, the old configuration default closes; the actual gap is reported for the candidate under-time.

`task.sar.pair_ranking=signed` Displays the matching by the symbol 2nd-order interactive coefficient; `absolute` maintains the original behaviour in absolute-value descending order. The full 45 coefficients (Top N=10 hours) remain and the report indicates the actual sorting calibre.

`task.sar.cohort_filter.reference_sample_count` When the frequency filter is enabled, the reference is obtained through the original Dataset selection interface to calculate the full hint word mean/std and the super-threshold frequency of the last word of the reference; `dataset.sample_count` continues to assign the actual SAR number separately. The reference number is set as nulla. If the specified reference number is not enough, the reference selection source and the subject identity are kept with the filter record. The reference snapshot corresponds to the DAR subject by sample ID and full prompt, and the sequence changes are not mixed; the target theme is not captured independently in the reference concentration.

## Top N Single Characteristic Intervention and Generation

`intervene` and `generate` both support `task.feature_selection=capture_top`, `task.top_feature_num=10`. Use the end of Capture token Top N to start separate experiments for each feature from the same original prompt. `dataset.sample_count` Specifies the number of questions; `task.prompt` specifies that a single issue is to be executed only for that issue. The old `configured` single feature mode continues as the default.

`task.feature_ranking=non_whiten` Sorts by the original activating absolute value; `whiten` returns the full prompt token mean/std and absolute z score rankings, with reference to the currently selected sample of Dataset. Whitten of the visible question also uses the Dataset sample as a reference collection. Count only std > 0; these models follow the rules of independent Capture, and the SAR exclusive filter is only effective for SAR.

Use `task.capture_source=/absolute/path/to/capture/run` to directly reuse saved rankings; the current Dataset selected question (or visible prompt) corresponds to the full programt SHA256, checking models, SAE, layers, hook, sorting and weight prints. Saved Top N is not enough, the problem is not matched or the identity is inconsistent. Select the ranking that corresponds to the result; its existing filter selection follows the saving list as it is.

`intervene` Replaces each feature separately with `task.value` (default 0); `generate` applies `task.intervention` and `task.value` separately for each feature, e.g. `intervention=ablate` means zero, and `intervention=scale value=2`. Generates the feature ID selected for the first fixed question and perform the feature at the end of the new prefix at token at each step; each feature is generated independently from the original problem. To maintain the semantic expression, Capture_top is used separately from the custom protocol/conditions.

The official result `metadata/<task>_result.json` includes a feature-by-issue result; `reports/<task>_features.jsonl` incremental record ranking, character interpretation, activation, complete question, selection of identity and intervention/generation result. The line already written before the subsequent failure is retained in the document; it can be restored by `runtime.mode=resume` to re-use the successful line by problem, feature, sorting, weighting and enforcement protocol measurement, and re-try failure line. whiten statistics are also saved at `metadata/capture_normalization.*`, the size of the result quoted product and SHA256.

Example (Top 10 is a case-by-case item):

```bash
python -m saebase.runtime.train model=gpt2small dataset=ioi dataset.sample_count=1 task=intervene task.execute=true task.feature_selection=capture_top task.top_feature_num=10 task.feature_ranking=non_whiten task.value=0 experiment.sub_name=top10_intervene
python -m saebase.runtime.train model=gpt2small dataset=ioi dataset.sample_count=1 task=generate task.execute=true task.feature_selection=capture_top task.top_feature_num=10 task.feature_ranking=non_whiten task.intervention=ablate task.value=0 task.max_new_tokens=8 experiment.sub_name=top10_generate
```

2026-11-15 Default configuration protocol: Capture default `task.mode=whiten`; intervene and generate default `task.feature_ranking=whiten` in `capture_top` mode. Notn_whiten is visible to recreate original activation rankings; the old result retains the original agreement.

## Mainly explore the configuration of full intervention batches

`conf/tasks/intervene/task.yaml` Fixed ten models, twelve data sets and two tasks; Llama uses the current L21 and the rest uses the historical SAE layers identified in the main survey notes. Dataset still requests 500 questions, which are actually implemented by the available sample pool of the data set. Unified Whistle Top10, Zero Quantities; generate uses greed to decode up to eight new tokens.

`python -m tasks.intervene.matrix --config conf/tasks/intervene/task.yaml --report <report_directory>` executed using the official Runtime modulation interpreter to load the 12 data sets in the same period with a model/SAE. First complete the intevene phase, then executegenerate; failed second round of independent retests, successfully skip units. The title is frozen to the inputs/; verified existing Whistlecapture rankings are directly reused, and other units first implement new Capture. Each unit retains the model, SAE, layer, source capture, running directory, and cause of failure.

Use `task.inline_results=false` in a large number of times, the complete feature-by-characterization results are stored in JSONL, quoted in the official result, to avoid copying all of the resulting tracks to the master list. `tasks.intervene.top_features.read_feature_results(run_dir)` returns the latest record of each measurement. `metadata/<task>_progress.json` Provides instant progress; `task.continue_on_error=true` Keeps a record of failure and continues other independent features. Checks the identity of each measurement before recovery, and complete lines that have been interrupted at the end can be repaired; the intermediate damage is still clearly reported.

`task.reuse_prompt_observations=true` Reuses the same token prefix, SAE position and actual rewriting activation calculation between independent feature experiments on the same original issue. The reuse does not use KV history and does not change the step-by-step intervention protocol; the field of action is released at the end of each issue and a maximum of 64 remain. The 10 real GPT-2 features have been validated for intervention logits, fully generated token and step-by-step indicators are identical to independent reference calculations.

Generate Default `task.stop_at_context_limit=true`: Keeps the model as it was propt tokenization and records `context_limit` when creating prefixes beyond the model primary execution window, keeping the previously generated token. `max_new_tokens` is to generate the upper limit; the output that reaches the upper limit in the context is shorter than that value. Set false to restore the behaviour of reporting errors when exceeding the window.

The newly created self-action capture in the complete intervention batch closes the Neuronpedia completion condition and is only responsible for the verifiable numerical character; the explanation of the existing source remains as it stands. The explanation overwhelming does not prevent the execution of unexplained models intelvene/generate.

## SAR saved four-state mode search

`python -m tasks.capture.pair_search --run <existingSARrun_directory> --output <new_report_directory> --minimum 0.5 --spread 0.05 --gain 0.05` Searches for all the pairs of the original selected feature. The average token probability required for 00, 10, 01 is strictly higher than the minium, the maximum span of the three is no more than sspread,11 the increase in the relative maximum baseline is at least gain, and the logprob four is interactive. The probability is uniform as exp (average token logprob) and the margin parameter is 0 to 1 (5 percentage points in 0.05).

The four states must come from a completely identical background; repeat mask fixes the first original measurement and maintains all duplicate line numbers. The script validates the SHA256 of NPZ, returns the strict hit number, the full baseline qualified candidate and the nearest candidate for each subject, and clearly records the feature ID, explanation, background state, four g and probability, the original measurement line. `--surrogate` Use the original SAR design function and the proposed sample, perform a second-order projection check for all binary masks, which is saved separately from true measurements. This entry only reads historical experiments and does not run models.

SAR Four Search Default calibrated:`--gain` Default does not set an additional probability increase threshold. `--minimum`,`--spread` controls;selection requirements 11 Higher than the highest baseline, measured I It's right and measured. logprob Interactive I. Show each item I The maximum eligible candidate, with an average probability of two rows. token Probability combined with answer, and the last column calculates the respective. 11 Decreases the highest baseline. History is visible `--gain 0.05` Save original threshold meaning for search results.

CaptureMatrix supports `matrix.dataset_overrides` for the independent setting of the dataset parameters such as the actual number of observations in this batch; `task.sar.cohort_filter.reference_sample_count` continues to independently control the number of screening questions.

## precheck of mission performance and project execution

Precheck adds `task.mode=performance` to support the real small-scale ingestion search and numerical control of Capture, Intervene, General; standard tasks are automatically enabled by `task.performance.enabled=true`. Candidates for thebatch search include preheat/repeated measurements, OOM records and abdications, visible residuals, precision rejections and optimal measurement of candidate selections; benefits are compared to the original assigned mass baseline, with observational reuse settings. The configuration and description are available at `tasks/precheck/README.md`.

Capture uses the validated prefix prefix TL/HF, state derenewal, original decoded baseline reuse, quantitative I ranking; Intervene/Generate resets the inspected tail layer using public observations, condition batch processing and duplicate prefixes. Generates are still performed in a complete prefix by each track; precision, weight, feature selection and answer indicators remain defined. Performance JSON/Markdown separates the first correction from the actual sample time, providing time-consuming distribution, candidate speed/surveillance/ error and backloading.

Capture/Intervene/Generate BatchSystem policy for the executive level of the Precheck shared model; vllm policy is strictly consistent with rating, floating point indicators and discrete output, and failure to pass reports failure to pre-check and does not treat the scale as repair. The default watch candidate is extended to 64, Capture control state at most 64. Standard answer ratings, feature filters and I mathematical definitions are maintained, and the numerical policy is written into the operating source.

Project defaults are uniform: Normal Runtume and Matrixworker apply `runtime.cpu_threads=4` to limit the calculation line of Torch and BLAS before loading; Intervene/Generate by default start the validated observation reuse. The sub-processe environment is centralized in Runtume and results and performance reports record the actual thread pool and environment. Precision, stabilization of Attention, significant surplus, current sequence schedule and Triton calculation path continue with the validated settings. The configuration, duties and sources are available at [project default](../documentation/source/development/batch-invariant-execution.md#engineering_defaults)].

Capture answers are rated as a pre-merger cache of HF and TL; the original score is still compared to each. The answer button caches are grouped by two buckets of full sequence length and control status numbers, shared between precheck and Capture models by the same loaded pair; OOM drop-down updates the subsequent selection. The visible candidate releases the unobserved cache before peaking. Titles, filters, indicators and zero error acceptance conditions are maintained; Qwen v5 values are moved separately by model identity.

## RSS Control capture (2026-11-16)

`capture mode=rss` Accepts visible clean prompt, `task.rss.contexts` and a trusted local original `task.rss.source`. Set `filter=all`, at the end of the same end goal token to extract complete codes, directly run the original RSS selforator. RSS input only activates the margin; explains the query after the selection and numerical candidates have been selected.

Output `rss_case.json`, `rss_activations.npz`, `rss_case.md`, with the joint mission entry remaining `metadata/capture_result.json`. `scripts/stable_support/run_contrast.py` Reuse CaptureTask to run five distinct contexts, first processing banana bread and then saving the remaining 23 demo candidates.


Capture's orientation run: `task.sample_ids=[...]` only observes the designated ID, `dataset.sample_count` and `task.sar.cohort_filter.reference_sample_count` in the fixed subject group. Continue to define the full screening reference group.


I's stand-alone ranking: `task.mode=I task.sar.i_feature_selection=g_s task.filter=10`. The other models continue to use the original selection method only for I. First, the semantic/native density/reference group frequency filter is applied, the single feature is eliminated for all remaining non-zero-activated features of the topic, and up to 10 of the average token logarial probability (S) descends from the standard answer, then the list of I is exhaustive. The larger the other all features are left as they are; the larger the g(S) indicates that the answer is higher after the deletion of the feature. The actual number is retained for less than 10 times; at least two can be calculated to be interactive. `null` Inheritance the original `feature_selection` and the obvious selection of `whiten` or `native_activation` as an I candidate.

`sar.preselection` for the new I result saves the complete characterization, selected IDs, number, use of standard answers and performance records; NPZ also saves `screen_feature_ids` and `screen_deleted_responses`. `candidate_feature_ids` is the final entry to I; the behavioural record `feature_selection_uses_answer=true` allows for a distinction from the original unanswered filter. The original individual feature and I formula and the old result are read and maintained.


When? I It's... g(S)When the number of candidate pools is not up,`task.sar.i_g_s_shortfall=available`(Use actual number by default;`relax_filters`First, the reference group frequency filter on the subject is relaxed on an ad hoc basis, then the semantic and primary density filter is relaxed when it is still insufficient, and the remaining non-zero features are pressed. g(S)Quantity. The original filtering configuration and other topics are maintained; the original number of candidates for each issue, the actual scoring pool, and`relaxed_filters`Save in`sar.preselection.pool_filtering`.Return specific errors when all non-zero features are still insufficient and keep non-zero bounds.


## Capture default log the original answer to the model

All Capture mission models and token-by-token observations are stored at `model_answer`. Directly re-generated with upstream greed from the actual loading model and KV Cache, default new word caps 256, volume 8; same weight, precise input totoken, tokenizer, numerical strategy and production conditions are cross-modeled. The original answer is separated from the standard answer rating, and the g(S)/I) meaning is maintained. The answer stage fails to retain the feature and report the specific reasons separately; the old result is supplemented by `CaptureTask.refresh_model_answers()` and formally read at the entry. See [Capture Statement](../../../tasks/capture/README.md#each-capture-includes_original_model_reply) for parameters and products.


## g(S) Expansion of the Whiten primaries

g_s adds a new special purpose `task.capture.g_s.candidate_count=100`: To use the absolute z-score pren after filtration and to output a maximum of 10 at the g(S) level. The output quantity continues to be controlled by the commonoutput.feature_count, default 10 when this mode is omitted. Saves all candidate fractions independent of the primary identity, I and other modes; history does not state that the formulation in this section maintains the old selection calibre.
