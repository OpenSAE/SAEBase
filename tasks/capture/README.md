# Capture screening, number of observations and answer scores

## Purpose and scope of application

Describes the common selection, output and results of five end-word capture modes. The running entrance is `saebase task=capture`, configured at `conf/tasks/capture/task.yaml`, and the results are read at `CaptureTask.read_samples` and `export_report`.

## Parameters

- `task.capture.selection.candidate_policy=active_whiten`: The five models share the same process. First, the terminologies, primary active density and reference frequency filters should be used, and then the current end phrases should be kept as non-zero-activated and reference standard deviations greater than zero, sorted in absolute z-score.
- `task.capture.selection.candidate_count=20`: The maximum number of candidates. Use the actual number when it is insufficient. Zero candidates keep the original answer and filtering record and have been omitted from the Markdown and HTML reports. I or SAR records the undefined reasons for less than two candidates.
- `task.capture.selection.candidate_count=all`: Skip the above filter and use all SAE features. Whitan still excludes std=0.
- `task.capture.g_s.candidate_count=100`: Independent primary range for replicating g(S) under `candidate_policy=legacy`.
- `task.capture.output.feature_count=null`: 10 by default for the new formulation of g_s, with the remaining mode default output numbers corresponding to all available results at the time of filter; filter=all. The number of a single feature mode cannot exceed the number of candidates for the number filter. The SAR output is the number of pairs, which exceeds the number of candidates, up to the number of different pairs.
- `task.capture.selection.cohort_filter.reference_sample_count=100`: Number of filtering references, independent of the number of candidate/output features.
- `dataset.sample_count`: Number of actual observation topics maintained in the previous configuration of Dataset.

The semantic/density rule is repeated at `tasks/capture/sar_filter.py`; mean/std uses all prompt token, the frequency rule is marked as z at the end of the reference title, and the default z>1 is excluded when it reaches 50% of the subject. Filter details can still be set in the tsk.capture.select.feature_filter, cohort_filter. std=0 is excluded without a positive std lower limit.

## Capture Mode

- `mode=whiten`: Presentation by the shared candidate in absolute z-score.
- `mode=non_whiten`: Reorder in the same candidate pool by the absolute active natural value.
- `mode=g_s`: A candidate in the Zero Pool each, with the rest of the characteristics remaining in the original value. The average token logarithmic probability drops in the deleted standard answer. `delta_g` minus the original rating for the deleted rating.
- `mode=I`: All candidates remain on the list, calculating I =g11-g10-g01+g00 for all fixed backgrounds of each pair running through the other candidates, with the original value down after each maximum I. Ten candidates correspond to 1024 states, 45-x256 background. Mask, g, background and source are saved to NPZ and result JSON.

The g(S) and I mode defaults the standard answers to test forcing without EOS; the change is the activation of the last token of the original prompt at the selected SAE position and the retention of the original refactor's defects. The lower layer of the `answer_backend=full_context_tail` cache complete sequence is activated, the tail layer is calculated using the same cascading/sequence shape and checked with the original whole front; the back end of the tail layer interface is not supported with the complete forward.

Use of the original SAR sample regression method `mode=sar` requires a visible and credible sar.py; its pair coefficient is two indicators, as is the count of four quartz I. The default upper limit of 16 for the number of candidates, the too large pool of candidates will clearly report errors prior to the measurement, and the return prediction cannot be used as an all-time basis. `filter=all` The same ceiling is applied to the large SAE ' s count of SARs.

Captures set the candidate pool separately from the output number. Interventions and the generation of filter and output configurations that can be fed into the capture are used to reuse the same candidate selection. History configurations are read from their original number fields. `task.capture.output_scope=all_tokens` Saves word-by-word trails, uses `task.capture.selection.candidate_count=all` and controls the number of features saved by `task.capture.max_active_per_token`.

## use

```bash
# Default not exceeding primary active density 1% Most filtered features 20 One. Whiten Features
python -m saebase.runtime.train task=capture task.execute=true
# Same candidate pool, output only g(S)The highest one.
python -m saebase.runtime.train task=capture task.execute=true task.capture.mode=g_s task.capture.selection.candidate_count=10 task.capture.output.feature_count=1
# Do not filter, output original activation Number one.
python -m saebase.runtime.train task=capture task.execute=true task.capture.mode=non_whiten task.capture.selection.candidate_count=all task.capture.output.feature_count=1
```

Models and data sets are still selected through standard model, dataset configurations. The fixed title of the five previous QA data sets and the Llama L21 configuration are reproduced using `scripts/response_surface/run_capture.py`; 50 issues have been reviewed offline `scripts/response_surface/check_capture_reproduction.py`.

Reruns the historical candidate policy with `task.capture.selection.candidate_policy=legacy` and following the previously run filtered overlay. The default density rule excludes features that are stricter than 1% and can be adjusted by `task.capture.selection.feature_filter.min_native_nonzero_fraction` and `task.capture.selection.feature_filter.density_strict`.

SAR The model measurement and the saved scoring re-calls`tasks.capture.response_surface.fit_responses`Compatibility and sequencing, and continued use of original papers internally`paper.sar`with`paper.design`.When the same question, the same candidate, the same standard answer, the same weight and the integrity of the numerical agreement already exist I, you can rate it as it is.`paper.coalitions`Draws the corresponding scores to be combined and avoids duplication of forwards; repeater is responsible for preserving the results of the original series of Hashi, the character sequence and the complete status check, and clearly marks the derivatives.

When SAR uses a standard answer rating and opens performance, sample combinations repeat the original editor of g(S)/I, state weighting, tail cache and autobatch rating; sample scheme and original regression are maintained. For each question `sar.performance` save selection and rating time, use the original standard measure when closing. n_fit must cover at least `1+K+K*(K-1)/2` regression parameters; K=32 requires at least 529 groups to be combined.

When running the batch `python -m tasks.capture.matrix`, the matrix configuration `matrix.dataset_overrides` is combined into each data set preset. For example `dataset_overrides: {sample_count: 100, sampling_seed: 42}` sets the grids of this batch to 100 questions without changing the public default of the data set. The mode is specified by `--capture-mode`.

Official Capture Default Open `task.performance.enabled=true`.Precheck First, do the authentic performance probe for the corresponding mode. Executor selects the tail layer cache and state according to the precision constraint. batch;`performance.enabled=false` Keeps the original amount. Each item takes time and is recorded separately at the sharing stage of standardization, screening, interpretation, forwarding, ranking, writing, etc.. I Save complete after reset. 1024 Status response; extra recording of genuinely executed independence code Number of states. For specific budgets, reports and value thresholds, see `tasks/precheck/README.md` "Performance precheck and auto-modification"”.

## RSS: Fixed Target and Added Context

Sets `task.capture.mode=rss`, `task.capture.selection.candidate_count=all`, `task.capture.output_scope=last_token` to provide `task.prompt`, `task.capture.rss.contexts` and a trusted local original version `task.capture.rss.source`. The default RSS parameter is `M=400, K=20, rho=kappa=0.8, tau=1e-5`. Each varant is `context + space + clean prompt` to read the full original SAE codes on the last token target.

`rss_case.md` Provides a full selection of RSS and values focused on variant's explanation candidate; `rss_case.json` Saves original explanation, recurrence, direction, exclusion cause; `rss_activations.npz` Saves the full clean, variants and delta. Shows the candidacy requirement recurrence < rho, positive change > tau, at least five times more than other variables, and maintains, in order of magnitude of change, all characteristics that meet the value requirements.

This is a temporary complete entry of five distinct contexts: `python scripts/stable_support/run_contrast.py`. The script is based on the RSS Open Demo GPT-2 loading settings, which generate each case by `CaptureTask.run` and output to the new `experiment/authored_contexts/` directory.


## & Under Old Policy I Press g(S) to select a candidate

The following settings are used for `task.capture.selection.candidate_policy=legacy`.

`task.capture.i.feature_selection=g_s`Change only I Candidature source: After a separate clean-up of all non-zero features that are filtered g(S)Before remove`task.capture.selection.candidate_count`One, and then a few. K It's a signature. 2^K State. Use the actual number when the candidate is short and record`sar.preselection.selected_count`.g(S)The selection uses standard data set answers. This fact is saved with behavioural identity. Capture And return. SAR Follow original Sorting; I Current Default`whiten`,Visible`null`Inherit common candidate sorting configuration.


When? I It's... g(S)When the number of candidate pools is not up,`task.capture.i.g_s_shortfall=available`(Use actual number by default;`relax_filters`First, the reference group frequency filter on the subject is relaxed on an ad hoc basis, then the semantic and primary density filter is relaxed when it is still insufficient, and the remaining non-zero features are pressed. g(S)Quantity. The original filtering configuration and other topics are maintained; the original number of candidates for each issue, the actual scoring pool, and`relaxed_filters`Save in`sar.preselection.pool_filtering`.Return specific errors when all non-zero features are still insufficient and keep non-zero bounds.


## Each Capture Auto-System Original Answer

All task modes (`non_whiten`, `whiten`, `g_s`, `sar`, `I`, `rss`) and `all_tokens` observations are accompanied by original model answers without feature intervention. Enter the full original tip and actual token of this capture, with no separate chat templates or standard answers. Directly repeat the upstream generation method of the loaded model and KV Cache; respond with a greedy decode.g(S) and I still rate the standard answers to the original definition.

- `task.capture.model_answer.max_new_tokens=256`: Default generation of up to 256 new characters.
- `task.capture.model_answer.batch_size=8`: Generate by the same input length, with a maximum of eight inputs.
- Each `result.model_answer` saves `text`, generates `token_ids`, `status`, `finish_reason`, `truncated`, the actual weight and hint token identity, and the cache recovery. The single input results save the same field directly. RSS saves `variant_model_answers` for each context variant.
- Saves the generated text when the word or model context cap is reached and marks the cut; this is an original continuation and has not been automatically extracted from the final option or rating. Generates failed to record errors individually, preserves the successful capture feature and marks the response stage as incomplete.
- The data set detail is still at `reports/capture_samples.jsonl`; a summary of `stages.model_answer` reflects whether the response phase has been completed. JSON, Markdown and HTML feature reports read model answers.
- Caches are located on item `data/_shared/model_answers/` and are reused by actual model weight, tokenizer, precise input token, numerical policy and production conditions; not related to mode, SAE layer and feature ranking.

Capture is available to supplement only model answers and to keep the original feature and the interactive file unchanged:

```python
from tasks.capture.task import CaptureTask

capture = CaptureTask()
capture.refresh_model_answers("/absolute/path/to/capture_run")
rows = list(capture.read_samples("/absolute/path/to/capture_run"))
```

The additional results are written in `reports/capture_model_answers.json` and `metadata/capture_model_answer_update.json`, and the official reading access is automatically merged. When you do so, check the actual weights and the original input identity. When the new experiment is completed at the feature stage and the answer is not completed, you can use `runtime.mode=resume` only for the original configuration; the above-mentioned supplementary interface is used when the historical operation cannot continue directly due to a change in source identity.


## & Independent primary range for g(S) under old strategy

```bash
python -m saebase.runtime.train task=capture task.execute=true task.capture.mode=g_s task.capture.selection.candidate_policy=legacy task.capture.g_s.candidate_count=100 task.capture.output.feature_count=10
```

`task.capture.g_s.candidate_count` controls the Whiten primary range n for positive integers; final output continues to be controlled by `task.capture.output.feature_count`, omitting the current mode of output of 10, with no greater than n. Adopts the existing Whiten filtering and standard deviation rules, including keeping the original active candidate 0 and z-score valid. For each candidate, it is only zero and all the remaining features remain the same; preserves the full single feature score and displays Top10 in descending order by g(S). `g_s.preselection` and measures the identity record Whiten primary selection method, request n and actual candidate.

This setting only affects the g_s mode. I still uses its own candidate strategy. History does not state that the formulation of the g_s-specific section is read by the shared candidate profile that is saved; old and new candidates are distinguished in the running identity, and the old test results remain in their original calibre.
