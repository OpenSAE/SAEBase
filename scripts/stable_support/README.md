# RSS temporary case story

Fixed target, with five publicly available demo FACTS in front of it, to observe the common RSS response features and variant specific features. Using GPT-2 Small, `gpt2small_res_jb_l8`, `blocks.8.hook_resid_pre`, token to extract the complete SAE codes in the last target.

Run from the SAEBase root directory:

```bash
python scripts/stable_support/run_case_study.py
```

The default runs all 24 open demo prompts, output to `experiment/authored_contexts/<UTC_timestamp>/`. `--output <new_directory>` Specifies the saving position, `--device cpu` with CPU, `--limit 1` to see only the first case. The directory must not yet exist to preserve the old result.

- `index.md`: All case entry points.
- `case_XX/rss_case.md`: RSS and variant-specific tables, with five real change values and a Neuronpedia link.
- `case_XX/rss_case.json`: prompts, token IDs, models and SAE identities, parameters, selections, activation values and explanations.
- `case_XX/rss_activations.npz`: `clean`(F), `variants` (5xF), `delta` (5xF) complete original codes and change matrix.
- `neuronpedia.json`: Open interpretation of all selected features and actual access status.
- `run.json`: Run parameters, source and completion status.

The model uses `HookedTransformer.from_pretrained_no_processing("gpt2", center_writing_weights=True)` for open demo, SAE loads, activates extraction and explains reuse SAEBASE API; actually loads settings to write results.

RSS directly uses open warehouse `rss.py`, default `M=400, K=20, rho=kappa=0.8, tau=1e-5`. `inputs/source.json` Saves upstream file identity, `LICENSE` is upstream Apache 2.0 license. Prompts `inputs.py` and FACTS are from the same warehouse `assets/demo/demo.py`.

Variant-specific rules: Remove the feature > 1e-5 from 100 before the absolute change in the variant and at least five times the maximum absolute change for the other four variants, excluding RSS support, with a maximum of three for each variant. Keep an empty set when there are no eligible features. The five-fold rule is used only for the case presentation.

The table is output-based, and Neuronpedia explains keeping the original English text. The semantic attribution of the target or new context needs to be judged by the result. All cases are saved; activating the pre-set, explaining the acquisition failure, can also continue to see the real changes.

## Second round: five distinct contexts

```bash
python scripts/stable_support/run_contrast.py
python scripts/stable_support/run_contrast.py --sae-key gpt2small_fs_24576_l8
```

Run the banana bread and run the remaining 23 open demo prompts; support `--limit 1` and `--output <new_directory>`. Fixed basketball, astronauts, violins, chess, African elephants are executed in the context of the new access to `CaptureTask`/ `mode=rss`. The model still uses open demo GPT-2 loading settings. The context numbers run through the complete dictionary, < 0.8, the current positive change > tau, at least five times the maximum absolute change, and retain all of the current changes and obtain an explanation.

The current round of SAEs and Feature Splitting SAE each have 24 cases, each of which are catalogues `experiment/authored_contexts/gpt2small/capture/rss/distinct_contexts` and `experiment/authored_contexts/gpt2small/capture/rss/feature_splitting`. The two true tables and the results of the case selection are the former `contrast_report.md`. The report clearly records that banana bread has three target-related, two-way-changing RSS features in Feature Splitting SAE, but has not yet found five eligible contextual features in the same case.

Read the report from the current round with the results already available:

```bash
python scripts/stable_support/analysis/review_contrast.py \
  --original experiment/authored_contexts/gpt2small/capture/rss/distinct_contexts \
  --alternative experiment/authored_contexts/gpt2small/capture/rss/feature_splitting \
  --output /tmp/rss_contrast_review
```

The keyword `explanation_review.json` is used only for navigation reading; the RSS selection is determined entirely by the original selector on the basis of activation. The `case_selection.json` record shows what is actually being done and what is missing. The report does not treat the expected interpretation match as an observation.

## Full candidate found: Llama 3.1 8B, player 21

The final true homogeneity results are shown at `experiment/authored_contexts/llama31_8b/capture/rss/banana_recipe_adaptation/case_report.md`, showing three goal-related, two-way changes in RSS characteristics, and five RSS-excluded features, each of which corresponds to the new background, recurrence = 0.2. Full input and 8 display rows are saved `selected_case.json`; all codes, 20 RSS support and all numerical candidates remain.

This case is based on `Adapt a banana bread recipe to this setting.`, with new basketball, space, classical music, chess, and African elephant background. The goal is to recast the formulation according to setting, which is a background-related request and should not be described as the first round of unrelated experiments.

Recurring (`--output` for new directories):

```bash
python scripts/stable_support/run_contrast.py \
  --model-key llama31_8b --sae-key llama31_8b_lxr_8x --layer 21 \
  --prompts-json assets/reproduction_inputs/stable_support/files/final_case_prompts.json \
  --contexts-json assets/reproduction_inputs/stable_support/files/final_case_contexts.json --limit 1 \
  --output experiment/authored_contexts/llama31_8b/capture/rss/llama_complete_case_rerun
python scripts/stable_support/analysis/export_final_case.py experiment/authored_contexts/llama31_8b/capture/rss/llama_complete_case_rerun
```

`search_llama.py` is the provisional search script that is used to explore background and request on the 16th, 21st floors; all 96 exploration cases are stored at `experiment/authored_contexts/llama31_8b/capture/rss/prompt_context_search`. The final complete group of five variants is recaptured and all display lines are read from the same group results. `run_contrast.py` Supports `--model-key`, `--layer`, `--prompts-json` and `--contexts-json` are available in the current directory.

## Eight cases in which the term "literal-input " was designated

Complete report: `experiment/authored_contexts/llama31_8b/capture/rss/qualitative_comparison/summary.md`. Eight original sets of inputs are at `qualitative_batch_inputs.json`, the list of actual requests is at `qualitative_batch_prompts.json`, and the AI reading judgement is at `qualitative_batch_review.json`.

```bash
python scripts/stable_support/run_contrast.py \
  --model-key llama31_8b --sae-key llama31_8b_lxr_8x --layer 21 \
  --prompts-json assets/reproduction_inputs/stable_support/files/qualitative_batch_prompts.json \
  --contexts-json assets/reproduction_inputs/stable_support/files/final_case_contexts.json --limit 8 \
  --output experiment/authored_contexts/llama31_8b/capture/rss/eight_case_batch_rerun
python scripts/stable_support/analysis/batch_candidates.py experiment/authored_contexts/llama31_8b/capture/rss/eight_case_batch_rerun
python scripts/stable_support/build_qualitative_batch.py experiment/authored_contexts/llama31_8b/capture/rss/eight_case_batch_rerun
```

`batch_candidates.py` Exports unselected features of the full dictionary RSS statistics and all five-fold value conditions from the complete codes, saves the specific exclusion reasons, and reads the explanation after determination of the ID. It retains the candidate for recurrence who is qualified but excluded for other conditions, so that it can be judged on a case-by-case basis in accordance with the requirements of this instalment. `build_qualitative_batch.py` consolidates the original input, the true features and clearly documented AI reading judgement into a table and eight comparative summaries for each case. The original 20 RSS and all the values are stored in each case at `all_candidates.md` and JSON, where the gap is clearly shown when there is a lack of a clear semantic match.

Eight cases of this instalment have actually been implemented, 48 entries have been captured at the end of the period (token ID 13) and 146 different characterizations have been obtained. Cases have not yet been identified under fixed input that fully meet the three conditions for display. C02 has game scores, space matches, C06 has space; the rest is 0. C06, C01, C02 is the preferred manual review trail, and it is ultimately decided whether to use the author to review the results and this time no final rebuttal has been drafted.

## E1-E4 diagnostic case

The full input of the four diagnostic experiments (retain a line break inside the annex) is at `diagnostic_inputs.json`. The results are compared to the four groups at `experiment/authored_contexts/llama31_8b/capture/rss/misleading_context_diagnostics`, `comparison.md`, each of the E1-E4 directories is `case_report.md`, with a complete display of all 20 RSS:clean values, five deltas, recurrence, direction, original interpretation and URLs, followed by a discussion of up to three candidate relationships.

```bash
python scripts/stable_support/run_diagnostics.py --output experiment/authored_contexts/llama31_8b/capture/rss/diagnostics_rerun
python scripts/stable_support/build_diagnostics.py experiment/authored_contexts/llama31_8b/capture/rss/diagnostics_rerun
```

`diagnostic_review.json` For all four completed AQLs. Scripts do not treat additions as better, do not spell unrelated labels as a mechanism to transfer, do not generate model answers or carry out quantitative assessments. Retaining a literal line change means that the target's layout is not exactly the same as that of clean and variants, and this is clearly stated in the report.

Four reports for this cycle, ZIP: `experiment/authored_contexts/llama31_8b/capture/rss/historical_reports/reports/rss_diagnostic_four_reports.zip`, include the Markdown report for E1-E4 only. Full Raw code, delta, RSS selection, original explanation, token IDs, input and configuration are left in the laboratory directory.

## Strict semantics maintained: six input and double line separations fixed

Enter: `semantic_preservation_inputs.json`, from the complete new scheme provided by the user. Target sentence and five short prefixes are fixed; each variant is strictly prefixed + two line breaks + the original target sentence.

```bash
python scripts/stable_support/run_semantic_preservation.py --output experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation_rerun
python scripts/stable_support/build_semantic_preservation.py experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation_rerun
```

The current actual results are at `experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation/` and the directory contains the seven requested documents: main report, original result JSON, full codes/delta, all 20 RSS forms, strict background list, config.yaml and Run.json. `rss_case.md` has only one consolidated table and none of the matching items is clearly empty; ZIP is in the parent directory and contains only three Markdown reports.

Background value candidate strictly from the original RSS h Front 400 Ike, ask Ben. variant It's changing.>tau,The other four.|delta|<=tau;No previous five-fold rule or off-site candidate. The original explanation is obtained after the value set. None of the current ten candidates supports the desired background syntax., RSS Select Only F8961 A two-way combination of broad food labels with no clear cooking characteristics. Qualitative judgement is recorded in `semantic_preservation_review.json`,Do not modify input to pursue expected results.

## A rich, independent, factual version

A new version of the complete entry is saved at `semantic_preservation_rich_inputs.json`, with the background details of During... separated from the same standard banana bread problem in two rows, without the scene fit command.

```bash
python scripts/stable_support/run_semantic_preservation.py \
  --inputs assets/reproduction_inputs/stable_support/files/semantic_preservation_rich_inputs.json \
  --output experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation_rich_rerun
python scripts/stable_support/build_semantic_preservation.py experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation_rich_rerun \
  --review assets/reproduction_inputs/stable_support/files/semantic_preservation_rich_review.json
```

The actual new measurements are kept at `experiment/authored_contexts/llama31_8b/capture/rss/semantic_preservation/rich_context` and the old version remains the same. The new RSS selections only have broad food-related recommended labels and continue to decline; there are no clear two-way groups for cooking. Strict background candidates [4,1,1,3,6], 14 have been identified with explanations that do not clearly match the background, and F19953 has no open explanation and is not known. Full 20 RSS, 15 candidates, 34 of 35 IDs have original interpretations and 1 missing, all active and aligned records are kept.

## Short target issue A/B independent comparison

Enter `banana_ab_inputs.json`: A reserves the first sentence of The scene is, and B only deletes the first sentence; the rest of the text is the same and connects to the same short cooking problem in a single space. Effectively run only one pair, one catch, five varians and RSS independently.

```bash
python scripts/stable_support/run_banana_ab.py --output experiment/authored_contexts/llama31_8b/capture/rss/banana_ab_rerun
python scripts/stable_support/build_banana_ab.py experiment/authored_contexts/llama31_8b/capture/rss/banana_ab_rerun
```

GPU When there is a limited amount to add `--prefix-offload`:Yes. CPU Load weight, embed layer, blocks0–21 with SAE Move To GPU,Follow-up not implemented CPU;Existing Capture Formal interface in 21 Stops after the layer, and the relevant calculations are still used GPU& & Precision.

Actual results: `experiment/authored_contexts/llama31_8b/capture/rss/scene_prompt_comparison/comparison.md`. Both groups selected three cooking-related features for F207000 and F12293/F9190 reduction, with five varians in exactly the same direction. B also shows only F3044, r=2 for the Chess variant and is excluded because recurrence is insufficient; A background confirms 0/5, B for 1/5. The complete 20 RSS, all of the top 400 candidates, strict monovariate candidates, explanations and exclusions are saved in the group files.

Report ZIP: `experiment/authored_contexts/llama31_8b/capture/rss/historical_reports/reports/rss_banana_ab_reports.zip`, containing only A and B reports and comparative summaries.
