# SAR explores scripts and results

Purpose: To centralize this Llama 3.1 8B, Llama-Scope L21 SAR explore scripts, original measurements and a thematic comparison in Chinese. The range is GSM8K, MMLU-Pro, ARC-Callenge, TruthfulQA, BBH, each of 10 questions, following the statistics of each of the 100 reference topics and the SAR-specific filters.

## Official Capture entrance

Filtering, g(S) and exhaustive interaction have been linked to official Capture, with parameters defined in [Capture Note](../../tasks/capture/README.md).`mode=sar` corresponds to the original regression method,`mode=I`. The new operation is using `run_capture.py`, which follows the original experiment's data set configuration and model/SAE, implemented through official Runtime and automatic pre-checking.

```bash
# Default Filter 10 One candidate, only one to delete. g Top single
python -m scripts.response_surface.run_capture --dataset GSM8K --mode g_s --filter 10 --capture-count 1
# Same candidate pool with the largest output after full count. I It's... pair
python -m scripts.response_surface.run_capture --dataset GSM8K --mode I --filter 10 --capture-count 1 --tag capture_I_top1
# Revert all old tables SAE whiten / non-whiten Top1;The title of reference remains: 100 Dow.
python -m scripts.response_surface.run_capture --dataset GSM8K --mode whiten --filter all --capture-count 1 --tag capture_whiten_all_top1
python -m scripts.response_surface.run_capture --dataset GSM8K --mode non_whiten --filter all --capture-count 1 --tag capture_raw_all_top1
# Use existing 50 It's true. cube Check formal Capture Candidatures, four. Top1 And original values, no re-inference.
python -m scripts.response_surface.check_capture_reproduction
```

`--questions` controls the actual number of questions individually, default 10; `--check-config` only check the configuration. The operating product uses the standard `experiment/<dataset>/<model>/capture/` path, and the old `results/` report continues. The default filer=10 will allow wihten and non-whiten to be filtered; both of the old reports use filter=all, so when the old table is reproduced, you have to set it clearly in the way above.

The four-digit active script ranking and the single ranking are directly called at `tasks.capture.behavior` and the full context end-level rating at `tasks.capture.answer`; `pair_search.py` is also forwarded to official Capture. Historical source snapshots remain at `legacy/` and `results/code/`.

## View Results

- `results/report_zh.md`: Full title, answer, max I pair, max g01 single, whiten and non-whiten first, Chinese explanation and values.
- `results/report_ablation_zh.md` and the same name JSON: Users confirm in their calculations that single is a real-time version after "only zero"; `watch_reports.py` is independently updated and synchronized to the final report.
- `results/report.json`: Structured data for the same content, with the full ranking of the maximum I for the remaining eight full-reserved, and the top I for the top 10 singles for the same baseline.
- `results/by_dataset/<dataset_directory>/<question_id>/coalitions.npz`: All measurements g of 1024 combinations, with feature_ids, masks, historical tags and answers to token.
- Everything. `verification.json`,`complete.json`,`measurements.jsonl`:Re-validation, Results SHA,Group Breakpoints and Sources.
- `source_sar.npz` for each subject: a copy of the original file of the original 384 SAR measurements, checked with the original SHA.
- `results/reference/`, `reference_sources.json`: The five data sets are now mean/std and the filtering configuration copy and source SHA.
- `results/plan.json`: Freeze original title, complete input, original candidate, source product SHA, model and SAE identity.
- `results/execution.json`, `results/logs/exhaustive.log`: Run protocol and log.
- `results/audit.json`: Review the results of the examination of the maximum I, full mask coverage, original measurements, single and probability using independent ten-dimensional differentials.

- `results/exports/` Save Four Top 1 It's... CSV and streamline the report in Chinese. `results/by_dataset/gsm8k/GSM8K-01/`.

## caliber

For each of the 10 candidates originally selected, all 1024 reservations/zero status status; other SAE features and re-constructing debris are maintained at original values. For 45 pairs, 256 background items are counted for each of the remaining eight items, calculated `I=g11-g10-g01+g00`, with the original value ** maximum **. Not all of the 32768 DSAE characteristics are paired, nor are the proposed SAR II coefficient.

Single is defined by the user-confirmed definition: only one candidate is cleared, the remaining nine entries and features other than Top10 remain the same, with the highest g01 after zero. The probability changes relative to the status of all original reservations. Structured data also saves better single in the maximum I context and avoids mixing values from different backgrounds.

Whiten is the highest absolute zscore for all SAEs, non-whiten is the highest absolute original activity for all SAEs, and does not apply SAR-specific filters; reports keep an additional SAR-filtered first place. zscore uses mean/std for all prompt token, std=0 excluded, no positive std lower limit.

g The standard answer is the average token log probabilities; the average token probabilities are exp(g) and the answer combined probabilities are exp(Tg), excluding EOS. The explanation given by Neuronpedia from the original SAE, which is translated in Chinese, cannot be based solely on interpretation to assert causal meaning.

The original 384 times/problems have 324 different states (including total retention); each issue has 700 additional tests. Old values are used only if the model is consistent with SAE weight SHA, the original activation is consistent with native codes, the end layer calculation and completeness is consistent with the historical g. The original BF16 model, FP32 SAE, FP32 log_softmax and FP64 are used. The first 22 floors are activated between GPUs and offloading rights to CPUs, and the cache complete sequence; each mask calculates the last 10 layers of the full context without using KV approximation or proxy models.

Only the original code decodes in this exploration runner have been cached, with known limited non-replacement combinations; the new code is still rounded by original SAE decode, with two-step BF16s `edit_features`. Each probe must be equal to the formal interface patch. Different executive versions are saved by source code SHA at `results/code/`.

## Run and run again.

Execute from a warehouse root directory:

```bash
python -m scripts.response_surface.prepare_plan
python -m tasks.capture.scoring.response_surface_reproduction
python -m scripts.response_surface.build_exhaustive_report
python -m scripts.response_surface.verify_results
```

`exhaustive` Runs in clusters by each subject; `--limit 1` may first verify the title. The completed subject does not repeat the measurement, and the report is automatically updated for each completed item. The report entry also allows for the generation of partially completed reports on a stand-alone basis, with a clear indication of the number of completed topics. `verify_results` Requires the completion of 50 questions. Backstage runs with the tmux session `sar-exhaustive-20261115`, which was the intervene/generate matrix session, to be held independently.

Current calculation period single We've got a caliber confirmation. We're running independently.`python -m scripts.response_surface.watch_reports`Update confirmed reports in real time; tmux Session is`sar-reports-20261115`,Log As`results/logs/report_builder.log`.The script's on 50 Finishing the subject, writing the final report after the reasoning process has been withdrawn and performing a complete check.

## History Script

`legacy/` Retain scriptshots of previous batches of the five data sets, the Chinese report, the four-bard mode search report and the original SAR algorithm; `pair_search.py` for existing survey access. `script_sources.json` Record the original path. History scripts keep the original path syntax for retroactive use only; this entry is for the three above-mentioned modules and the results are integrated into `results/` of this directory.

The official Capture realization is still at `tasks/capture/`, and model scoring and characterization editing is reused at `base/interfaces/`; here the user-specified exploratory script is saved.
