# SAR Four Grid Search Executing Records

## Background, objectives and holistic approach

Users want to start with Llama 3.1 8B L21 HotpotQA In question 10, we found it. 00,10,01 Average token It's more than likely. 50%,Three is close, but... 11 Clearly elevated features. Search for real combinations, other features, then original. SAR Second stage for unobserved mask Make a prediction check, make a clear distinction between predictions and measurements. intervene/generate Batch to run.

## Action and judgement

- Read the original 100 screenings, 10 SAR reports and 10 NPZs, checking each original product SHA256.
- Achieving reusable CLI at `tasks/capture/pair_search.py` with parameters to control the increase in the lowest baseline probability, the extent of the baseline, and the relative maximum baseline. True four cells require the remaining feature to be fully consistent with the mask; repeat the mask take the first record and keep all duplicated line numbers for checking.
- This threshold: three baselines are strict over 50 per cent, with a maximum span of 5 percentage points,11 an increase of at least 5 percentage points over the highest baseline, and the logprob four are positive. The range is 45 pairs each of the ten characteristics originally selected for each topic.
- The original 384 clusters/problems found a total of 1280 complete background four, with zero strict hits. None of the observed averages of Token for questions 5, 7 and 8 have reached 50%. The remaining seven remain the closest to the candidate, but are not called strict hit.
- The original `paper.design` and the same 256 proposed rows are used to rebuild the second-order proxy model, with a total of 1,024 masks per subject; the forecast results are kept separately from measurements and are not used as a combination of experimental evidence.
- Software inspection: Ruff 0.15.22 passed; four product tests were passed, covering real AND shape recognition, prohibition of mixing background, 50 per cent of strict boundaries and prohibition of selection of higher repeat measurements.

## Results and follow-up recommendations

The result is at `experiment/HotpotQA/llama31_8b/capture/hotpot10_L21_ref100_freq50_sar_20261114_42/reports/pair_pattern_search_50_5_5/` and contains the full JSON candidate, the measurement line number, the background status and the Chinese report.

AI judgement: The current evidence supports some of the weaker double features that add up, and does not support a clear leap that claims to have found three flat, high baselines. The search conclusion is limited to the original Top10 and the observed combination; the second tier does not constitute an exclusion of the whole SAE or all real mask. If the candidate characteristics are extended or new reasoning is required, the measurement protocol and the results should be preserved separately, with the same background and clear thresholds maintained.

## Follow-up search for five other data sets

Following the same threshold, the original scanning portal was implemented for each of GM8K, MMLU-Pro, ARC-Challenge, TruthfulQA and BBH. 50 NPZs were checked through the original SHA256, with a total of 6,400 complete factual background four cells, with a strict hit of zero. MMLU-Pro3 and TruthfulQA8 have weak candidates with baselines that are good and are known to interact; the remaining 39 are not meeting the 50 per cent baseline condition. Appearing candidates are selected in order of 11 increases above the maximum baseline, with a requirement of I > 0, so as not to misname positive probability increase as positive logprob interaction.

Delivery As `experiment/central-maintenance/sar_fiveqa_ref100_20261114/five_datasets_pair_patterns_zh.md` Correspond JSON.Each of the five sets of questions, topics and original complete input, answers, explanations and two lines of probabilities are maintained, and the increase in both lines is calculated and validated on their respective scales..

## Re-execution for the purpose of user correction

Remove the threshold of at least 5 percentage points for non-user-designated 11; retain the baseline > 50 per cent close to the current maximum difference of 5 percentage points; 11 above the maximum baseline and measured I > 0, in order of I=g11-g10-g01+g00. The original CLI default was changed to no additional increase threshold, and the visible-gain can still be used to repeat the historical search. Re-scan 50 questions, get a four-gue of the 230-comb background, spread out to 11 questions; take the largest group of I. Show only the eligibility question and no more 39 outstanding baseline questions.

The original NPZ rating and the full mask were verified for each of the four selected, with the probability of an increase being calculated for each of the two rows, all of which were measured and not predicted by the proxy model. Six product tests were passed, and the real gain of having a new coverage of less than five percentage points was not excluded by default, and was ranked by interaction rather than probabilities. New deliveries were `experiment/central-maintenance/sar_fiveqa_ref100_20261114/five_datasets_pairs_by_interaction_zh.md` and JSON; historical reports retained their original calibre for retroactive use.

## Top10 is a comparison of the four options.

The user requests that the relevant scripts and 50 questions of this exploration be centralized at `scripts/response_surface/`. The original provisional scripts are copied to Legacy and the source is recorded, with 10 questions of the original five data sets fixed and the original filtered Top10; each of the 1024 reservations is asked to find the largest realistic I of the original values in 45 x 256 other feature settings. Compare the top g01 for a single candidate, the top absolute Zscore for all SAEs, and the highest absolute activation, with an explanation.

The new rating is based on the original BF16 TransformerLens, FP32 SAE, the end of the complete sequence, 10 layers, and the first 22 layers are executed in the form of CPU load weights. The old g must be re-checked before the original activation, full original code, and history g/complete. The status, authentication and grouping breakpoints are saved to assets/history_experiment/response_surface/files.

Three differences were identified and addressed: when CPU loaded SAE, the original one-in-one fold was given a different FP32 weight, converted to model CPU load, SAE loaded with the original CUDA, and the model and SAE SHA are now consistent with the original records; TransformerLens allocated an additional value of about 1 GB each time unembed conversion, instead of pre-positioning the original value with the same final matrix layout and limiting the budget for this process to 27%; the original NPZ exported the BF16 action to FP32, which had to be restored to BF16 before calling official edit_features, otherwise the same mask generated an approximately 0.0154g difference. The previous two rear layers of restorations were accurately aligned with the previous phases and were in the process of re-establishing historical scores after re-activation.

The first three probes corresponded to the original g, complete forward, and complete end of the context; each of them was followed by a full forward probe and three historical g probes, reducing the repetition of prefix weights for about one minute each. The automatic updating of the Chinese report was completed on a subject-by-issue basis and the cluster breakpoints could be restored. A new prepare_plan could accurately re-establish the frozen input from the original 50 records, which was fully reconciled. The first eight issues were completed through the independent 10-dimensional limited-scale maximum, all 8192 states, retention of the original value, single maximum, two probability and change points, and the Chinese interpretation check, details assets/history_experiment/response_surface/files/audit_partial.json.

Under shared GPU conditions, edit_features repeats the GPU standard check for each known limited zero replacement, creating a large amount of synchronization; the original code is also double-calculated. Only the original code decodes in the SAR explorer runner, leaving the known zero replacement combined value; each new code still passes the original SAE decode and maintains the same FP32 difference as the official API, BF16 delta and action. Each of the three probes is equal to the edit_features final patches, so that it can be activated. The different measured implementation versions are stored by source code SHA at assets/history_experiment/response_surface/files/code and maintain the original execution metadata.

User stepback confirms single is defined as “the other nine keep the same and only one clean”. The sequence has been changed to compare g (full_mask XOR forecast_bit), the baseline is original all-reservedg; no model reasoning has been added. The first 23, 23,552 states of correction have been independently checked. As the process of reasoning is already under way, the old report code has been imported, and scripts/response_surface/watch_reports.py has been continuously generated, avoiding reloading the GPU model for the reporting selection rule; 50 questions, and the reasoning process has been withdrawn, automatically writing back the final report and implementing the full audit. The new security session issar-reports-20261115.

## All completed

The final report and JSON are located at assets/history_experiment/response_surface/files/report_zh.md and report.json, all sources NPZ, statistical and filtering configuration, actual implementation source version and SHA are associated in this directory.

The maximum 50 questions range from 0.0140,7362 to 0.77165,0314; the first 50 questions of non-whiten are 18562; the first corresponds to the highest single after the deletion of g. The maximum I corresponds to 11 more than the remaining three-state topics is 19; the number of eligible pairs under the original high baseline is 11 and the additional candidate is retained in JSON.

AI comment: The results can be used directly to compare the four options; the maximum value of I should be explained in relation to the fixed background and the four-bar probabilities, and Single scores the highest answer after this time when it is clearly stated that the feature has been deleted. The current complete cumbe supports subsequent re-screening and sorting, without any need for further reasoning in the current Top10 context. When the scope is extended to new candidate features, real measurements will be added. The original post-primary-feature-tasks-20261115 matrix remains in operation, and the SAR reasoning and report sessions are normally over.

## Export in a user-simplified format

Add scripts/response_surface/build_compact_report.py, and export 50_questions_four_top1_zh.md from the final verified JSON. Each question retains only the title (Chinese and complete original option), standard answers, four Top1 explanations and selection values in Chinese, using the bullet, without probabilities and four-barrel details. 50 questions, 200 Top1 records and downloading bytes have been confirmed, and the script is actively exported through Ruff. Download copy is mnt/data/SAR_50_questions_top1_zh.md; original scientific data remains the same.

## Eleven Tables Export

Generate 50_questions_four_top1.csv (50 line titles) by 11 columns specified by the user: full title, standard answer, four Top1 features, whiten zscore, native activity, g after removal of a single feature, delta g, max I. Feature cells retain ID and Chinese explanations, leave all options in the original title, retain original value accuracy; use UTF-8 BOM and standard CSV multiline references to open it to Excel. Read back to check 50 lines, 11 columns and all values. The original Markdown SHA is unchanged and downloading copy has been checked. The output script is checked through Ruff.
