# Feature Effects and Context Lab

Both live workspaces open on preserved measurements. Feature Effects now opens on a new, separate next-word measurement. The older six score-only pairs stay in the closed details.

## What is published

The archived six pairs and `GPU-04/generation_v2.json` still have no aligned token probabilities. A bounded follow-up used weights already on disk: Pythia 410M at layer 12 and Gemma 1B at layer 13, four project-authored prompts each (8 prompt-site pairs). Each prompt records the original next-word list and three edits of the strongest active feature: remove it, cut its activity in half, and add 1. Chances are a full-vocabulary softmax. The landing example is Gemma 1B on “The capital of France is”: removing the feature changes the top guess from “a” (23.0%) to “Paris” (35.9%). Cutting the activity in half leaves “a” first. This run id is `effects-authored-20261021`, separate from the six saved pairs.

Context Lab opens on the project-authored six-factor grid. All 64 states match sparse codes for `pythia410m:pythia410m_mlp_topk:L12`. The observed support is 55 dictionary feature ids. Marginal effects are the mean of the 32 paired on-minus-off differences.

Input Transformations uses the position-corrected RSS store, 3,000 parent files. Defined setting rows: 8,965. Undefined supports: 35, all `mistral7b_instruct` `local_format`. Those 35 stay out of the means and are counted in the denominator. The older `GPU-06/rss/*.jsonl` files have no position flag and were not pooled. Invalid JSONL files were not treated as measurements.

| Site | local_format defined/undefined | instruction_wrapper | context_wrapper |
| --- | --- | --- | --- |
| qwen3_1_7b_base | 500 / 0 | 500 / 0 | 500 / 0 |
| gemma1b | 500 / 0 | 500 / 0 | 500 / 0 |
| gemma3_4b_pt | 500 / 0 | 500 / 0 | 500 / 0 |
| gemma3_4b_it | 500 / 0 | 500 / 0 | 500 / 0 |
| llama31_8b | 500 / 0 | 500 / 0 | 500 / 0 |
| mistral7b_instruct | 465 / 35 | 500 / 0 | 500 / 0 |

External prompt text and token ids were not published. The 64 controlled prompts are the project-authored grid. RSS rows are browsed by source category, opaque sample id, Mean RSS, held-out rate, and support feature ids. `semantic_review` remains false. The label on the page is "proposed transformation".

Feature Atlas content is unchanged (`features.json` `f3bc09f9e9ee6528`). The intervention button does not open a different prompt: none of the Atlas examples share prompt identity with the six pairs, so `effect-links.json` is empty.

## Checks

`apps/research-lab/tests/test_export_real_cases.py`: 3 passed. The new test rebuilds each displayed probability from the token logit and the full-vocabulary log-sum-exp. Chromium on desktop and a 390px viewport opened Effects on the France example, switched the edit, and still found 64 context cells. The live host showed the same page with no console errors and no failed assets. Context Lab’s controlled grid was not redesigned.

## Commits and URLs

- Repository commit: `7378ccc350f6b2e2bacc7f8b01e610190e630565` on `work-plan-06-real-evidence`
- Space commit: `179bc506602e5d82c7121a2922d8c84c8c409d19`
- Feature Effects: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp06b#effects
- Context Lab: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp06b#context
- Feature Atlas: https://opensae-saebase.static.hf.space/index.html?v=wp06b#play
- Space page: https://huggingface.co/spaces/OpenSAE/SAEBase

Reused: the six score-only pairs, the 64-state grid, and the position-corrected RSS aggregates. Newly measured: eight authored prompt-site pairs and their next-word lists.
