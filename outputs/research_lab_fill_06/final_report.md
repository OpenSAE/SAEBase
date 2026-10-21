# Feature Effects and Context Lab

Both live workspaces open on preserved measurements. No new GPU run was started.

## What is published

Feature Effects has six cases, one recorded prompt group on each deep site. Every case has the five stored operators `ablate`, `scale`, `add`, `clamp`, and `coalition`. The chart is logit L2 for those operators on that same prompt. The pairs do not contain token probabilities or a generated sentence, so the page does not draw either. `GPU-04/generation_v2.json` has five token ids for a different sample and was not joined.

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

`apps/research-lab/tests/test_export_real_cases.py`: 2 passed, including a byte-identical rebuild from the run directory. Chromium on desktop and a 390px viewport opened Effects, the 64-cell grid, and Input Transformations with no console errors and no failed asset requests. Keyboard activation of a factor changed the authored prompt. Switching the operator changed the displayed logit L2 (`184.2337` to `100.0485` on the first case).

The same checks passed on the live host.

## Commits and URLs

- Repository commit: `8672ff05aa3ea4d1bb961558bc1307540df9b871` on `work-plan-06-real-evidence`
- Space commit: `ce09617cf3c5c0209cebd156a628ca627a2c21cc`
- Feature Effects: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp06#effects
- Context Lab: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp06#context
- Feature Atlas: https://opensae-saebase.static.hf.space/index.html?v=wp06#play
- Space page: https://huggingface.co/spaces/OpenSAE/SAEBase

Optional GPU follow-up was not run. The within-prompt operator chart is the recorded result.
