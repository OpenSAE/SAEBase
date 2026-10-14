# `saebase/sae/` — SAE evaluation

Unified SAE infrastructure and Hydra task (`task.name=sae`).

## Profiles

| sae key | model | layers | hook | loader |
|---------|-------|--------|------|--------|
| `pythia70m_resid_sm` | pythia70m | 1,3,5 | `hook_resid_post` | sae_lens |
| `pythia410m_mlp_topk` | pythia410m | 4,12,20 | `hook_mlp_out` | eleutherai_topk_mlp |
| `gemma270m_scope_resid` | gemma270m | 5,9,12 | `hook_resid_post` | sae_lens |
| `gemma1b_scope_resid` | gemma1b | 7,13,17 | `hook_resid_post` | sae_lens |
| `gemma1b_pt_l25_transfer_big` | gemma1b (+IT target) | 13,25 | `hook_resid_post` | sae_lens |
| `gemma1b_pt_l25_transfer_small` | gemma1b (+IT target) | 25 | `hook_resid_post` | sae_lens |
| `gemma1b_it_l25_native_big` | gemma1b_it | 25 | `hook_resid_post` | sae_lens |
| `r1_distill_qwen15b_resid_l20` | r1_distill_qwen15b | 20 | `hook_resid_post` | eleutherai_topk_mlp |
| `r1_distill_llama8b_llamascope_l23` | r1_distill_llama8b | 23 | `hook_resid_post` | sae_lens |
| `gemma2_2b_pt_l25_canonical` | gemma2_2b_base (+IT) | 13,25 | `hook_resid_post` | sae_lens |
| `gemma3_4b_pt_scope_resid` | gemma3_4b_pt | 9,17,22,29 | `hook_resid_post` | sae_lens |
| `gemma3_4b_it_scope_resid` | gemma3_4b_it | 9,17,22,29 | `hook_resid_post` | sae_lens |
| `llama31_8b_lxr_8x` | llama31_8b | 8,16,21,27 | `hook_resid_post` | sae_lens |
| `qwen3_8b_resid_w64k` | qwen3_8b | 9,18,23,31 | `hook_resid_post` | qwen_resid_topk |
| `gemma2_9b_pt_scope_resid` | gemma2_9b_pt | 9,20,31 | `hook_resid_post` | sae_lens |
| `gemma2_9b_it_scope_resid` | gemma2_9b_it | 9,20,31 | `hook_resid_post` | sae_lens |
| `gemma3_12b_pt_scope_resid` | gemma3_12b_pt | 12,24,31,41 | `hook_resid_post` | sae_lens |
| `gemma3_12b_it_scope_resid` | gemma3_12b_it | 12,24,31,41 | `hook_resid_post` | sae_lens |
| `qwen35_9b_resid_w64k` | qwen35_9b | 8,16,21,27 | `hook_resid_post` | qwen_resid_topk |
| `gemma3_27b_pt_scope_resid` | gemma3_27b_pt | 16,31,40,53 | `hook_resid_post` | sae_lens |
| `gemma3_27b_it_scope_resid` | gemma3_27b_it | 16,31,40,53 | `hook_resid_post` | sae_lens |
| `qwen35_27b_resid_w80k` | qwen35_27b | 16,32,42,54 | `hook_resid_post` | qwen_resid_topk |

`sae=none` is a registry placeholder; `task=sae` rejects it at config validation.
Quantized model checkpoints are allowed. Layer/id details: [`doc/00_model_sae_info/00_model_sae_info.md`](../../doc/00_model_sae_info/00_model_sae_info.md).
Standalone Phase A registers `sae` only. Other protocol tasks live outside this package.

## Evaluations

- `activation_health` — last-token hook activation + SAE encode/decode contracts

## Commands

```bash
python scripts/check_env.py --mode analysis
python3 -m saebase.sae.verify_registry --list
python3 -m saebase.run \
  prompt_source=simple_copying model=pythia70m sae=pythia70m_resid_sm \
  sae.layer=3 evaluation.n_prompts=5
```
