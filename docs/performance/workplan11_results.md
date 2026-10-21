# Work Plan 11 results

Base: Work Plan 10 `bc9b1fb4638a01be19756d508dc0d1c89277e0b2` (PR #45, CI green, not merged).
This wave did not start from stale `main`.

## What is faster

Opt-in block-suffix replay on Hugging Face `model.model.layers` stacks. One clean forward keeps the full block output. Each later edit runs only the blocks after that site. The original hook path stays the default.

Measured on one RTX 5070, Qwen3-1.7B Base, residual TopK SAE, layer 7, bf16, 8 public prompts, 7 edits per prompt, 1 warmup + 5 timed repeats. Timers cover the edit forwards only, after the shared clean capture. CUDA synchronized.

| Path | Five repeats (seconds) | Median |
|---|---|---:|
| Original `run_with_patch` | 0.47772, 0.47773, 0.47774, 0.47780, 0.47797 | 0.47774 |
| Suffix replay | 0.37870, 0.37900, 0.37909, 0.38079, 0.40510 | 0.37909 |

Median speedup: **1.260x**. Peak allocated memory during the run: 4.25 GB. Peak reserved: 4.26 GB. The suffix p95 repeat (0.405 s) is still below every full-hook repeat.

Parity on the same prompts: identity, ablate, scale, clamp, add, coalition, reconstruction, and an edit at token 0 with readout at the last token. Activation and logit max abs error **0**. Nonzero SAE feature-support flips **0**. Tolerance used for the gate remains 1e-2; nothing was relaxed.

The 1.15x gate passes for this workload, so the opt-in stays. It is not the default: Pythia and Gemma were not remeasured, and an untested family must keep the original hook.

## What stayed the same

- Public SHA-256 still hashes current bytes. The trusted lease is not a fresh checksum.
- ResultWriter, OOM splitting, and the disabled fused bf16 batch are unchanged.
- SAE encode was not retuned. Work Plan 09 already measured it at about 0.013 s against multi-second model time.

## What failed or is blocked

| Item | Result |
|---|---|
| Incremental clean-prefix KV steering | **BLOCKED**. The edit must read clean earlier positions and must not write an edited residual into the cache that produces the next token. The installed Qwen cache does not expose a verified split between those histories. Each steered step still recomputes the clean prefix. A suffix-only edit of one clean prefix matches the full patch on the CPU toy model. |
| Persistent edited-history KV | Not implemented. It is a different intervention, not a faster form of the current one. |
| vLLM / interp-engine | **NOT_REFERENCE_EQUIVALENT**. vLLM is not installed in this environment. Work Plan 09 already showed residual mismatch from block 0. No newer pin was adopted, and the failed path was not rerun. |
| W1 capture throughput | Not re-benchmarked. This wave does not change the capture kernel except to retain the block output when suffix replay is explicitly enabled. |

## Stop rule

The large edit-forward opportunity was implemented, measured, and left opt-in after it passed parity and 1.15x on the defined Qwen layer-7 workload. The KV and vLLM paths were attempted only as far as a correct design allowed and are disabled. No new stable release is cut from this branch.
