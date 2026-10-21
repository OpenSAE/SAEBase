# Work Plan 11 — RTX 5070 BatchEngine measurements

Generated-token HF KV and vLLM steering are a different protocol matrix.
Those 8/32/128 timings live in [workplan11_final_results.md](workplan11_final_results.md).
The 1.276× number below is only the 25×7 suffix-replay workload.

Base is Work Plan 10 `bc9b1fb`, stacked through the suffix-replay branch and the clean-KV / vLLM branch. Measurements below were taken on one RTX 5070, 12227 MiB, compute capability 12.0. HF path: torch 2.14.1+cu130, transformers 5.20.2. vLLM path: torch 2.13.0+cu130, vLLM 0.28.0, interp-engine 1.14.0, in a separate environment. Tolerances stayed frozen: bf16 atol 1e-2 rtol 1e-2. Prompt text and token ids are not in this file. CPU suite on the parent commit: 341 passed, 13 skipped.

## What is faster

`BatchEngine(..., enable_suffix_replay=True)` keeps one clean block output per prompt and reruns only the blocks after layer 7 for each edit. The default remains the original hook. A forward lock is held while those block methods are replaced.

This is the full 25-row × 7-condition workload, with the identity replay inside the timer, 1 warmup + 5 synchronized repeats, alternating nothing else on the GPU.

| Path | Median seconds | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| Original hooks | 2.409 | 4.030 GB | 4.272 GB |
| Suffix replay | 1.888 | 4.034 GB | 4.272 GB |

Speedup **1.276x**. Metric max abs **0**. Changed metric cells **0**. Identity replay passed. The five suffix repeats are 1.971, 1.880, 1.888, 1.879, and 1.958 s. The five original repeats are 2.456, 2.417, 2.409, 2.374, and 2.394 s. Suffix accounting: 25 prefix forwards and 200 suffix replays. The earlier 1.260x figure was edit-forwards only and is not this end-to-end number.

The 1.15x gate passes for this Qwen site, so the opt-in stays. It is not the default. Pythia and Gemma were not remeasured on this path.

## Clean-prefix KV steering

The prototype keeps a clean DynamicCache, deep-copies it before the edited step, and discards the edited cache. The cached prefix of the first four tokens matches a full forward at layer 0 and layer 7, max abs 0. The first new token does not.

| Layer | Activation max abs versus full forward |
| --- | ---: |
| 0 | 0.00390625 |
| 1 | 0.03125 |
| 7 | 0.25 |
| 14 | 1.0 |
| 27 | 16.0 |

Feature-support flips at layer 7: 0. Clean and edited logits max abs: 0.1875. Removing the attention mask, passing position ids, and pinning SDPA to math, flash, mem-efficient, or cuDNN does not bring layer 7 inside 1e-2. Eager attention is 0.125. Both the 8-step and 32-step runs stop at that first token, so there is no speed number. The reference stepper stays the default.

## vLLM SAE capture

Dynamic `vllm` at `gpu_memory_utilization=0.50` applies the requested residual delta inside its own forward (max abs 0). Against the HF reference at the same layer-7 site: capture max abs 0.0625, edited activation max abs 0.0625, clean logits 0.3125, edited logits 0.25, TopK support does not match. That is `BACKEND_NATIVE`, not `REFERENCE_EXACT`.

Static taps at 0.50 fail before a forward: the static buffer budget is 0 bytes after weights, graphs, and KV (8 MiB requested). At 0.70 the static forward runs. Its own edit max abs is 0.03125, and the HF gaps match the dynamic run, including the support mismatch. Neither backend is a scientific default.

## Stop rule for this hardware and these workloads

The edit-forward reuse that clears 1.15x on the defined Qwen intervention workload is implemented, parity-checked, and opt-in. Clean-prefix KV and vLLM SAE capture were run on the real checkpoint and rejected for the frozen bar. No remaining untried change on this machine has a measured Amdahl bound above 1.15x that is still scientifically equivalent: the leftover time is the model forward itself, and the two ways of skipping it change the numbers. No new stable release is cut here.
