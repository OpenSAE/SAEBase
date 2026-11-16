# Work Plan 11 — generated-token steering on one RTX 5070

These measurements are a different set of protocols from the 25×7 block-suffix
result in [workplan11_results.md](workplan11_results.md). That 1.276× figure
stays the BatchEngine result. Nothing below replaces it, and none of these
paths is the scientific default.

Hardware: one RTX 5070, 12227 MiB, compute capability 12.0. Source SHA for the
timed runs: `4510803e6b6032caa3c708e3f1114a4e160ca82b`. HF path uses torch
2.14.1+cu130 and transformers 5.20.2. The vLLM path uses a separate
environment: torch 2.13.0+cu130, vLLM 0.28.0, interp-engine 1.14.0. Model
SHA-256 `8f952f43284eed52d505ed6e851a68e2c5bd9960209aa35a890e3ddbb34cdf4e`.
SAE SHA-256 `d68dfbf3796d0d6f1bb27fcc4c905ae3085e5c783594b4e0315065d384f194b4`.
Site: Qwen3-1.7B Base, residual layer 7, registered TopK SAE. Frozen bf16
tolerance remains atol 1e-2, rtol 1e-2. Prompt text and token ids are omitted.
Each timing cell is 3 prompts, 1 warmup, and 5 synchronized repeats.

## Capability matrix

| Protocol | Status | What the GPU run showed |
| --- | --- | --- |
| `reference_clean_prefix_stepwise_v1` | GPU measured reference | Slow full-prefix recompute at every new token. Finite 8/32/128 timings. This row is the reference, not a speedup. |
| `persistent_live_sae_kv_v1` | GPU_SPEED_PASS | One prefill, then one cached decode per token, with a fresh SAE edit each token. Identity condition matches raw cached greedy on all 3 prompts at 8 tokens. Not the same trajectory as the stepwise reference. |
| `persistent_decoder_direction_kv_v1` | GPU_SPEED_PASS | Fixed decoder-vector steering (`decode(one-hot) - decode(zeros)`, strength 2) through interp-engine `steer` + `generate_stream`, `generated=True`. Dynamic and static both ran. This is not live feature ablation. |
| `block_suffix_replay_v1` | GPU_SPEED_PASS | Opt-in only. Full 25×7 workload median 2.409 s → 1.888 s (1.276×), metric max abs 0. See the earlier results note. |
| Legacy clean-history KV (`kv_cache_exact.py`) | BLOCKED | Cached prefix matches a full forward. The first new token misses the frozen bar (layer 7 activation max abs 0.25). Not enabled. This does not mean cached generation is impossible; the live path above is a different protocol and did run. |
| One-prompt interp-engine SAE feature edit | EXPERIMENTAL | The write tap applies on a single prompt. HF residual, logits, and TopK support do not match. Not continuous generated-token ablation, and not a scientific default. |

CPU CI on `4510803` was green (quality, offline 3.11/3.12, package, analysis, localization). A later local `pytest` on this tree, with CUDA hidden, was 348 passed and 13 skipped. Neither result is the GPU evidence in this note.

## Generated-token timings

Wall time is p50 and p95 over the 15 timed repeats (3 prompts × 5). Tokens/s is the median of the three per-prompt medians. HF peaks are this process. vLLM parent-process peaks undercount the engine worker; worker startup figures from the engine log are listed under the table.

| Protocol | Tokens | p50 s | p95 s | TTFT p50 s | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF live SAE KV | 8 | 0.090 | 0.091 | 0.0115 | 89.0 |
| HF live SAE KV | 32 | 0.349 | 0.391 | 0.0114 | 91.0 |
| HF live SAE KV | 128 | 1.432 | 1.450 | 0.0115 | 88.8 |
| HF stepwise reference | 8 | 0.162 | 0.169 | 0.0202 | 49.3 |
| HF stepwise reference | 32 | 0.662 | 0.686 | 0.0201 | 48.5 |
| HF stepwise reference | 128 | 3.012 | 3.058 | 0.0204 | 42.7 |
| vLLM dynamic direction | 8 | 0.054 | 0.055 | 0.0118 | 146.7 |
| vLLM dynamic direction | 32 | 0.205 | 0.211 | 0.0130 | 156.4 |
| vLLM dynamic direction | 128 | 0.809 | 0.811 | 0.0118 | 158.3 |
| vLLM static direction | 8 | 0.055 | 0.056 | 0.0093 | 144.8 |
| vLLM static direction | 32 | 0.208 | 0.208 | 0.0095 | 154.1 |
| vLLM static direction | 128 | 0.817 | 0.818 | 0.0094 | 156.6 |

HF live peak allocated / reserved at 128 tokens: 4.029 / 4.276 GiB. HF reference: 4.070 / 4.280 GiB. vLLM dynamic worker at utilization 0.55: about 3.5 GiB weights, 0.1 GiB activation, 2.72 GiB KV. vLLM static at utilization 0.70, `max_model_len=1024`: about 3.52 GiB weights, 0.1 GiB activation, 0.28 GiB CUDA graph, 4.43 GiB KV. No OOM.

At 128 tokens the live HF path is about 2.1× the stepwise reference wall time. Dynamic vLLM is about 1.8× the live HF path. Those ratios cross protocols. Static was not faster than dynamic on this short-context greedy run.

## Validity checks

Live HF identity: for each of the 3 prompts, 8 greedy tokens with an identity SAE condition matched a raw cached decode that installed no hook. The same worker then ran the ablate timings. One prompt's ablate changed 6 of the first 8 tokens. The other two prompts kept those 8 tokens. Feature ids were the first active code at the last prompt position on each prompt (one of them is the stored HF-reference id 185). That is a real edit on one prompt, not a claim that every feature flips greedy text.

vLLM and vLLM-static, feature 185 taken from the stored HF reference capture, strength 2: prompt-capture edit max abs 0.21875 at the last prompt position on all 3 prompts, and 0 on the excluded earlier positions. The following 8 greedy tokens matched the unsteered control on every prompt. Token agreement is not treated as proof that the tap was skipped; the residual delta is the evidence the tap wrote. Static buffers fit at utilization 0.70. The earlier 0.50 static load remains a budget failure (0 bytes left, 8 MiB requested) and is not a timing.

Forward counts on the HF live path equal the number of generated tokens (the prefill is the first of those forwards). The stepwise reference still recomputes a growing prefix.

## What stays off

Suffix replay stays opt-in. Live SAE KV and decoder-direction steering stay explicit protocol calls, not `BatchEngine` defaults. The legacy clean-history prototype stays off. One-shot vLLM SAE edits stay off the HF scientific comparison. No public GPU server and no 48-site sweep.
