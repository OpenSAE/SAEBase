# Stage D final summary

Updated: 2026-10-16T03:16:28.667338+00:00

## Counts

- Catalog sites: 48
- Pair verified (incl. prior): 48
- New GPU pair passes: 44
- Benchmark verified (this run smoke_only or prior): 20
- Reliability verified: 20
- Status histogram: {"pair_verified": 28, "reliability_verified": 17, "skipped_prior": 3}

## Hardware

- GPU: RTX 5090 (~31.4 GiB)
- Torch: 2.11.0+cu128
- Protocol: BF16/FP16 full checkpoints, 3-prompt pair acceptance, smoke_only M3/M4 for representative sites

## Notable loader fixes

- Qwen3-8B / Qwen3.5-4B / Qwen3.5-9B: native HF residual hooks
- Mistral-7B: bf16 load; registry hook corrected to `hook_mlp_out` per published cfg.json; TopK layer_norm decode workaround

## Incomplete / caveats

- M3/M4 runs in this sweep are labeled `smoke_only` (3 prompts), not 64-sample population audits
- Prior CPU pair sites (Pythia L4, Qwen3-1.7B-Base L7, R1-Qwen L20) reused; Qwen3.5-2B L12 revalidated on GPU (D1)
- Public PR contains compact JSON only; no weights or tokens
