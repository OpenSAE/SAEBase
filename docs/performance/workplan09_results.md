# Work Plan 09 — RTX 5070 pilot (redacted)

Public summary of the private run `rtx5070-20261021`. Prompt text, token ids, and weights are not included. This is not a 48-site result and not a Research Lab corpus result. GPU-05 was not run.

Base `339fbb603bc6d999fd126a91ea8709fee1c599c2`. The GPU rows were measured on candidate HEAD `db8e652d818ea2ec6995ebe87b9a001d623db2bb` with the default-off change still uncommitted, then recorded in `c93cb5ca846eabdbe90d81496845c56112802ca4`. CI for `db8e652`: https://github.com/OpenSAE/SAEBase/actions/runs/39546669785. CI for `c93cb5c`: https://github.com/OpenSAE/SAEBase/actions/runs/39550371815 (offline 3.11/3.12, analysis-install, package, localization-native, and quality succeeded; research-lab-corpus-cpu skipped).

Machine: one RTX 5070, compute capability 12.0, 12227 MiB. Pilot torch 2.14.1+cu130. Tolerances fixed before the run: fp32 atol 1e-5 rtol 1e-4; bf16 atol 1e-2 rtol 1e-2. Speed bar: 1.15x median warmed throughput over 5 repetitions.

## Sites

| Site | Backend | Result |
| --- | --- | --- |
| Pythia-410M TopK MLP, layer 4, `blocks.4.hook_mlp_out`, fp32 | TransformerLens | Tensor parity PASS (0 feature flips). Identity-replay FAIL on both commits, max logit error 3.325e-4. Throughput ratio 0.987. |
| Qwen3-1.7B-Base residual TopK, layer 7, `blocks.7.hook_resid_post`, bf16 | HF hooks, serial default | PASS. 0 feature flips. Identity PASS both sides. Ratio 1.009. Peak allocated about 4.03 GB both sides. |
| Same Qwen site, masked HF batch | HF hooks, opt-in path | FAIL. Ratio 1.418, peak 4.05 GB vs 4.03 GB, 5 feature-id flips, candidate identity-replay max logit error 0.375. Flips: 9801/10708, 4862/2637, 11659/23359, 6202/18175, 26199/23288. |
| Gemma-3-270M Scope residual, layer 5 | — | BLOCKED. Gated repo returned 401. JumpReLU was not measured. |
| R1-distill Qwen-1.5B residual, layer 20, registered substitute | TransformerLens | Tensor parity PASS. Identity-replay FAIL on both commits, max logit error 2.289e-5. Ratio 1.295. This does not replace the JumpReLU cell. |

Checkpoint SHA-256 values are in the private `summary.csv`. Direct encode of the cached rows matched serial feature ids at batch sizes 1, 8, 32, 128, and 256 on every measured site. A second `encode_from_store` reused all 25 rows.

The synthetic encode microbenchmark (2048 by 1024, 32768 features) reached 4658, 33877, 128815, 244052, and 239361 rows/s at those batch sizes. That is not end-to-end evidence.

## Decision

Masked HF batching stays off by default. The serial path is the measured match to the base. Coalesced cache encoding did not change feature ids on the sites that ran.

Injected OOM retained the reduced microbatch (8/8 kept). A corrupted activation shard was rejected with `CacheIntegrityError`.

GPU-06: SAEBase full-prefix ablate on Qwen layer 7 ran (feature 185, greedy tokens unchanged, step-0 logit gap 0.375). Eager interp-engine 1.14.0 captured `resid_post` and generated 4 tokens on GPU; the residual disagreed with SAEBase at the frozen bf16 bar (max abs 0.25, relative L2 0.0138). vLLM 0.28.0 did not complete a forward: engine init reported `SM 12.x requires CUDA >= 12.9` and FlashInfer `requires GPUs with sm75 or higher`. `vllm-generate` refused capture. KV-cached SAE steering was not implemented. GPU-06 equivalence is BLOCKED.

PR #44 must not merge the masked HF batch as the default: that path fails Qwen feature identity. `c93cb5c` leaves it opt-in, and CI on that commit is green. Cross-version tensors on the serial default match the base. Full G2 is still not a pass: Pythia and R1 miss the 1e-5 identity-replay gate by the same amount on both commits, and Gemma JumpReLU did not load. GPU-06 stays BLOCKED and is not part of this PR.
