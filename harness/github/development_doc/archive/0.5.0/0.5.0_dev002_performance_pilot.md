# Work Plan 09 — RTX 5070 pilot (redacted)

Public summary of the private run `rtx5070-20261021`. Prompt text, token ids, and weights are not included. This is not a 48-site result and not a Research Lab corpus result. GPU-05 was not run.

Base `339fbb603bc6d999fd126a91ea8709fee1c599c2`. The first GPU rows were measured on candidate `db8e652d818ea2ec6995ebe87b9a001d623db2bb` and recorded in `c93cb5ca846eabdbe90d81496845c56112802ca4` and `1ae89ff7e1a30b1e57886335fc8b6e7dbed4a594`. CI for `db8e652`: https://github.com/OpenSAE/SAEBase/actions/runs/39546669785. CI for `c93cb5c`: https://github.com/OpenSAE/SAEBase/actions/runs/39550371815. CI for `1ae89ff`: https://github.com/OpenSAE/SAEBase/actions/runs/39550761574. Row-wise Qwen parity and the same-width identity gate are `09bbe9c`. The hash-cache timings below were measured on the working tree after that commit.

Machine: one RTX 5070, compute capability 12.0, 12227 MiB. Pilot torch 2.14.1+cu130. vLLM comparisons use torch 2.13.0+cu130 in a separate environment. Tolerances stayed frozen: fp32 atol 1e-5 rtol 1e-4; bf16 atol 1e-2 rtol 1e-2. Speed bar: 1.15x median warmed throughput over 5 repetitions, against the archived base medians from this same machine.

## Final benchmark

| Site | Numerical | Median seconds (candidate / archived base) | Ratio | Peak allocated | Status |
| --- | --- | --- | --- | --- | --- |
| Pythia-410M TopK MLP, layer 4, fp32 | Prior tensor parity PASS, 0 feature flips | 0.920 / 2.363 | 2.569 | 2.43 GB | PASS |
| Qwen3-1.7B residual TopK, layer 7, bf16, serial default | PASS, 0 feature flips, identity PASS | 2.363 / 5.587 | 2.364 | 4.02 GB | PASS |
| Same Qwen site, fused masked batch | FAIL, 5 feature-id flips, identity max logit error 0.375 | 3.940 / 5.587 | 1.418 | 4.05 GB | FAIL, disabled |
| Gemma-3-270M Scope residual, layer 5, fp32 | Activations, codes, baselines, and 182 matched metrics max abs 0 | 1.198 / 2.898 | 2.418 | 3.00 GB | PASS on matched metrics. Automatic patch-index compare remains an index artifact |
| R1-distill Qwen-1.5B residual, layer 20, fp32 substitute | Prior tensor parity PASS, 0 feature flips | 3.450 / 10.021 | 2.905 | 9.01 GB | PASS. Does not replace Gemma |
| Eager interp-engine, SDPA, same torch as SAEBase | All 28 `resid_post` layers max abs 0 | not a throughput path | — | — | PASS |
| vLLM 0.28, eager, bf16 | Layer 7 residual outside the frozen bar | not benchmarked | — | — | FAIL |

The warmed Qwen repetitions were 2.346, 2.328, 2.376, 2.363, and 2.384 s. Pythia peak was 2,429,628,928 bytes, Qwen 4,024,979,456, Gemma 3,004,868,608, R1 9,011,744,256. Those peaks match the archived runs within a few hundred kilobytes.

## Where the time went

One serial Qwen engine pass, before the cache, spent 2.057 s in 225 model forwards and 0.013 s in 25 SAE encodes. Decode, metrics, and the result writer were 0.14 s, 0.08 s, and 0.006 s. The other ~3.1 s was `sha256_state_dict` copying the full model and SAE to the host on every `BatchEngine` construction. A direct measurement of that copy is 2.728 s for the Qwen model and 0.447 s for its SAE. The second call, on the same unchanged tensors, is 0.0015 s and 0.000058 s.

The digest bytes are unchanged. The cache key is storage address, shape, stride, offset, and the tensor in-place version. An in-place edit misses the cache and is hashed again. CPU check: 75 passed, including that repeat-and-invalidate test (`test_performance_hf_batch.py`, `test_m1_store.py`, `test_m2_interventions.py`, `test_m3_execution.py`).

After the cache, the dominant remaining cost on serial Qwen is those 225 model forwards: one capture, one logit readout, and one forward per condition. Fusing the capture and the logit readout would remove 25 of them, about 0.23 s. That is smaller than the hash that the profile actually identified, so it was not the change made here. A fused padded GEMM is still not used.

`enable_hf_masked_batch` stays false. `run_batch` walks each unpadded row through the single-example kernel. That matches serial Qwen at max abs 0. It is not a batching speedup, and it is not what produced the ratios above.

Coalesced cache encoding is unchanged: direct encode of the cached rows matched serial feature ids at batch sizes 1, 8, 32, 128, and 256, and a second `encode_from_store` reused all 25 rows. Injected OOM retained the reduced microbatch (8/8). A corrupted shard was rejected with `CacheIntegrityError`.

## Qwen batching and identity

The five feature-id substitutions came from the fused bf16 forward. Equal-length rows with no padding already diverge from the single-example forward: activation max abs 0.125, logits max abs 0.5, 3/3 TopK flips. Layer-0 inputs matched; the gap opens inside block 0. Grouped GEMM, deterministic cuBLAS, SDPA backend selection, CUDA graphs, and a second stream replica did not produce a fused forward that matches the single-example kernel and is faster.

The Pythia and R1 identity misses on the archived summaries are a separate protocol error. The pilot baseline uses `sample_batch=8` and the old gate replayed identity at `microbatch=4`. Same-width write-back is bitwise on both the base commit and the candidate. The candidate gate replays at the baseline width. Qwen's full serial workload was re-run with that gate after the hash cache: PASS.

Gemma's automatic comparer can still report failed patch tensors because patch logs are keyed by batch index and the identity-width change altered how many patch batches were recorded. Matched sample-and-condition metrics are the cross-version result.

## vLLM numerical mismatch

SAEBase, eager interp-engine, and vLLM were compared on the same token ids, the same bf16 checkpoint, and `resid_post`. Eager interp on the pilot torch with SDPA matches SAEBase at every layer, including the layer-0 input: max abs 0. The earlier 0.25 gap was eager attention against SAEBase SDPA.

vLLM 0.28 completes a forward with `CUDA_HOME` pointed at the cu13 nvcc, `VLLM_USE_FLASHINFER_SAMPLER=0`, `enforce_eager=True`, and prefix caching off. The layer-0 residual input matches SAEBase at max abs 0, so the capture is not reading a doubled residual and the embeddings match. The first difference is inside block 0.

| vLLM config | Layer 0 attention | Layer 0 MLP | Layer 0 residual | Layer 7 residual | Worst residual |
| --- | --- | --- | --- | --- | --- |
| Triton attention, vLLM RMSNorm | 0.0078125 | 0.03125 | 0.0625 | 0.0625, outside the bar | layer 24, 8.0 |
| Triton attention, native RMSNorm first | 0.0078125 | 0.0078125 | 0.015625 | 0.25, outside the bar | layer 26, 8.0 |
| FlashAttention, native RMSNorm first | 0.0078125 | 0.0078125 | 0.015625 | 0.25, outside the bar | layer 25, 8.0 |

Native RMSNorm reduces the layer-0 MLP gap and does not bring layer 7 inside the frozen bar. Triton and FlashAttention agree with each other once the norm implementation is the same, and both differ from HF SDPA. This vLLM build has no decoder SDPA backend. The error is an attention-kernel and norm difference that accumulates through the residual stream. It is not fixed by changing the hook, the token position, or the dtype. CUDA graphs stayed off. No vLLM throughput number is reported.

## Decision for PR #44

Merge the verified core. Do not treat vLLM as part of that release.

The core is serial HF execution as the default, row-wise `run_batch` kept only as a same-kernel opt-in, the same-width identity replay, coalesced SAE encoding, OOM microbatch retention, and the state-hash cache. Those paths keep the frozen tolerances. The cache is the change that clears 1.15x on the four timed sites. High-speed fused HF batching stays disabled. vLLM stays experimental and disabled: every configuration measured here fails the bf16 residual bar, so it is not a GPU-06 pass and it gets no speed claim.

CPU tests: 75 passed. GPU-05 and the 48-site sweep were not run.
