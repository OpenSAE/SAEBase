# Work Plan 09 — RTX 5070 pilot (redacted)

Public summary of the private run `rtx5070-20261021`. Prompt text, token ids, and weights are not included. This is not a 48-site result and not a Research Lab corpus result. GPU-05 was not run.

Base `339fbb603bc6d999fd126a91ea8709fee1c599c2`. The first GPU rows were measured on candidate `db8e652d818ea2ec6995ebe87b9a001d623db2bb` and recorded in `c93cb5ca846eabdbe90d81496845c56112802ca4` and `1ae89ff7e1a30b1e57886335fc8b6e7dbed4a594`. CI for `db8e652`: https://github.com/OpenSAE/SAEBase/actions/runs/39546669785. CI for `c93cb5c`: https://github.com/OpenSAE/SAEBase/actions/runs/39550371815. CI for `1ae89ff`: https://github.com/OpenSAE/SAEBase/actions/runs/39550761574. The follow-up below was measured on the working tree after `1ae89ff`.

Machine: one RTX 5070, compute capability 12.0, 12227 MiB. Pilot torch 2.14.1+cu130. Tolerances stayed frozen: fp32 atol 1e-5 rtol 1e-4; bf16 atol 1e-2 rtol 1e-2. Speed bar: 1.15x median warmed throughput over 5 repetitions.

## Sites

| Site | Backend | Result |
| --- | --- | --- |
| Pythia-410M TopK MLP, layer 4, `blocks.4.hook_mlp_out`, fp32 | TransformerLens | Tensor parity PASS (0 feature flips). Throughput ratio 0.987. |
| Qwen3-1.7B-Base residual TopK, layer 7, `blocks.7.hook_resid_post`, bf16 | HF hooks, serial default | PASS. 0 feature flips. Identity PASS both sides. Ratio 1.009. Peak allocated about 4.03 GB both sides. |
| Same Qwen site, fused masked batch | HF hooks | FAIL. Ratio 1.418, 5 feature-id flips, candidate identity max logit error 0.375. That kernel is no longer what `run_batch` executes. |
| Gemma-3-270M Scope residual, layer 5, fp32 | TransformerLens | Activations, codes, baselines, and 182 matched intervention metrics agree exactly. Ratio 1.027. See Gemma note. |
| R1-distill Qwen-1.5B residual, layer 20, registered substitute | TransformerLens | Tensor parity PASS. Ratio 1.295. This does not replace the Gemma JumpReLU cell. |

Checkpoint SHA-256 values are in the private `summary.csv`. Direct encode of the cached rows matched serial feature ids at batch sizes 1, 8, 32, 128, and 256 on every site measured before Gemma. A second `encode_from_store` reused all 25 rows. Gemma encode reports also matched feature ids at those batch sizes.

The synthetic encode microbenchmark (2048 by 1024, 32768 features) reached 4658, 33877, 128815, 244052, and 239361 rows/s at those batch sizes. That is not end-to-end evidence.

## Qwen batching

The five feature-id substitutions came from the fused bf16 forward, not from a wrong token position. Equal-length rows with no padding already diverge from the single-example forward: activation max abs 0.125, logits max abs 0.5, 3/3 TopK flips. A repeated identical row at batch 4 had activation max abs 0.25 and 0 flips. Masked and unmasked equal-length batches matched each other. Mixed lengths were measured separately. Layer-0 inputs matched; the gap opens inside block 0 and is large by layer 7.

Grouped GEMM, deterministic cuBLAS, SDPA backend selection, CUDA graphs, and a second replica on another stream did not produce a fused forward that matches the single-example kernel and is faster. `run_batch` now runs each unpadded row through `get_resid` and `run_with_patch`. After that change, equal-length, mixed-length, and repeated-row groups have activation and logit max abs 0 and 0 feature-id flips at the frozen bf16 bar. The engine opt-in path matches the serial default at max abs 0, with identity PASS, on both an equal-length group and a mixed-length group.

`enable_hf_masked_batch` stays false. The opt-in path is the single-example kernel, so it does not meet the 1.15x bar. The fused kernel remains outside the tolerance and is not re-enabled.

## Identity replay

This is separate from candidate-versus-base tensor parity. The pilot baseline uses `sample_batch=8` and the old gate replayed identity at `microbatch=4`. Same-width write-back is bitwise on Pythia and R1 for both the base commit and the candidate (max abs 0). Cross-width replay fails the 1e-5 gate. The base commit still raises the archived errors 3.325e-4 (Pythia) and 2.289e-5 (R1). TF32 was off. The gap is the batch shape used by the reference protocol, not a hook that writes the wrong residual.

The candidate gate now replays at the baseline width. On an 8-row mixed-length group it PASSes for Pythia and R1. If that replay runs out of memory, each half is rebaselined before it is compared. Edit chunks that are narrower than the baseline group are not treated as identity failures. Saved edit microbatch widths are unchanged.

## Gemma

`google/gemma-3-270m` and `google/gemma-scope-2-270m-pt` loaded with the instance token. Layer 5 residual, fp32, TransformerLens. Candidate median 2.821 s, base median 2.898 s, ratio 1.027, peak about 3.00 GB. Encode feature ids matched. Activations, codes, and baselines differ by max abs 0. All 182 intervention rows matched on logit and activation metrics (max abs 0) when keyed by sample and condition. 0 feature-id flips.

The automatic comparer reported 20 failed patch tensors and worst max abs 18.9. Those keys are batch-index logs. The candidate records 91 patch batches and the base records 65, because the identity replay width changed, so a shared index is not the same edit. The matched metrics above are the cross-version result. Candidate identity gate PASS. Base identity gate FAIL at max logit error 2.575e-5, the same width-mismatch protocol. Speed gate not met.

## GPU-06

Eager interp-engine 1.14.0 on the pilot torch, with SDPA attention, matches SAEBase at layer 7 `resid_post`: residual, clean logits, identity patch, and SAE-ablation logits max abs 0. Ablation residual max abs 4.88e-4, inside the bf16 bar. Four-token greedy generation matches. The earlier max abs 0.25 was interp's default eager attention against SAEBase SDPA. Hook name, token positions, and bf16 dtype already agreed.

vLLM 0.28.0 completes a forward when `CUDA_HOME` points at the cu13 nvcc (13.4), `VLLM_USE_FLASHINFER_SAMPLER=0`, `enforce_eager=True`, and prefix caching is off. FlashInfer autotune starts and finishes. Triton attention and FlashAttention both capture. Each misses the frozen bf16 residual bar at max abs 0.0625. An identity add of zeros is bitwise on the vLLM capture. The SAE ablation residual has the same 0.0625 gap. Greedy and stepwise token decisions match SAEBase. CUDA graphs stayed off. No vLLM benchmark, because the residual bar fails.

## Other checks

Injected OOM retained the reduced microbatch (8/8 kept). A corrupted activation shard was rejected with `CacheIntegrityError`. Coalesced cache encoding is unchanged. CPU: 74 passed across `test_performance_hf_batch.py`, `test_m1_store.py`, `test_m2_interventions.py`, and `test_m3_execution.py`.

## Decision

Serial Qwen parity PASS. Opt-in `run_batch` parity PASS and not faster, so masked batching stays off. Fused Qwen batching FAIL and is not the implementation. Pythia and R1 cross-version tensors PASS. Their old identity-replay miss is the batch-width protocol; same-width replay passes, including on the candidate gate. Gemma activations and matched intervention metrics PASS; the automatic patch-index compare does not, for the reason above; the speed bar is not met. Eager GPU-06 with SDPA PASS. vLLM forward PASS and vLLM residual equivalence FAIL at max abs 0.0625.

PR #44 stays a draft. The vLLM residual is still outside the frozen tolerance, and the HF batch flag is not a 1.15x acceleration.
