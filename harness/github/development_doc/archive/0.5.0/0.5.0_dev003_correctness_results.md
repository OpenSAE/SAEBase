# Work Plan 10 — RTX 5070 execution reuse (redacted)

Public summary of the private run `rtx5070-20261021`. Prompt text, token ids, and weights are not included. This does not certify vLLM or a high-throughput engine. Those belong to Work Plan 11.

Base `20752d916a82b3c46c2591beb74650d18399b560`. Work Plan 09 is merged at `41c51f4f77dfa37e9d771050d7294e5af9fbd083`. Machine: one RTX 5070, compute capability 12.0, 12227 MiB. Torch 2.14.1+cu130, transformers 5.20.2. Tolerances stayed frozen: fp32 atol 1e-5 rtol 1e-4; bf16 atol 1e-2 rtol 1e-2. Each mode is 1 warmup plus 5 synchronized repeats. CPU tests: 87 passed.

## What changed

`sha256_state_dict` reads the sorted state bytes on every call. The hash cache from Work Plan 09 is gone. A matching storage address or `Tensor._version` is not a content checksum: `parameter.data.add_` leaves `_version` at 0 while the bytes change, and the old cache returned the pre-edit digest.

Warm reuse is a `VerifiedCheckpointSession`. It is minted only after a strict hash, it cannot be pickled, and `leased_sha256` repeats the minted digest when a storage heuristic still matches. That heuristic is not a fresh checksum. `strict_recheck` reads the bytes again and closes the lease on mismatch. A `.data` edit after mint aborts the result release and writes no shard.

On HF hooks, one unpadded forward returns the site activation and the baseline logits. `enable_hf_masked_batch` stays false. The fused padded GEMM stays off. Serial HF remains the default.

Identity and full reconstruction do not decode. Scale and add stay on device and do not call `.item()`. Metrics move one stacked tensor to the host. Real edits still decode both code vectors. Subtracting the stored reconstruction missed the Qwen bf16 bar (activation max abs 0.0625, logits max abs 0.25), because that tensor is stored in the activation dtype. That shortcut was reverted.

Multi-site capture was not added. This pilot is one site per experiment, and `collect_activations` is already one site per call.

## Numerical result

| Site | Capture | Edit oracle (8 samples x 7 conditions) | Identity gate |
| --- | --- | --- | --- |
| Qwen3-1.7B residual TopK, layer 7, bf16, HF hooks | 25/25, activation max abs 0, logit max abs 0, feature-id flips 0 | max abs 0 | PASS |
| Pythia-410M TopK MLP, layer 4, fp32 | TransformerLens, fused HF capture not used | max abs 0 | PASS |
| Gemma-3-270M JumpReLU residual, layer 5, fp32 | TransformerLens, fused HF capture not used | max abs 0 | PASS |
| R1-distill Qwen-1.5B residual, layer 20, fp32 | TransformerLens, fused HF capture not used | max abs 0 | PASS |

Qwen checkpoint SHA-256: model `8f952f43284eed52d505ed6e851a68e2c5bd9960209aa35a890e3ddbb34cdf4e`, SAE `d68dfbf3796d0d6f1bb27fcc4c905ae3085e5c783594b4e0315065d384f194b4`. The other three sites match the Work Plan 09 digests.

## Timing

Medians are the middle of five repeats. The pre-cache column is the Work Plan 09 measurement that hashed on every engine construction. The cached column is the warm time after the hash cache this plan removes. Strict hashes on every construction. Warm lease skips that copy. Model-only is the same lease with an in-memory journal.

| Site | Pre-cache s | Cached s | Strict s | Warm lease s | Model-only s | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen L7 | 5.587 | 2.363 | 5.358 | 2.126 | 2.012 | 4.03 GB |
| Pythia L4 | 2.363 | 0.920 | 2.448 | 0.856 | 0.774 | 2.43 GB |
| Gemma L5 | 2.898 | 1.198 | 2.857 | 1.127 | 1.077 | 3.01 GB |
| R1 L20 | 10.021 | 3.450 | 9.987 | 3.433 | 3.350 | 9.01 GB |

Warm-lease ratios against the pre-cache medians are 2.63, 2.76, 2.57, and 2.92. Those ratios retire an unsafe cache; they are not a second independent 2.6x. Against the cached medians the increments are 1.11x, 1.07x, 1.06x, and 1.00x. Qwen is the path that also lost 25 forwards: the engine counts 200 model forwards (25 combined captures and 175 patches) where Work Plan 09 counted 225. One Qwen hash pair is 3.168 s, so the strict column sits on the pre-cache time.

Pythia strict is 2.448 s against a pre-cache median of 2.363 s. The five repeats span 2.369 to 2.656 s.

A caller who does not pass a lease pays the strict column. The lease is the fast path, and a result is a fresh content checksum only after `strict_recheck`.

TransformerLens counters in this run stay at 0. That backend returns from the batched cache path before the HF serial increments. The wall times above are the measurement.

## Decision

Keep the strict public hash, the in-process lease, the HF one-forward capture, the device-side scale and add, and the identity decode skip. Do not ship reconstruction subtraction. Do not enable fused padded GEMM or masked HF batching. Do not treat this run as a vLLM result.
