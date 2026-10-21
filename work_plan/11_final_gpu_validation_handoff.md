# Work Plan 11 — Final RTX 5070 GPU validation handoff

Status: Source code submitted in stacked Draft PR #47. New CUDA paths are NOT GPU validated.
Dependencies: PR #45 -> PR #46 -> PR #47.
Hardware: single RTX 5070, approximately 12 GB VRAM; keep original checkpoints and source identities unchanged.

## Nonnegotiable four deliverables

A. Fix CI for both PR #46 and PR #47. CPU tests and Python Ruff lint/format must all pass on final SHAs. Actual failed checks must be fixed before merging.

B. Run the COMPLETE 25-sample x 7-condition BatchEngine workload comparing suffix replay against the original engine, using the exact same source tokens and checkpoint hashes. The earlier 1.260x is for edited forward calls only. Need true end-to-end medians, numerical parity, exact feature IDs, peak allocated/reserved VRAM, and model-forward counters. Commit results, not just claims.

C. Run the implemented CleanPrefixKVDecoder prototype on real Qwen with 8 and 32 generated-token steps. Its clean cache is never advanced by a modified forward; a separately deep-copied cache supplies the edited step. Compare clean AND edited logits, selected SAE support, activation and next tokens against the full-prefix reference on every step at frozen bf16 tolerance 1e-2. Benchmark five repeats each for baseline full-prefix vs clean KV; optimize the cache fork only after parity. A slower or numerically inconsistent KV path stays experimental. Do not confuse persistent edited-history KV with the reference clean-prefix protocol.

D. Run an actual interp-engine vLLM hooked SAE edit, both dynamic vllm and static vllm-static if memory permits. InterpVLLMSAE measures a real resid_post capture, uses the ORIGINAL SAEBase encode/edit and sends the exact one-position residual delta through interp-engine per-request steering. It verifies the write physically affected the target tensor; a separate Hugging Face process saves the original tensor/logits for reference. Test numeric parity for activation, SAE feature IDs, edited tensor AND downstream logits before marking scientifically equivalent. Generation-only vLLM does NOT count. Do not assume that a successful vLLM forward fixes the previously measured mismatch.

## Source delivered in PR #47

- saebase/interventions/kv_cache_exact.py : standalone experimental clean-prefix KV implementation with explicit full-prefix oracle and fail-closed comparison.
- tests/test_clean_kv_exact.py : CPU oracle covering multiple generated tokens and three edit conditions.
- saebase/vllm_backend/sae_interventions.py : interp-engine vLLM SAE capture and per-request scoped edit, dynamic or static point.
- scripts/workplan11_acceptance_gpu.py : actual GPU e2e, kv, hf-reference, vllm commands.

## Exact execution order

Protect original local checkout and experiment archive. Work from a separate branch worktree at origin/perf/wp11-complete-kv-vllm-20261021.

~~~bash
git fetch origin
git worktree add ../SAEBase-wp11-final origin/perf/wp11-complete-kv-vllm-20261021
cd ../SAEBase-wp11-final
python -m pip install -e '.[analysis,dev,quality]'
python -m pytest -q tests/test_clean_kv_exact.py tests/test_suffix_replay.py tests/test_performance_hf_batch.py tests/test_m2_interventions.py tests/test_weight_identity_modes.py
python -m ruff check saebase/interventions/kv_cache_exact.py saebase/vllm_backend/sae_interventions.py scripts/workplan11_acceptance_gpu.py tests/test_clean_kv_exact.py
python -m ruff format --check saebase/interventions/kv_cache_exact.py saebase/vllm_backend/sae_interventions.py scripts/workplan11_acceptance_gpu.py tests/test_clean_kv_exact.py
nvidia-smi
python scripts/workplan11_acceptance_gpu.py e2e
python scripts/workplan11_acceptance_gpu.py kv --steps 8
python scripts/workplan11_acceptance_gpu.py kv --steps 32
python scripts/workplan11_acceptance_gpu.py hf-reference
~~~

For actual vLLM, create a CLEAN separate Python environment and install the currently compatible interp-engine[vllm] release; do not modify the working HF/Torch environment. Carry the privately saved HF reference artifact to the vLLM process but never upload it to public GitHub.

~~~bash
python -m pip install 'interp-engine[vllm]'
python scripts/workplan11_acceptance_gpu.py vllm --backend vllm
# After the first worker process terminates, in a NEW process:
python scripts/workplan11_acceptance_gpu.py vllm --backend vllm-static
~~~

Report exact CUDA, Torch, Transformers, vLLM, interp-engine versions and drivers. If vLLM-static OOMs on 12 GB, record the measurable limit and attempt the dynamic hooked backend, without silently quantizing or changing checkpoint. Keep the model/SAE and vLLM memory fractions within detected free VRAM.

## Acceptance gates

- Scientific: zero feature-support changes and matched tokens, layer, patch/readout position, model weights and input protocol. Frozen fp32 atol 1e-5 rtol 1e-4; bf16 atol/rtol 1e-2.
- End-to-end: compare entire ResultWriter metric maps by stable measurement ID, never patch batch index. All metrics within tolerance, no missing or duplicate measurements, >=5 measured repeats and CUDA synchronization.
- KV: stepwise exact SCIENTIFIC parity. A speedup requires >=1.15x real median 8/32 step workload; a deep-copied cache alone does not establish speed.
- vLLM: real worker must prove capture, actual SAE write and downstream effects, not just API import or generated text. Any HF residual or logit mismatch remains explicitly NOT_REFERENCE_EQUIVALENT and disabled by default.
- CI: package, Python 3.11/3.12, quality and focused CPU tests on exact final PR SHAs. No public GPU endpoint enabled.
- If experimental suffix temporarily swaps module.forward, it requires exclusive single-owner execution; do not advertise it as concurrency-safe until the implementation explicitly enforces that constraint.

## Deliverables

Private directory under WP11_RUN (default /workspace/experiments/performance/workplan11/final_5070):
end_to_end.json, clean_kv_parity.json, hf_reference_private.pt, hf_reference_status.json, vllm_vllm.json, vllm_vllm-static.json, final_report.md.

The local agent must fill final_report.md with actual model/SAE and code hashes, per-path PASS/FAIL/BLOCKED, max absolute and relative errors, all repeated times, GPU memory, complete workload speedups, failed backend logs and exact PR commit URLs. Public report must omit raw private token IDs/weights.

DO NOT finalize or merge the experimental KV/vLLM defaults based on theoretical design. Fix failing CPU/API mismatches, run RTX 5070, and only enable a measured scientifically validated fast path. Stop building unrelated infrastructure. User goal is completing these four specific validation/engineering tasks, not another round of planning.
