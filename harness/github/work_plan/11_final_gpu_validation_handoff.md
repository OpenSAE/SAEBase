# Work Plan 11 — Final Implementation and RTX 5070 GPU Acceptance

Updated: 2026-10-21. THIS IS THE CURRENT HANDOFF; it supersedes earlier instructions to stop after a KV/vLLM BF16 mismatch.

Execution owner: local GPU agent. Reviewer: OpenSAE / SAEBase.
Source branch: perf/wp11-complete-kv-vllm-20261021, stacked on PR #46 and PR #45. Protect existing local GPU assets and noncommitted work.
Hardware: one RTX 5070, nominal 12 GiB; inspect physical and free memory. No mandatory 48-site sweep or public GPU server.

## Latest code completion and first GPU commands

**Code has already been written and pushed** to PR #47; the agent must debug/test it on the RTX 5070 instead of rewriting the core acceleration algorithms.

The implemented source includes:
- HF persistent live SAE editing with real past_key_values in `saebase/interventions/hf_cached_steering.py`; one Qwen prompt prefill followed by one cached decode forward per generated token, freshly encoding and editing SAE codes each step, with separate TTFT/throughput counters.
- Neuronpedia interp-engine persistent fixed SAE decoder-direction generation in `saebase/vllm_backend/fast_steering.py`; upstream `steer(..., generated=True)` and `generate_stream(...)` perform actual vLLM KV-cached decode, not a simulated generation result. Includes real first-token latency and explicit `stop_at_eos=False` for length-matched timing.
- `saebase/tools/workplan11_fast_steering_gpu.py` measures **8, 32, 128** generated tokens over 3 public prompts, five repeats per condition, full elapsed and TTFT, actual tokens/sec, peak allocated/reserved VRAM, source SHA, token IDs kept in a PRIVATE artifact. Its HF live mode independently runs a no-hook raw KV oracle against identity SAE steering. Its vLLM mode performs a separate unsteered/steered control request. A file cannot claim a PASS solely from token agreement.
- `tests/test_fast_sae_steering.py` includes synthetic multiple-token identity equivalence, actual SAE editing effects, hook cleanup and a scoped vLLM stream mock. CPU tests are not substitutes for the real Qwen GPU run.

**Mandatory validation order (in separate supported environments):**

~~~bash
git fetch origin
git worktree add -b perf/wp11-gpu-evidence ../SAEBase-wp11-gpu origin/perf/wp11-complete-kv-vllm-20261021
cd ../SAEBase-wp11-gpu
python -m ruff check saebase/interventions/hf_cached_steering.py saebase/vllm_backend/fast_steering.py saebase/tools/workplan11_fast_steering_gpu.py tests/test_fast_sae_steering.py
python -m ruff format --check saebase/interventions/hf_cached_steering.py saebase/vllm_backend/fast_steering.py saebase/tools/workplan11_fast_steering_gpu.py tests/test_fast_sae_steering.py
python -m pytest -q tests/test_fast_sae_steering.py tests/test_clean_kv_exact.py tests/test_interp_vllm_sae_contract.py tests/test_suffix_replay.py
python -m pytest -q
nvidia-smi
export WP11_RUN=/workspace/experiments/performance/workplan11/final_5070
# HF environment, same Qwen and registered SAE:
python saebase/tools/workplan11_fast_steering_gpu.py --mode hf-live --lengths 8,32,128
python saebase/tools/workplan11_fast_steering_gpu.py --mode hf-reference --lengths 8,32,128
# Existing previously validated Qwen feature-intervention throughput:
python saebase/tools/workplan11_acceptance_gpu.py e2e
python saebase/tools/workplan11_acceptance_gpu.py hf-reference
~~~

Before vLLM tests choose a **source-verified registered SAE feature ID**, for example from the private `hf_reference_private.pt` produced by `python saebase/tools/workplan11_acceptance_gpu.py hf-reference`; verify the value is present in the SAE and, for claimed prompt activation, its TopK support. Do NOT let scripts silently substitute 185 or some default. Export it as `FEATURE_ID`:

~~~bash
# In the isolated interp-engine/vLLM installation:
FEATURE_ID="$(python -c 'import os,torch; from pathlib import Path; p=Path(os.environ["WP11_RUN"])/"hf_reference_private.pt"; print(int(torch.load(p,map_location="cpu",weights_only=True)["feature_id"]))')"
python saebase/tools/workplan11_fast_steering_gpu.py --mode vllm --lengths 8,32,128 --feature-id "$FEATURE_ID" --strength 2
# New worker process, sufficient VRAM to allocate declared static taps:
python saebase/tools/workplan11_fast_steering_gpu.py --mode vllm-static --lengths 8,32,128 --feature-id "$FEATURE_ID" --strength 2 --utilization 0.70
~~~

No source code in this branch has yet proven **new** HF-live/vLLM-stream 8/32/128-token performance on the target GPU. Preserve prior Work Plan 11 Qwen W2 **1.276x** as a SEPARATE completed benchmark; do not recycle that number. For these new protocols, aim first for a working streamed decode that is materially faster than the old full-prefix baseline; then analyze numerical behavior and scientific comparability separately.

**Acceptance definition:** GPU startup, 8/32/128 actual generated tokens, no cache/hook/request leakage, real intervention effect, stable output and finite metrics, bounded RTX5070 memory, source-locked 5-repeat latency/TTFT/tokens-per-second tables, CPU+CI green. An accelerated but scientifically DIFFERENT mode may be retained under its explicit new protocol ID; old clean-prefix experimental results are reproduced only by the original reference path.

**Important note on CLI:** `--feature-id` is REQUIRED for vLLM modes, and fixed-length benchmark runs now use `stop_at_eos=False`; the HF modes select an active feature from each exact prompt unless the user explicitly supplies one. Do not alter original SAE feature operators or research metrics.

---

## Mission

The upstream Neuronpedia / decoderesearch interp-engine ALREADY implements tokenwise KV-cached vLLM generation with per-request steering and optional static CUDA graph taps. SAEBase now has two NEW source implementations, in addition to the historical clean-prefix reference.

**GPU agent must run and improve actual accelerated implementations first**, then perform numerical, scientific, memory and release verification. Do NOT count a numerical mismatch with a different backend as proof that KV/vLLM cannot work. Do not sacrifice the existing reference scientific protocol or silently substitute a different one.

Current stacked source and interfaces:

- saebase/interventions/hf_cached_steering.py : generate_live_sae_cached. Real Qwen3 HF KV; one prefill + one forward per generated token, fresh SAE encode and SAEBase residual-preserving edit on every token, persistent edited KV. Public protocol: persistent_live_sae_kv_v1.
- saebase/vllm_backend/fast_steering.py : decoder_feature_direction and generate_decoder_direction_cached. True interp-engine steer() + generate_stream() with generated=True and cached vLLM; uses actual SAE decoder difference and static/dynamic vLLM taps; protocol: persistent_decoder_direction_kv_v1.
- saebase/interventions/kv_cache_exact.py : historical reference-compatible clean-history KV experiment. It forks a clean cache for edits and discards edited past; on earlier Qwen BF16 test incremental/full-prefix numerical differences persisted. This is an INDEPENDENT research protocol, not the only possible accelerated generation.
- saebase/vllm_backend/sae_interventions.py : real per-request prompt-position live SAE capture + one-shot feature edit in interp-engine. Distinct from generated-token continuous dynamic ablation.
- saebase/tools/workplan11_fast_steering_gpu.py : NEW 8/32/128 token end-to-end generation timing for the old clean-prefix reference, HF live KV, and interp-engine vLLM/static fixed SAE direction.
- saebase/tools/workplan11_acceptance_gpu.py : existing complete BatchEngine E2E suffix replay and HF reference / vLLM SAE capture / historical clean-KV numerical gate.
- tests/test_fast_sae_steering.py : NEW CPU test contracts for live HF KV and vLLM streaming scopes.
- tests/test_clean_kv_exact.py, tests/test_interp_vllm_sae_contract.py, tests/test_suffix_replay.py : previous scientific and safety regressions.

## 0. First ten minutes: source, environment and CI (P0)

1. Record Git worktrees/status/HEAD, code commit, GPU/driver, PyTorch/Transformers/SAE Lens, interp-engine and vLLM package versions. Save original private experimental results. Do not reset/force-push.
2. Check GitHub Actions on the LATEST source SHA, not an older green SHA. Run Ruff check, Ruff format, and complete CPU tests. Fix any failure and push. Never declare the code complete merely because it imports.
3. Prepare a dedicated HF environment AND separate interp-engine[vllm] environment. Do not break the verified Torch/CUDA installation. The upstream engine may require nvcc 13.x and FlashInfer fixes on Blackwell; use supported pinned dependency combinations in an isolated env.
4. Review the exact upstream implementation:
   - https://github.com/decoderesearch/interp-engine
   - https://github.com/decoderesearch/interp-engine/blob/main/interp_engine/steer.py
   - https://github.com/decoderesearch/interp-engine/blob/main/harness/github_development_doc/PERFORMANCE.md
   - https://github.com/decoderesearch/interp-engine/blob/main/harness/github_development_doc/USAGE.md
   - https://github.com/decoderesearch/interp-engine/blob/main/tests/test_vllm_kv_isolation.py
   - https://huggingface.co/harness/github_development_doc/transformers/main/kv_cache
   Verify actual installed public APIs, CUDA kernel flags, static read/write declarations and the per-request cache-salt policy. Prefer existing upstream code to bespoke vLLM extensions.

~~~bash
git fetch origin
git status --short
# Use a fresh safe worktree or the existing protected checkout.
git worktree add -b perf/wp11-gpu-final ../SAEBase-wp11-gpu origin/perf/wp11-complete-kv-vllm-20261021
cd ../SAEBase-wp11-gpu
python -m pip install -e '.[analysis,dev,quality]'
python -m ruff check saebase/interventions/hf_cached_steering.py saebase/vllm_backend/fast_steering.py saebase/tools/workplan11_fast_steering_gpu.py tests/test_fast_sae_steering.py
python -m ruff format --check saebase/interventions/hf_cached_steering.py saebase/vllm_backend/fast_steering.py saebase/tools/workplan11_fast_steering_gpu.py tests/test_fast_sae_steering.py
python -m pytest -q tests/test_fast_sae_steering.py tests/test_clean_kv_exact.py tests/test_interp_vllm_sae_contract.py tests/test_suffix_replay.py tests/test_weight_identity_modes.py tests/test_m2_interventions.py
python -m pytest -q
nvidia-smi
~~~

## 1. Mandatory HF live SAE KV generation (P0)

This is NEW substantive work. It must not be stopped because older clean-prefix output differs numerically.

Test the new generated-token SAE logic on actual Qwen3-1.7B Base, layer 7, registered residual TopK SAE, at 8, 32 and 128 tokens:

~~~bash
export WP11_RUN=/workspace/experiments/performance/workplan11/final_5070
python saebase/tools/workplan11_fast_steering_gpu.py --mode hf-live --lengths 8,32,128
python saebase/tools/workplan11_fast_steering_gpu.py --mode hf-reference --lengths 8,32,128
~~~

Expected internal contract: ONE full prompt prefill, then ONE one-token KV forward per new token; freshly compute actual SAE z for each emitted token and use SAEBase apply_condition() to edit h. Edited downstream history is retained, so this is Neuronpedia-style persistent steering, not the old clean-prefix science. Require 1 warmup + >=5 timed repetitions, max allocated/reserved VRAM, forward counters, token IDs (PRIVATE), actual feature IDs from at least one traced run, no NaN/OOM/invalid position, hook cleanup and no state leakage between requests. Test zero-strength or identity alongside ablate/scale to ensure actual causal effects. Test two prompts and repeated runs for state isolation.

If BF16 differs from historical full-prefix results, record the difference and keep the distinct protocol identity; **do not disable a working and correct cached-generation path merely because the old full-prefix numerical protocol differs**. Fix within-backend bugs, wrong feature indices, old KV contamination and memory leaks.

Implementation restrictions: model.eval(), torch.no_grad/inference mode, exclusive HF worker lock, one loaded model + SAE per GPU, bounded max context, no unsupported training/gradient callbacks. The result must not be silently used as prior Stage D science.

## 2. Mandatory genuine interp-engine vLLM/static generated-token steering (P0)

The code in fast_steering.py already constructs an SAE decoder direction from original SAE.decode(onehot)-SAE.decode(zeros), then applies it with upstream AddSpec at resid_post. It sets generated=True and masks all earlier prompt token positions, leaving the last prompt token and every subsequent generated token steered. This must run on the vLLM worker, not be simulated with text replacement.

Run in SEPARATE clean vLLM+interp-engine environment/process; pin CUDA, torch, interp-engine, vLLM and dependencies. Static mode on 5070 had previously needed gpu_memory_utilization >=0.70 to obtain an actual tap buffer; do not mix different backends in the same process.

~~~bash
python saebase/tools/workplan11_fast_steering_gpu.py --mode vllm --lengths 8,32,128 --feature-id 185 --strength 2
# Restart in a new process after dynamic worker exits.
python saebase/tools/workplan11_fast_steering_gpu.py --mode vllm-static --lengths 8,32,128 --feature-id 185 --strength 2 --utilization 0.70
~~~

NOTE: fixed decoder-vector addition is a real SAE feature-direction steering method, but NOT identical to live tokenwise feature ablation/scale. Explicitly label it. If user feature 185 is not a meaningful registered feature for the chosen SAE checkpoint, pick an active feature from the HF reference capture with a documented stable ID; don't fabricate labels or strengths.

Required checks:
- Open upstream steer context reaches actual vLLM generation request via per-request steering_spec, generated=True applies after prefill, masks earlier prompt rows.
- A control unsteered run vs edited run with same seed and tokenization; prove an actual activation edit even if greedy output token sequence happens to stay unchanged.
- No cross-request leakage: unsteered -> steered -> unsteered and two independent prompts; prefix-cache cache_salt isolation and static tap declaration verified by actual worker.
- Independent time-to-first-token, total decode time, tokens/s and memory, warmup + 5 repeats, 8/32/128 output lengths. Compare same-backend steered vs unsteered cost, and clearly label cross-protocol HF speed.
- If static graph cannot fit 5070 due to verified buffer budget, report exact bytes and try smaller max_model_len/static tap set rather than silently making up times.

Do not require bitwise equality with Hugging Face as a prerequisite for measuring or implementing fast steering. If SAE TopK code support differs between backends, pin execution protocol and report which features; never reuse incompatible historical cache identities.

## 3. Retain and verify exact original experimental evaluation (P0)

PR #45 strict source hashes; PR #46 opt-in block-suffix replay measured on real Qwen: complete 25 samples x 7 edits, median 2.409 -> 1.888 s (1.276x), max metric difference zero. Keep this independently publishable.

~~~bash
python saebase/tools/workplan11_acceptance_gpu.py e2e
python saebase/tools/workplan11_acceptance_gpu.py hf-reference
~~~

Validate original and optimized sample/condition identities, exact SAE support and measured metrics, no stale prefix cache, OOM and exception cleanup, source provenance. For changed source changes since prior benchmark, rerun warmup + 5 repeats. Stable defaults remain old hook path unless real deployment is tested; opt-in suffix replay can ship with gates.

## 4. Historical clean-prefix KV attempt (P1, after real fast generation)

Only AFTER sections 1 and 2 are genuinely running, investigate the source of the Qwen incremental-vs-full-prefix BF16 difference seen before. Keep original full-prefix implementation for exact historical comparisons.

~~~bash
python saebase/tools/workplan11_acceptance_gpu.py kv --steps 8
python saebase/tools/workplan11_acceptance_gpu.py kv --steps 32
~~~

First check tokenization/BOS, RoPE positions, attention mask, cache ownership, and per-layer K/V vs full prefix; test fp32 diagnostic and batch-invariant or stable attention kernels if available. Report per-layer relative errors and **whether different support alters the actual experiment**, not just a global max abs. If cached numerical trajectory cannot reproduce historical full-prefix, expose it only as a distinct execution identity. Do NOT declare caching impossible.

## 5. Mandatory final report and decision

Under the gitignored private WP11_RUN directory save separate:
- fast_steering_hf_live.json and fast_steering_hf_reference.json with 8/32/128 series and source hashes
- fast_steering_vllm.json and fast_steering_vllm_static.json from actual GPU workers, or precise failure logs
- end_to_end.json and hf_reference_private.pt, full BatchEngine parity
- clean_kv_parity_8.json and clean_kv_parity_32.json
- final_report.md with real command transcript, frozen model/SAE/tokenizer checkpoint identities, environment and GPU memory, numeric and feature effects, all five repeats and p50/p95 where estimable.

Produce harness/github_development_doc/performance/workplan11_final_results.md as public redacted summary. Its capability matrix MUST distinguish:
1. reference_clean_prefix_stepwise_v1 (slow but reference-frozen)
2. persistent_live_sae_kv_v1 (HF dynamic SAE encode/edit every generated token)
3. persistent_decoder_direction_kv_v1 (interp-engine vLLM/static fixed SAE decoder-vector generation)
4. block_suffix_replay_v1 (exact 25x7 intervention forward reuse)
5. legacy clean-history KV prototype (independent work, not mandatory fast path)
6. interp-engine one-prompt live SAE feature edit (already implemented, not continuous generated-token ablation)

Mark each as CODE_READY, CPU_TESTED, GPU_RUNNING, GPU_PARITY_PASS, GPU_SPEED_PASS, EXPERIMENTAL or BLOCKED with actual evidence; do not equate green CPU CI with real GPU. A legitimate measured faster NEW scientific protocol can ship opt-in even if not numerically identical to the old protocol, provided semantics, source identity, output validity, and absence of hidden cache leakage are verified.

When GPU evidence is obtained, land PR #45 -> PR #46 -> PR #47 in dependency order after full CI, retain high-speed generation behind an explicit protocol/capability gate, no public server deployment. Do not create Work Plan 12. If a real performance blocker remains, fix code directly, rerun the **same** tests, report residual engineering limitation rather than returning another strategy memo.

## First instruction to the GPU agent

Use the latest PR #47 source. Finish CI and CPU tests; launch hf-live generation 8/32/128; launch real interp-engine vLLM+static generation 8/32/128 in isolated environment; benchmark the original full-prefix reference and the previous Qwen BatchEngine E2E; fix actual kernel/cache/SAE bugs, rerun; then publish only validated measured speedups and the final implementation/PR decisions. No alternative planning and no early BLOCKED unless a reproducible run proves the cause.
