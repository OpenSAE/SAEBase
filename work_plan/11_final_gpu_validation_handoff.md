# Work Plan 11 — Final GPU Execution, Debugging, and Release Acceptance

Updated: 2026-10-21. This document REPLACES the previous short handoff in the same file.
Status: engineering code present; GPU acceptance and final release NOT completed.
Executor: local agent with one RTX 5070 (nominal 12 GiB) and the exact original Qwen model + SAE.
This is an **implementation, debugging, measurement, and completion task**, NOT another planning task.

## Critical correction after inspecting Neuronpedia / interp-engine source (2026-10-21)

**Supersedes every instruction below that equates "not bitwise/numerically identical to HF" with "vLLM is broken" or treats "cached steering not byte-identical to full-prefix HF" as proof it cannot be shipped.** HF is a frozen historical reference, not a privileged mathematical ground truth. Different BF16 kernels and batch shapes may produce different but valid floating-point traces. The scientific question is whether it changes SAE feature identities/effects for the workload, and whether a new backend is clearly declared and independently validated.

### Read these upstream sources BEFORE another GPU tuning loop

- Engine implementation, supported capture/writes and backend choices: https://github.com/decoderesearch/interp-engine
- Real steering semantics, position_mask (EXCLUDED positions), and generated=True/False: https://github.com/decoderesearch/interp-engine/blob/main/interp_engine/steer.py
- Read/write graph taps, prefix-cache isolation, CUDA graph and vLLM-static design: https://github.com/decoderesearch/interp-engine/blob/main/docs/PERFORMANCE.md
- Accurate numerical comparison process and cases (do not invent thresholds): https://github.com/decoderesearch/interp-engine/tree/main/validator
- Neuronpedia inference and SAE feature/direction endpoints: https://github.com/hijohnnylin/neuronpedia/blob/main/apps/inference/README.md
- Actual capture+steering+generate timing suite, including prompt length and GPU specification: https://github.com/decoderesearch/interp-engine/blob/main/benchmarks/results-latest.md

Neuronpedia ALREADY has production vLLM dynamic hooks, a static-tap CUDA graph backend that can read/write declared residual sites, and KV-cached streaming generation. Do not reimplement that engine or dismiss its design based only on cross-library BF16 differences. Its published B200/Qwen3-4B results are evidence of architectural feasibility, NOT a speed claim for Qwen3-1.7B on RTX5070.

### P0: Explicitly separate THREE scientific protocols

**A. reference_clean_prefix_stepwise_v1 (historical reproducibility).** For each generated token, SAEBase recomputes the entire CLEAN prefix (of that steered trajectory's own actual token history), computes the live SAE feature and its residual-preserving edit ONLY on the current token, then computes edited next-token logits. A new incremental clean-cache implementation may speed it up, but old paper/benchmark outputs must remain pinned to their HF/TL execution protocol. For equivalence to those old results retain old dtype, attention kernel, exact activation site, tokenization and feature-id support; a divergent backend is NOT interchangeable in the same experiment identity.

**B. fast_persistent_steering_v1 (high-throughput Neuronpedia-style generation).** A per-token or fixed steering intervention runs during cached generation; **edited historical KV can persist**. This is a valid research/interactive generation protocol with different causal history. Implement using upstream `interp-engine` native `steer(model, SteeringSpec(...), generated=True)` and its real `generate_steps`/streaming API, using the declared `resid_post` site and tested vLLM/static taps. Do NOT demand byte equality with protocol A as an acceptance condition; do demand backend-local no-edit identity, finite activations, correct per-request isolation, actual steering effect and stable reproduction. Never report B's outputs under A's experiment identity.

**C. fixed_decoder_direction_steering_v1 versus live_feature_ablation_v1.** They are NOT the same operation. Neuronpedia's simple decoder-vector additive steering can add a fixed direction at each token, needing no SAE encode during decode. SAEBase's live ablate/scale/coalition computes z=encode(h_t) per current token and may require different deltas each step: h'_t = h_t + decode(z'_t) - decode(z_t). That requires a dynamic in-worker per-token capture/SAE callback or a verified segmented step-loop. The currently implemented `InterpVLLMSAE.measure` computes ONE prompt-specific delta and reuses it for that request; it does NOT prove dynamic SAE ablation across generated tokens. Do not mislabel fixed-vector steering as that capability. Implement and benchmark them as separate operations.

### P1: Fix the evaluation criterion rather than merely raising tolerances

For two different backends in BF16, report:
- same checkpoint/tokenizer/layer semantics/position; correct clean forward and no-edit/identity invariants within each backend;
- actual residual normalized relative L2, max abs with activation scale, cosine, and per-position logit differences;
- SAE TopK support overlap and **which** features flipped, TopK k-versus-k+1 margin at the boundary, ranks, actual z_i and patch vector effects;
- intervention outcomes on matched prompt variants, paired behavioral/metric changes (with uncertainty across prompts);
- reproducibility under each backend, exact backend/build/kernel/protocol in the experiment identity.

Small BF16 errors can cross a near-tied TopK boundary. That does not automatically make vLLM incorrect. However changed feature support means B must not silently reuse old A codes/caches/results. For strict reproduction of an existing result, keep the reference exact path; for a NEW high-throughput study, publish backend-native data and validate its scientific conclusions independently. Do not relabel `BACKEND_NATIVE` as `REFERENCE_EXACT` without stronger evidence.

### P2: KV engineering strategy

**First ship a real upstream-based fast steering baseline**: interp-engine vLLM with `generated=True`, per-request steering, prefix-cache isolation and, when VRAM permits, vLLM-static predeclared read/write taps. Benchmark 8, 32, 128 tokens with one/2/4 concurrent requests and capture+steering, on RTX5070. Verify actual public upstream API in the installed version. Do not confuse vLLM-generate with capture-capable backends.

**Then separately accelerate A**: maintain CLEAN history KV for tokens in the steered sequence, fork/snapshot it for the current-step edited suffix, discard edited KV, and advance ONLY clean history. The existing `CleanPrefixKVDecoder` attempts that. If incremental and full-prefix BF16 computations disagree, first test exact tokenization/cache mask, position ids, full-prefix versus one-token kernel/batch shape and fp32 diagnostic; record whether topk support or actual intervention metrics change. Crucially, incrementally computed KV need not have identical BF16 tensors to a full-sequence GEMM. Preserve A's old full-prefix reference for strict legacy comparisons; do not conclude KV caching is mathematically impossible. If clean-cache semantics hold but legacy HF tolerances fail, this becomes a separately identified cached numerical protocol, not a drop-in replacement for old experimental results.

For either mode, account for possible hook skipping when vLLM prefix cache hits. Neuronpedia's engine uses per-request cache_salt to isolate intervention requests; adopt the supported upstream policy rather than globally disabling prefix caching or inheriting stale steered KV. Never transplant historical KV between independent baseline and steered continuations.

### P3: Objective finish and report

1. Complete PR #45 and #46 independently after CI/parity; PR #46 already has Qwen full W2 25×7 median 2.409 s to 1.888 s, a measured 1.276x; don't withhold that benefit.
2. In PR #47 execute **two** fast-steering benchmarks: B using upstream engine native cached generation (this is a required actual capability target), then A's clean-cache reimplementation if needed for exact historical reuse. Do NOT replace B with mere single-prompt feature edits.
3. Demonstrate real SAE decoder-direction steering from feature ids on generated tokens; test live SAE activation-dependent edits separately and label unsupported if current upstream hook protocol lacks a dynamic worker callback.
4. Treat legacy HF difference as a *comparability finding* rather than wholesale vLLM failure; classify capabilities by per-protocol validation.
5. Publish a small capability matrix with columns: protocol, backend, prompt capture, generation steering, persistent KV, live SAE re-encode, numerical reference identity, RTX5070 throughput, peak VRAM and status. No "all accelerated" claim without measurements.

Agent must inspect the linked upstream code BEFORE modifying its own worker. No new generic engine or another unfocused vLLM numerical-tolerance sweep. Source-driven, measured reuse is the priority.

---

## 0. Starting facts — read and reconcile with live GitHub

Repository: OpenSAE/SAEBase.

The three stacked PRs are:
- #45 https://github.com/OpenSAE/SAEBase/pull/45 — Work Plan 10, strict full-weight hashes, VerifiedCheckpointSession, HF single-forward capture/logits. Its CI has passed; NOT merged at the time of this edit.
- #46 https://github.com/OpenSAE/SAEBase/pull/46 — opt-in block-suffix replay, Qwen L7 **edited-forward ONLY** benchmark median 0.47774 s -> 0.37909 s (1.260x). Latest #46 SHA 2d36c2a... has passed CI. NOT merged.
- #47 https://github.com/OpenSAE/SAEBase/pull/47 — optional experimental clean-history KV decoder, interp-engine hooked-vLLM SAE intervention adapter, end-to-end RTX 5070 test harness and CPU contract tests. It is a **stacked PR against #46**, NOT against main. Current source head when this revision was written was 4bd66e45; confirm current SHA before work.

At that specific #47 commit, GitHub Actions https://github.com/OpenSAE/SAEBase/actions/runs/39662842992 reported **all required offline Python 3.11, 3.12, package, analysis installation and localization jobs succeeded; only the quality job failed on Ruff formatting**:
- saebase/vllm_backend/sae_interventions.py
- scripts/workplan11_acceptance_gpu.py

Do not repeat already successful GPU results or assert the CI is still failing without checking the NEW head. Existing offline CPU tests do NOT establish real GPU parity. The #47 test for interp-engine is a **mock API contract**; in particular, the newly used `decode_residuals`, `static_points`, `static_writes`, `sync.capture`, `SteeringSpec` and `position_mask` behavior MUST be verified in the installed upstream version. Do not trust a mock as confirmation that the vLLM worker supports an operation.

No need to revisit GPU-05 / 48-site matrix, Research Lab UI, marketing, new job servers, or unrelated cleanup.

## 1. Preserve the working tree, repair CI and resolve PR layering (P0)

1. Capture current `git status --porcelain`, branch and HEAD, worktrees, exact dependency pins, available model/SAE checkpoints, and already executed private GPU output; never reset/overwrite raw data.
2. Use separate worktree from `origin/perf/wp11-complete-kv-vllm-20261021`. The #47 parent at initial creation was an earlier #46 head. Sync the final #46 fixes carefully so the stacked PR contains the verified current parent without throwing away the new KV/vLLM work. Do not force-push.
3. Fix the **two known Ruff files** using `ruff format`, then run the full touched-file lint/format; report the changed-file list. Do not modify unrelated scientific logic merely to placate formatting.
4. Run the focused tests and complete local CPU suite. When a failure occurs, find the FIRST real stack trace, add a minimal reproducer, repair code, rerun.
5. Push. Verify GREEN CI on the ACTUAL LAST commit (not an earlier passing SHA).

Commands (adapt only if an existing protected worktree has the same path):

~~~bash
git fetch origin
git status --short
git worktree add ../SAEBase-wp11-gpu origin/perf/wp11-complete-kv-vllm-20261021
cd ../SAEBase-wp11-gpu
python -m pip install -e '.[analysis,dev,quality]'
python -m ruff format saebase/vllm_backend/sae_interventions.py scripts/workplan11_acceptance_gpu.py
python -m ruff check saebase/interventions/kv_cache_exact.py saebase/vllm_backend/sae_interventions.py scripts/workplan11_acceptance_gpu.py tests/test_clean_kv_exact.py tests/test_interp_vllm_sae_contract.py
python -m ruff format --check saebase/interventions/kv_cache_exact.py saebase/vllm_backend/sae_interventions.py scripts/workplan11_acceptance_gpu.py tests/test_clean_kv_exact.py tests/test_interp_vllm_sae_contract.py
python -m pytest -q tests/test_clean_kv_exact.py tests/test_interp_vllm_sae_contract.py tests/test_suffix_replay.py tests/test_performance_hf_batch.py tests/test_weight_identity_modes.py tests/test_m2_interventions.py
python -m pytest -q
nvidia-smi
~~~

**Exit G1:** exact head CI all required checks success and no feature/provenance contract regression.

## 2. Full end-to-end SAEBase benchmark, not a kernel microbenchmark (P0)

Already implemented: `scripts/workplan11_acceptance_gpu.py e2e`.
Run against Qwen3-1.7B Base, original registered residual TopK SAE, layer 7, the same real frozen 25 selected input samples and **all seven** conditions. Compare `BatchEngine(enable_suffix_replay=False)` vs `True` under the exact same prompt IDs, SAE/weights, position, readout and dtype. Use one warmup + five synchronized repeats for each mode; alternate order or check for order bias, record per-run elapsed, output count, peak allocated/reserved GPU memory and model-forward counts. The existing 1.260x is a smaller edited-forward timing and must not be advertised as E2E.

Requirements:
- By `measurement_id`, same cardinality, operations and sample IDs, no dropped conditions, metrics within precommitted bf16 atol/rtol 1e-2, no nonfinite metrics, independent feature support/ID checks, identity replay PASS.
- Validate **real patch tensors and downstream logits** on a representative per-condition sample, not merely aggregate L2 metrics; a key-matched scalar result can conceal unintended feature changes.
- Confirm suffix cache key `tuple(sample.tokens)` cannot reuse stale hidden when site/dtype/model/attention changes; it is scoped to one model/experiment, cleared appropriately, and works under mixed lengths/duplicates and OOM splits.
- Caution: suffix path temporarily overrides block `module.forward`. Do not run the same model concurrently. Implement an explicit exclusive-execution guard or otherwise enforce single-owner worker access across ALL forward entrypoints before enabling beyond controlled tests. Restoration must occur in `finally`.
- If end-to-end speedup <1.15x, keep opt-in, report truthfully. If parity fails, keep the original path as default.

~~~bash
# Run inside the HF/Torch + SAE Lens environment, from the PR47 worktree.
export WP11_RUN=/workspace/experiments/performance/workplan11/final_5070
python scripts/workplan11_acceptance_gpu.py e2e
~~~

**Exit G2:** `end_to_end.json` with actual five-run timings, exact output parity and VRAM. Speed gate is reported, not assumed.

## 3. True clean-prefix KV-cached SAE steering (P0)

Already implemented: `saebase/interventions/kv_cache_exact.py`, `CleanPrefixKVDecoder`, with **clean prior KV separated from an edited copy**. The next step consumes the clean history for the steered sequence, not a persistent edited history. This is the right scientific protocol but implementation is not yet confirmed on real Qwen.

Do not simply mark `CLEAN_KV_STATUS = BLOCKED` and stop; that older constant describes a pre-implementation state. Test and repair the new candidate.

Execution / numerical acceptance:
1. CPU toy oracle tests must pass for full-prefix vs incremental cache across multiple steps. Inspect `transformers` Qwen3's actual `DynamicCache`, `get_seq_length`, `cache_position`, `past_key_values` and attention-mask contract for the pinned Transformers version.
2. Run **8 then 32 steps on real Qwen** with *forced identical continuation tokens* first. At EVERY step compare original site activation, complete SAE support set, clean logits and SAE-edited logits against `capture_features` + `apply_condition` + `patch_logits` on a freshly recomputed clean full prefix. Zero support flips required; bf16 `atol=rtol=1e-2`; record MAX errors and earliest divergent step.
3. Use an actually active SAE feature selected from the exact prompt's code rather than hardcoded coordinate 185. Include identity and ablation/scale, two distinct prompts, and a non-final hook position if the protocol supports it (otherwise declare unsupported). Then compare independent baseline and steered continuations with frozen greedy / sampled RNG and EOS.
4. If parity fails, inspect which layer/step diverges and whether the prior cache was mutated, the fork was shallow, the attention mask offset was wrong, or a clean step accidentally used edited KV. Fix and rerun; never "solve" it by relaxing tolerance or reusing edited history.
5. The current fork uses `copy.deepcopy(past)` for safety. This may scale linearly with sequence length and can erase the benefit of KV caching. Record clone time, peak allocated/reserved GPU VRAM and forward time independently; after parity passes, optimize with a read-only/forkable clean cache when real Qwen supports it. Do not design a new generic inference framework.
6. Compare real full-prefix reference vs incremental clean-KV steering with 1 warmup + >=5 runs for **8 and 32 steps**. Benchmark stepwise compute *without* including the expensive correctness oracle inside the timed path. Report TTFT, tokens/sec, memory and 1.15x median gate.

~~~bash
python scripts/workplan11_acceptance_gpu.py kv --steps 8
python scripts/workplan11_acceptance_gpu.py kv --steps 32
~~~

**Exit G3:** full forced-token per-step numeric PASS, or a reproducible concrete failure after an actual RTX 5070 run (not unsupported-by-assumption); performance must be measured separately. No default KV enablement without parity.

## 4. A real vLLM hooked SAE backend, not generation-only (P0)

Already implemented experimentally:
- `saebase/vllm_backend/sae_interventions.py`, `InterpVLLMSAE`: capture a real `resid_post` site, encode with the registered SAE, calculate exact residual-preserving edit, apply only the requested prompt token using interp-engine steering, capture output and read out real model logits.
- `scripts/workplan11_acceptance_gpu.py hf-reference` writes a PRIVATE reference file, then `vllm --backend vllm` and `vllm-static` run in SEPARATE processes/environments.
- `tests/test_interp_vllm_sae_contract.py` mocks the upstream structure; mock success does not prove worker-API compatibility.

DO NOT assume `sync.decode_residuals` actually exists or returns calibrated next-token logits in the installed interp-engine. On the REAL pinned version, first inspect the exact signatures and underlying worker implementations for `Address`, `sync_model.capture`, readout/unembedding, `steer`, `AddSpec`, static tap declarations and the meaning of `position_mask`. Replace invented/incompatible calls with documented supported real API **before** running expensive GPU tests. For a legitimate final-logit comparison, read logits from the same edited model execution, applying the correct final norm and lm_head; a raw residual is not itself logits. Do not infer logits from one generated token.

1. Create a separate compatible interp-engine/vLLM/Torch/CUDA environment (never clobber the trusted HF benchmark). Record `pip freeze`, exact package versions, compiler, compute capability and model/SAE hashes. Keep one model + SAE resident and conservative KV/graph memory on 12 GiB.
2. Run Qwen clean capture and compare original residual at layer 7 vs SAEBase HF (including EXACT tokenization, BOS, positions). Verify physical per-request additive write at ONE target token and that excluded token rows remain unchanged.
3. From the same capture/edit execution, read out clean and edited logits (ideally all-vocab comparison). Compare HF original/edit logits and SAE support and per-feature IDs at frozen BF16 bar; include identity, ablate and scale.
4. Run dynamic hooked `vllm`, then `vllm-static` only if it fits VRAM and its static read/write site list is declared before graph capture. If static CUDA graphs or the compatible backend fail with a reproducible OOM/kernel incompatibility, do NOT silently call generation-only PASS.
5. Diagnose first layer of divergence. The previous vLLM attempt had Qwen residual mismatches starting inside block 0 despite matched input; no tolerance relaxation, no hidden quantization, no scientific equivalent claim if it persists. If the backend is numerically different but internally consistent, label it `BACKEND_NATIVE`, not `REFERENCE_EXACT`. Preserve the HF default; a separate backend-native protocol must not reuse its scientific cache identity.
6. If scientific parity passes, repeat five real inference/capture/patch benchmarks, report throughput, p50/p95/TTFT/peak VRAM and compare identical workload. Otherwise report the exact blocker and the minimal feasible remedy, not a vague "not supported".

~~~bash
python scripts/workplan11_acceptance_gpu.py hf-reference
# New separate vLLM process/env (only after verifying actual interp-engine API).
python scripts/workplan11_acceptance_gpu.py vllm --backend vllm
# Again a distinct worker process, conditional on VRAM and declared static points.
python scripts/workplan11_acceptance_gpu.py vllm --backend vllm-static
~~~

**Exit G4:** a real worker capture + SAE edit demonstrated; reference-equivalence claimed ONLY on full logit and feature parity. Plain vLLM generation does not satisfy this task.

## 5. Release decision and files (P0)

Expected private outputs inside `$WP11_RUN`:
- `end_to_end.json` — complete reference vs suffix, five repeats and metric agreement.
- `clean_kv_parity.json` — stepwise oracle and 8/32-step timing and memory. Save separate files or runs so the 32-step test cannot silently overwrite the 8-step evidence.
- `hf_reference_private.pt` — original tokens/activations/logits/weights provenance, NEVER publish.
- `hf_reference_status.json`.
- `vllm_vllm.json`, `vllm_vllm-static.json` — measured backend-native and HF parity, per-operation evidence or reproducible errors.
- `final_report.md` — actual executed commands, SHAs, CI run links, repeated timings, VRAM, true speedups, parity matrices and blockers.

**Important implementation corrections to test harness:** do not overwrite the 8-step KV artifact when running 32 steps; preserve every repetition. Do not report a speed gain based on a microbenchmark that omits capture, snapshot edits, writer or identity checks while calling it end-to-end. The current GPU script has no real CI GPU coverage. Add only the minimum helpers needed to make it reproducible.

Do not merge experimental code before final CI and data. After G1/G2/G3/G4 have honest outcomes:
- Merge PR #45 to main first only if its verified strict-hash and HF tests remain good.
- Rebase/update #46 onto latest main, verify CI and Qwen E2E; merge if safe (suffix remains opt-in if further coverage is absent).
- Rebase/update #47 onto #46 or merged main, rerun tests and CI, merge only validated experimental paths behind OFF-by-default capability gates.
- Public GPU API remains disabled. Do not modify original Stage D/SAR/RSS records.

A credible FINALIZED result explicitly says what is enabled, what runs faster, what remains non-equivalent, and what specific work would be required for previously blocked acceleration. No indefinite tuning loops, but NEVER call a path COMPLETE without a real measured result.

## 6. Agent first instruction

Read PR #47 and this handoff. Protect existing local changes and GPU evidence. **Fix exact Ruff CI failures first**, then run CPU oracle and full Qwen BatchEngine E2E. Run clean-prefix KV 8/32 steps with exact full-prefix comparison, fix Qwen cache/position bugs, and profile clone overhead. Inspect installed interp-engine's REAL API before replacing any incompatible capture/readout/static tap calls, then run dynamic and static vLLM SAE edits in isolated GPU workers. Produce a factual final_report.md, push fixes, get final CI green, and close the PR chain in correct order only for scientifically validated code. Do the work; do not return a new work plan.
