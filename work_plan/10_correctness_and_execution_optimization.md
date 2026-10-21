# Work Plan 10 — Correctness, Execution Reuse, and CPU/GPU Synchronization

Created: 2026-10-21
Status: Execution handoff, NOT a claim of implementation or GPU validation.
Repo: OpenSAE/SAEBase
Base: current verified main (record exact HEAD at start); PR #44 is merged, merge commit 41c51f4f77dfa37e9d771050d7294e5af9fbd083.
Next: Work Plan 11 (high-throughput engines and final engineering acceptance).
Hardware: one actual RTX 5070, ~12 GiB VRAM; inspect available memory rather than assuming 12 GiB free.

## 0. Mission

Make SAEBase's existing research execution path correct, bounded, reusable, and fast without changing its science. This is a CODING AND TESTING plan. The local agent must implement, run, fix, commit, and report changes. Merely describing opportunities does not count. Prefer exact small improvements over broad refactoring.

**Design policy:** science and cache provenance MUST remain valid. The original published model/SAE outputs, feature-index identities, SAR/RSS/Localize estimators, intervention semantics, dataset groups and snapshot identities remain untouched. Do not change third-party SAE APIs or public Research Lab GPU availability. Preserve users' current scripts, existing flags and wire formats.

A key fact established by Work Plan 09: RTX 5070 Qwen3-1.7B Base L7, in the repeat-evaluation workload, approximately 2.057 seconds were 225 model forwards and only 0.013 seconds were 25 SAE encode calls. Repeated state hashing accounted for ~3.175 seconds before PR #44. The now-merged hash cache removed most repeated hashing, but **its signature is not a proof of unchanged tensor bytes**: PyTorch .data edits and alias writes can bypass a Tensor's _version. Fix this correctness risk first.

Work Plan 10 should deliver a validated speedup from eliminating unnecessary forwards and synchronizations, but report speedups as MEASURED, never promised. Work Plan 11 will decide whether any larger inference acceleration remains reachable on the 5070.

## 1. Protect source, freeze evidence and define measurement modes — WP10-00

1. Inspect current Git status, origin/main HEAD, working trees and any local-only experiments. Create a feature branch from current main; never reset, force-push or overwrite private data.
2. Read these real source files BEFORE coding:
   - saebase/store/identity.py: sha256_state_dict, _STATE_SHA_CACHE, _state_signature.
   - saebase/interventions/engine.py: BatchEngine.__init__, _baselines, _patched_logits, _metrics, OOM retry and ResultWriter journal.
   - saebase/api/model.py: ModelAdapter and hook site restrictions.
   - models/hf_hooks_backend.py: get_resid, run_with_patch, run_batch.
   - saebase/api/features.py: capture_features, edit_features, _decode, LN-stat state.
   - saebase/interventions/operators.py: apply_condition and scalar .item use.
   - saebase/store/collect.py: capture_selected, collect_activations, encode_from_store.
   - saebase/interventions/steering.py: reference full-prefix semantics.
   - docs/performance/workplan09_results.md and tests/test_performance_hf_batch.py.
3. Freeze three modes, **do not confuse them**: COLD (model load + checkpoint verification), WARM FIRST EVALUATION (loaded model, fresh output folder, original full verification cost), and WARM REPEATED EVALUATION (loaded verified model, fresh output folder, verified cache reuse).
4. Include a fourth timer called MODEL-ONLY measuring model forwards and patching without SHA construction or result writer. Change one factor at a time.
5. Choose deterministic existing real registered sites: Qwen3-1.7B Base L7 (HF hooks, bf16 TopK), Pythia-410M L4, Gemma3-270M L5 (JumpReLU), and R1 1.5B L20 only if VRAM allows. One model loaded at a time.
6. Freeze sample groups, actual IDs, lengths, patch/readout locations, conditions, dtypes, model/SAE checkpoint hashes, environment and repetitions. Use the same 8 public-safe short pilot prompt groups from Work Plan 09 where legitimately available. Save raw restricted inputs privately only.
7. Require 1 warmup + >=5 timed repeats, synchronized boundaries, full raw repetition series and allocated/reserved VRAM. Run the same INPUT+CONDITION populations and measure work done, not only script wall time.

**Deliver:** private experiments/performance/workplan10/<run_id>/ with environment.json, frozen_workload.json, phase_before.jsonl and baseline.md. Commit the public-safe benchmark harness.

## 2. Repair state-hash correctness without losing the performance gain — WP10-01 (P0)

Current issue in saebase/store/identity.py: cached SHA-256 is returned on matching data_ptr, shape, stride, storage_offset and Tensor._version. This does NOT detect all content changes. A module or adapter's state_dict may also include buffers, shared storage or nonstandard tensors. The existing test only covers parameter.add_(), which increments _version.

**Hard invariant:** whenever SAEBase claims an ACTUAL CONTENT SHA-256 verified at a boundary, it must come from a full byte hash of the loaded state at that boundary, or from an explicitly documented, auditable, trusted immutable execution context with a valid verification lease. _version-only matching must NEVER be silently described as a fresh content checksum.

Implementation design, in this order:

A. Add an explicit **strict** complete-content hashing entrypoint or mode whose semantics do not depend on mutable cached metadata. Preserve historical byte serialization/digest exactly (name, dtype, shape, contiguous tensor bytes). Do not break published historical hashes. The existing public sha256_state_dict contract should be unambiguously strict by default unless a backwards-compatible documented change is proven safe. Do not rely on caching for tests meant to check integrity.

B. Introduce an explicitly named VerifiedCheckpointSession / immutable identity lease for performance-sensitive evaluations. Mint it ONLY after full content hashes are checked at a trusted boundary; pin model and SAE instance identity, checkpoint source revision, device/dtype, statically captured tensor metadata and execution configuration. Only opt into cached hashes inside this session and ONLY in a process that owns weights, forbids training/weight edits, and treats user callbacks or arbitrary mutating plugins as untrusted. Session identity cannot be exported as an independent proof of current arbitrary tensor bytes.

C. Define the trust boundary honestly: Python cannot guarantee immunity from arbitrary .data, memory alias, custom CUDA writes or foreign code that mutates storage. A shape/ptr/_version signature is a fast *heuristic*, not proof of byte immutability. To give a strong public provenance guarantee for potentially mutable modules, run the strict hash at the result-release boundary and compare to the session start, refusing publication on mismatch; use a trusted isolated model worker for accelerated mode. A change that occurs and is reverted between checks is undetectable by end checks alone, so do not promise a cryptographic continuously immutable state without an enforced isolation boundary.

D. Keep model weights separate from mutable per-request SAE LN buffers. Never cache a hash of a live stateful module if its state_dict can change during capture. If an architecture's state_dict mutates under legitimate inference, disable fast identity reuse and report explicitly.

E. Ensure exact exception behavior in ResultWriter: staging results may be computed, but no result described as verified/committed or shared outside the run before the required identity gate passes. If old on-disk results exist, preserve integrity and never silently rewrite their provenance.

Tests MUST include all cases:
- Repeat unchanged model (digest same) and normal in-place parameter.add_/copy_ update.
- parameter.data.add_ with unchanged _version, .data assignment/rebind.
- Aliased shared storage mutation (where feasible), state_dict buffer mutation, load_state_dict, parameter replacement, dtype/device move, weight sharing and distinct module instances with same tensors.
- A loader that mutates after hash verification, failure at strict release gate, aborted result publication.
- A no-mutation trusted session with unchanged digests and no extra host copy on each engine construction.
- Restart process, rehydrate checkpoint, and verify that a stale in-process lease is never reused across loads.

Tests assert exact digest matches a fresh independent full-byte baseline after every mutation. If a proposed shortcut cannot meet this, keep the strict API and downgrade the shortcut to explicit trusted-session only.

**Gate WP10-01:** no stale successful CONTENT verification under adversarial mutation tests; no silent fast-cache trust in mutable public API; warmed repeated-evaluation performance does not regress catastrophically (measure and report).

## 3. Unify capture and baseline logits in ONE HF forward — WP10-02 (P1)

Today the HF serial branch of BatchEngine._baselines calls capture_features() and then ModelAdapter.forward_logits() for each uncached sample. Both execute full model forwards. For each sample requiring capture, a single model forward can produce:
  (a) the hook's original activation at patch_position and
  (b) the baseline logits at readout_position.
With 25 samples this can avoid ~25 forwards out of Work Plan 09's 225 forward calls; measure actual benefit, do not assume an overall multiplier.

Implementation:
1. In models/hf_hooks_backend.py, add a **single-example** capture_with_logits or run_with_capture_and_logits primitive; hook the exact registered layer, record h at the requested token and return the forward's logits from the same call. Preserve tuple outputs, always remove hooks in finally, avoid returning full-sequence activations to Python callers.
2. Keep the same unpadded, single-example HF kernel that already passed bf16 Qwen TopK parity. Do NOT turn on the previously failed fused padded GEMM or enable_hf_masked_batch by default.
3. Add a small capability method to ModelAdapter rather than branching on an incidental class name, e.g. supports_capture_with_logits / capture_with_logits. Existing models transparently fall back to capture + forward_logits. API additions optional, no existing API removal.
4. In _baselines, call the new primitive only when activation_reader is absent; when cached activations are present, keep one baseline forward unless a trusted, input-bound baseline-logit cache is already verified.
5. preserve exact patch/readout token indexing (including earlier-than-final sites), masking, prefix length, model eval mode and the same output dtype. Follow Work Plan 09's known same-width identity replay rules and compare logits by sample+condition identity rather than patch batch index.

Test on 8 prompt groups, equal and varied lengths, early and final positions, zero/identity/nonzero ablations; compare original activations, code nonzero ID set (not just approximate values), baseline logits, edited logits and metrics. Inject hook exception and verify cleanup. Assert one forward per uncached sample rather than two, using a counting backend. If the real Qwen hook changes TopK IDs, revert ONLY this optimization.

**Gate WP10-02:** Qwen feature-ID flips exactly ZERO and original/edit logits within frozen bf16 bar; Pythia/Gemma metrics unchanged; fewer measured HF forwards; benchmark phase and total speed.

## 4. Remove redundant SAE decode and synchronous per-condition scalar work — WP10-03 (P1)

Current observations:
- apply_condition() uses float(codes[i].item()) for scale/add, creating per-feature device-to-host synchronization.
- edit_features() decodes the original SAE codes for every edit, even though FeatureSnapshot.reconstruction already holds the original decode.
- _metrics() uses several individual .item() operations and int argmax for each condition; repeated GPU synchronization can dominate many small interventions.
These are candidates for measurable fixes, not license to change the mathematical intervention.

Implement safely:

A. Reuse the snapshot's original decode for all edits under a verified deterministic SAE decoder + snapshot provenance contract. Formula is EXACTLY original SAEBase semantic h_new = h + decode(z_edit) - decode(z_orig), preserving reconstruction residual. For layer_norm SAEs, always restore snapshot-specific ln_mu/ln_std and clear mutable buffers in finally. Do not replace arbitrary SAE decoders with a linear D @ delta approximation unless that SAE has proved affine decode and a separate parity test.

B. Keep scale/add values on the GPU as zero-dimensional tensors, using z[selected_ids] and vectorized index/where operations; never force values through Python float. Batched edits can be built as [conditions, width] or an equivalent sparse representation with bounded GPU memory. Handle 0/1/many selected features, repeated support, large d_sae, empty identity, coalition membership, exact dtype.

C. Batch condition decode ONLY for decoder implementations proven row-independent. SAE Lens LN-stat restoration and any per-batch normalization must be checked; otherwise group by site and use legacy individual decode. For identity, do not unnecessarily decode when the original h is unchanged. For full_reconstruction, continue returning snapshot.reconstruction, not h+delta.

D. Vectorize logits metrics with torch.float32 GPU reductions, a single synchronized finite-check and a bounded bulk CPU transfer per chunk. Preserve exact metric definitions, zero-negative-KL clamp policy and condition ordering; store JSON/JSONL unchanged.

E. Prevent an allocation explosion: build one bounded chunk of edited activations at a time, execute, compute metrics, journal, then free. No all-conditions-all-prompts dense tensor retained.

Failure injection and parity fixtures:
- Synthetic affine SAE, TopK, JumpReLU, layer_norm normalized SAE and residual-containing model; identity, ablate, scale, clamp, add, coalition, full reconstruction.
- Original decode value reproduced under LN, no leaked ln_mu/ln_std state between distinct prompts.
- For every original and optimized output, match logits/target token differences/KL/L2 and feature IDs; do NOT permit distributional similarity as a substitute for exact mask parity.
- Include CPU count of cuda sync sites or Nsight/PyTorch trace and actual wall time.

**Gate WP10-03:** bounded memory, unchanged original metrics/semantics, measurable reduction in decode and synchronization counts; no default change for unsupported SAEs.

## 5. Eliminate wasted model work across activation sites and data pipelines — WP10-04 (P2, conditional)

Before coding, instrument the full workflow across collection, SAE encoding, intervention, reliability and reporting. Existing collect_activations() performs one forward per SAMPLE per site; this is a candidate for reusing a single original model forward for multiple requested layers and positions.

A. Where the backend genuinely supports it, introduce capture_many_sites(input_ids, requested_sites_and_positions) that registers all requested hooks for ONE model forward. Store only selected rows, not [all tokens x all layers] in CPU memory. Identity for each site includes exact layer/hook/input/token positions; never mix incompatible residual sites or model weights.

B. Add bounded group planning: same model state+token IDs+tokenizer+inference dtype+execution semantics may share an unmodified forward. SAE checkpoints can reuse activation shards only when their required site and source identities match. Do not reuse one model forward for different prompts, re-tokenized variants or interventions.

C. Group cache reading and SAE encoding by rows, but preserve exact per-sample sample_id, group/split, sparse/dense layout and source bytes' SHA; do not accidentally use the same SAE code for a different checkpoint.

D. Add a scope-specific counter: separate model forwards from SAE decode, extra copies, CPU hashes and cache IO. Use a table with the number of avoided executions and memory/latency impact. Avoid complex generic schedulers in this first round.

**Gate WP10-04:** works for at least two available sites in the same model and same input without cross-site contamination; lower total forwards vs two independent captures, peak memory within budget, tested correct cache invalidation. If the workload rarely requests multi-site capture, document low impact and skip invasive implementation.

## 6. Recovery, planning and release quality — WP10-05

1. OOM handling must cover baseline/capture, identity replay and patch loop. Retries must halve a BOUNDED chunk, avoid infinite loops, remember safe microbatch for similar length groups, and emit clear failure at width=1. Do not classify arbitrary RuntimeError as OOM.
2. Keep exact identity-gate protocol: compare at matching batch shape and cached logits, not against a new narrower reference whose GEMM reduction changes. On OOM rebalance/rebaseline the corresponding subset explicitly and record it as a separate measurement context.
3. Flush/journal atomically. Inject SIGTERM/KeyboardInterrupt, duplicate request IDs, cache corruptions, out-of-disk, and concurrent writer attempts; ensure no silent lost rows, duplicated committed IDs or incorrect source identity.
4. Do not enable public POST /api/jobs. Existing Research Lab CPU queue, lease fencing, data redaction and cancellation controls remain the foundation; no new competing job system.
5. Establish stable counters and profiling utility in scripts/perf/ or scripts/: model_forwards, captured_rows, SAE_encodes, SAE_decodes, patch_forwards, cpu_gpu_syncs_when_instrumented, host_copies, disk_bytes, peak_vram, retries, cache_hits/misses and verification mode.

## 7. Concrete tests and performance acceptance

Use these commands as a minimum; update actual paths only after inspecting the current repo:

~~~bash
git fetch origin
git switch -c perf/wp10-execution-reuse origin/main
python -m pip install -e ".[dev,quality]"
python -m pytest -q tests/test_m1_store.py tests/test_m2_interventions.py tests/test_m3_execution.py tests/test_performance_hf_batch.py
python -m ruff check saebase/store/identity.py saebase/interventions/ models/hf_hooks_backend.py
# Additional new tests to create:
python -m pytest -q tests/test_weight_identity_modes.py tests/test_capture_with_logits.py tests/test_intervention_vectorized.py tests/test_perf_resume.py
~~~

Use a source-pinned baseline (Work Plan 09 merged main), matching precision and full fresh output directories. On RTX 5070 execute Qwen L7, Pythia and Gemma where model access exists. For each workload compare 5 repeat samples plus warmup and separate:
  - STRICT full verification / cold session / warmed session.
  - baseline capture+logits / new fused single-forward capture+logits.
  - old per-condition decode+metrics / new chunked decode+metrics.
  - repeated per-site captures / multi-site capture when applicable.

Acceptance gates:
- P0: no stale hash falsely claimed as current content SHA under adversarial weight mutation.
- Identity: record hashes, sample/position/condition IDs, data source and original model/SAE checkpoint exact.
- Scientific: zero TopK/JumpReLU selected-feature identity disagreements, no unintended changed directional signs; frozen numerical thresholds, unchanged SAR/RSS estimator and logit metrics by stable keys.
- GPU: finite valid outputs, bounded 12 GB memory and correct OOM/resume behavior.
- Performance: report separate strict, warmed repeated, and model-only gains; work on hot paths must remove a measured bottleneck and show a reproducible improvement. An optimization is NOT accepted merely for reducing lines of code or time in a synthetic-only microbenchmark.
- Regression: no material p95 degradation, unbounded startup/VRAM increase, or extra cache misses in common case.

A >=1.15x median gain is a target for a materially affected HOT PATH, not a mandatory project-wide number for every correct micro-optimization. If gain <1.15x, retain only clearly safe low-maintenance fixes with demonstrated execution-count savings; otherwise revert.

## 8. Required files and stopping rule

Deliver in a private run directory:
- run_manifest.json: source/base/candidate hashes, model, SAE, tokenizer, inputs and package pins.
- before_after.jsonl: 5+ repetitions, strict/relaxed session mode, GPU/CPU phases and VRAM.
- parity.jsonl: each real site/sample/condition with exact IDs, max_abs/max_rel, feature-set flips.
- failures.jsonl: every OOM, expected exception and recovery outcome.
- breakdown.md: model forwards, hashes, cache accesses, decoded conditions and sync costs.
- report.md: final PASS/FAIL/BLOCKED per WP10-00 through WP10-05, code changes, remaining limits.
- a public-safe redacted summary at docs/performance/workplan10_results.md.

Create targeted CPU unit tests plus real-GPU scripts, add CI with tests that can run without CUDA (GPU tests self-skip with explicit status), make small commits on a WP10 branch, and open a PR. Do not publish private prompt text, token IDs, weight files or secrets. Do not merge if P0 hash correctness or mandatory scientific parity fails.

**Agent first instruction:** audit current sha256_state_dict and reproduce a .data mutation stale hash, build safe strict/trusted identity modes and tests, then run the Qwen capture+logits one-forward prototype and compare reference tensors. Continue to intervention decoding, metric synchronization, multi-site reuse if beneficial, and full release gates. Implement and execute rather than returning a new proposal.

**Completion boundary:** Work Plan 10 is complete only when measured evidence, corrected hash semantics, real GPU parity and an auditable PR are delivered. It does not certify vLLM GPU-06 or full high-throughput generation; those are Work Plan 11.
