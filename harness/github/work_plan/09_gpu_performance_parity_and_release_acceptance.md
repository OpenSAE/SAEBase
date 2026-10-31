# Work Plan 09 — GPU Performance, Numerical Parity, and Release Acceptance

Created: 2026-10-20  
Status: READY FOR LOCAL GPU EXECUTION. No GPU performance or real-checkpoint parity claim is made by this document.  
Owner: OpenSAE / SAEBase.  
Primary code: [performance PR #44](https://github.com/OpenSAE/SAEBase/pull/44), branch **perf/throughput-reliability-20261020**.  
Historical comparison: pre-optimization main commit **339fbb603bc6d999fd126a91ea8709fee1c599c2**.  
Related: [performance notes](https://github.com/OpenSAE/SAEBase/blob/perf/throughput-reliability-20261020/harness/github_development_doc/performance.md), [verified model matrix](../github_development_doc/verified-model-matrix.md), [Work Plan 05](05_research_lab_cpu_closure_and_gpu_handoff.md).

## 0. Mission and nonnegotiable rules

**Execute the GPU verification; do not stop after writing another plan.** Determine whether PR #44 improves real SAEBase workloads while leaving their scientific meaning, data identities, metrics, and failure behavior intact. Fix discovered implementation errors on the PR branch, rerun the relevant checks, and produce reproducible results with exact commit/checkpoint identities.

PR #44 currently implements only these changes:

1. Coalesced SAE encoding of previously captured activation rows inside a bounded store read batch.
2. Explicit masked Hugging Face (HF) batch capture/patch interface, integrated into the intervention BatchEngine. The older one-prompt adapter remains a fallback.
3. Retention of the reduced patch microbatch after a CUDA OOM, instead of repeatedly retrying the original unsafe size.
4. Offline parity tests, an encoding microbenchmark, and engineering documentation.

**Do not claim that activation collection has already been batched, that cached generation has been accelerated, or that vLLM/interp-engine integration is present.** Those are separate prospective optimizations. The public Research Lab GPU endpoint must remain disabled.

Scientific correctness overrides throughput. Do not alter SAE checkpoint weights, normalizations, intervention operators, SAR/RSS/Localize estimators, split assignments, prompt text, feature identities, or source manifests merely to make new measurements match.

Preserve existing local changes and previous experiments. Before checkout: record current branch/status/HEAD, inspect untracked work, and create independent worktrees or a protected copy. Never run destructive reset, force-push, delete raw measurements, or replace the live Hugging Face Space as part of this plan.

## 1. Target environment and resource preflight — G0

Expected GPU: **one NVIDIA GeForce RTX 5070 (nominal 12 GB VRAM)**. Inspect the *actual* GPU, driver, free/total VRAM, host CPU/RAM and free disk; do not assume the previous machine's CPU, RAM or storage capacities still apply. Run one model and its required SAE at a time, use short prompts and small microbatches, and verify the true free VRAM before loading any checkpoint. Never attempt the full 48-site sweep in this work plan.

Record:

- UTC start time; machine hostname class (no private host identifiers in public reports), OS, Python, CUDA driver/runtime, torch, transformers, SAE Lens, TransformerLens, vLLM if installed.
- GPU name, compute capability, physical VRAM, free VRAM, memory pressure; CPU/RAM/disk free; available model/cache directories.
- Git base SHA and candidate SHA; dirty/clean state; source weights, SAE revisions/checkpoint SHA-256, tokenizer version and hook names.
- Environment differences. Build a clean candidate environment; do not change an already working system CUDA/torch installation without a reproducible isolated environment.
- Maximum allowable GPU use, disk growth, model downloads and total job wall time. Put actual numeric limits in the private run manifest before launching GPU work. Prioritize smaller registered checkpoint/SAE pairs that demonstrably fit the RTX 5070 without quantization; a quantized checkpoint is a separate experiment identity. Do not launch unbounded tasks.

Start with the existing GPU environment if it satisfies package requirements. Run source tests before expensive model downloads. The following commands are examples to execute from repo root, using separate worktrees for the two code versions:

~~~bash
git fetch origin
git worktree add ../SAEBase-wp09-base 339fbb603bc6d999fd126a91ea8709fee1c599c2
git worktree add ../SAEBase-wp09-candidate origin/perf/throughput-reliability-20261020
cd ../SAEBase-wp09-candidate
git rev-parse HEAD
python -m pytest -q tests/test_performance_hf_batch.py tests/test_m1_store.py tests/test_m2_interventions.py tests/test_m3_execution.py
python saebase/tools/benchmark_encoding.py --device cuda --rows 2048 --width 1024 --features 32768 --repeat 5 --batch-sizes 1,8,32,128,256
~~~

Worktrees must not overwrite an existing directory. Install this checkout in a separate Python virtual environment before running its CLI. If a command or dependency has changed, inspect its actual help/API and document the substitution, rather than guessing its behavior.

**Gate G0 PASS:** candidate source and baseline are pinned, CPU tests and package installation are green, real GPU is detected, available resources and explicit budgets are recorded. If not, fix the preflight issue first; no large workload starts.

## 2. Scientific and performance design — fixed before measurement

### 2.1 Two code versions, one workload

Compare the old implementation at pinned base commit and the new implementation at pinned candidate commit. Hold constant the actual model and SAE checkpoint, layer/hook, input tokens, tokenizer/chat template, activation/code cache identity, dtype, patch/readout token position, conditions, seeds, and hardware. Retain a baseline without batching **and** the optimized candidate under identical experimental protocol.

Use the existing verified Stage D 64-group records when possible. If the newer Research Lab 500-group archive is available, record its **separate** identity and permissions. Never pool the old 64-group population with the newer 500-group population. Do not create new prompt labels or silently substitute datasets when an archive is absent. Recreate only legitimately available data under the original frozen source revisions.

### 2.2 Focused, RTX 5070-sized representative coverage

Use a **small, fixed real-checkpoint pilot** with distinct token lengths and at least two patch positions per site; include two right-padded mixed-length batches and at least one equal-length batch. Begin with a small registered HF site, such as a compatible Pythia-410M or Gemma-3 270M SAE, then a second model/SAE family if it fits. Use `harness/github_development_doc/verified-model-matrix.md` to identify real supported model–SAE combinations; never invent a pairing.

**Required target:** 3 real model–SAE sites spanning at least 2 compatible architectures/backends when the 12 GB card permits, with 8 distinct prompt groups/site, varied lengths and token positions, and all supported edit operations. Check model + SAE + overhead VRAM before each load; if an individual site cannot fit or its weights are unavailable, record **BLOCKED** with measured evidence and substitute another *registered* small site. Preserve the initial list and record substitutions explicitly.

**GPU-05 is intentionally removed.** Do **not** run the 48-site compatibility sweep, a full-corpus research integration expansion, or the old Tier B/Tier C matrix. Spend the recovered GPU time on the required GPU-06 accelerated-backend and steering experiments.

Freeze the chosen site/sample list *before* seeing speed or result differences. Keep failed and unsupported cases in the denominator. The requested research validation is targeted, not a new claim that all 48 sites were retested.

### 2.3 Numerical acceptance rules

For each sample and condition compare, before aggregation:

- Input token IDs, tokenizer identity, attention mask, valid length, hook/layer, patch/readout positions, prompt-group and split IDs: **exact**.
- SAE checkpoint identity, source activation SHA-256, record count, feature width, row order and code cache provenance: **exact**.
- Capture activations, SAE codes, decoded edits, baseline/edited logits, target-token changes, KL and L2 measures: compute max absolute, max relative and a meaningful normed error. Save differences even if they pass.
- Identity intervention and unchanged logits, zeros/threshold crossings, selected feature IDs and direction signs: independently check. A near-threshold feature flip is an explicit scientific discrepancy, not just an innocuous floating-point error.
- SAR/RSS/Localization estimators: unchanged source-pinned estimator, masks, train/held-out indices, independent prompt groups, ranking and result interpretation. Recompute from actual measured arrays, not from summaries.

Initial candidate numeric tolerances are fp32 **atol=1e-5, rtol=1e-4** and bf16 **atol=1e-2, rtol=1e-2** for logits/activations. Record precision and the applicable tolerance *before* the run. This is a starting protocol, not permission to relax a threshold to pass an observed failure. Always report both absolute and relative errors and require **identity patch parity** under the original SAEBase protocol. Stronger exactness checks on IDs, positions and discrete masks are not replaced by numeric tolerances.

### 2.4 Fair timing and resource reporting

Run warmups separately, then at least 5 repeated timed runs per candidate configuration with `torch.cuda.synchronize()` around GPU timing boundaries. Record each repetition, not just the fastest result; report median, range, p95 where measurable, peak allocated and reserved GPU memory, OOM retries, disk size, model-forward count, processed real tokens/rows, and cold-start separately.

Workload families and units:

| Workload | Exact comparison | Primary throughput unit |
| --- | --- | --- |
| Cached SAE encoding | per-record baseline vs bounded batched candidate on same activation store | encoded activation rows/s |
| HF baseline + capture | legacy single-prompt capture/forward vs masked candidate | real prompts/s and forwards |
| HF patched logits | legacy sequential patch vs candidate microbatch across fixed conditions | interventions/s |
| Whole Benchmark | identical SAEBase configuration and report outputs on both commits | completed measurements/s |
| Research Lab SAR/RSS | same original estimator and actual inputs; no synthetic result substitution | completed prompt groups/s |
| Steering and high-speed backends (GPU-06) | reference HF/TL vs verified interp-engine/vLLM path, plus reference vs a tested cached-steering prototype where possible | output tokens/s, prefill/decode latency and exact per-step logit error |

Do not call the included `saebase/tools/benchmark_encoding.py` an end-to-end benchmark: its random matrix is synthetic and isolates the encode kernel only. End-to-end evidence must use actual model/SAE checkpoints.

**Performance gate:** correctness is mandatory. To label a change a meaningful acceleration, target at least 1.15x **median warmed throughput** on its intended real workload without unacceptable memory growth, failures, changed metrics or sharply worse tail latency. If it is neutral or slower, document the regression and keep that path opt-in or revert it; never cherry-pick a faster-looking subset after seeing the results. This 1.15x is an engineering selection threshold, not an expected measured speedup.

## 3. Ordered GPU execution packages

### GPU-00 — Protect source, verify CI, record the real environment

1. Inspect CI for PR #44 at the exact candidate SHA; no unreviewed failing code/format/package gate can be ignored.
2. Record worktrees, clean dependencies and registered model/SAE sites.
3. Perform the focused representative-site preflight and a tiny no-network cached experiment before downloading weights.
4. Write private `run_manifest.json`, `environment.json`, `site_plan.json`; choose bounded resource limits.

**Output:** environment report, immutable base/candidate identity and a fixed experiment matrix. **Stop gate:** no safe runnable target or missing exact checkpoint identity.

### GPU-01 — Prove batched cache encoding is equivalent

1. Capture or reuse a **single verified activation store**. Compute both old per-sample encodings and batched encodings into separate output directories; never overwrite source shards.
2. Check exact number of samples, positions/rows, source hashes, code identity, CSR/dense layout readback and resume behavior.
3. Cover small/large feature counts, different number of selected token positions per sample, all-zero features, last partial batch, batch sizes 1/8/32/128/256 where memory permits, and a cache where some samples already exist.
4. Exercise real ReLU/TopK/JumpReLU/other supported SAE variants. Check that batch-dependent normalization or selection does **not** alter per-token codes; if it does, gate that SAE class to the original safe serial path.
5. Benchmark repeated warm timing and peak memory with identical cache and dtype.

**PASS:** scientific parity plus correct resume/provenance. If real SAE results differ, produce a minimal counterexample and patch the batching eligibility logic instead of masking discrepancies.

### GPU-02 — HF capture, masking and intervention parity

1. For each selected real HF-backed site test single prompt vs batched input on equal and mixed right-padded lengths, including nonfinal patch/readout positions and the final valid token.
2. Compare capture **and** baseline logits from the one-pass batched call against individual reference calls. Confirm no logits depend on unrelated padded tokens.
3. Run identity, ablation, scale, clamp, add, coalition and full-reconstruction conditions wherever the real SAE/site supports them. Compare both the patched tensors and logits with the original one-example path.
4. Confirm unsupported left padding, invalid positions, missing hook site, nonfinite patches and unsupported architecture all **fail explicitly**, never return plausible-but-wrong values.
5. Make sure hook handles are removed after a successful call and after an injected model exception. Repeated calls must not accumulate hooks or persistent GPU allocations.
6. Run the existing TransformerLens path unchanged and verify no regression in its outputs.

**PASS:** per-example scientific parity, correct masking/error behavior, true batched forward count and no retained hooks/memory leaks. A syntax-compatible hook with incorrect model output is FAIL.

### GPU-03 — OOM recovery, bounded work, resume and stability

1. Test OOM handling using a deterministic injected OOM first, then a **bounded** real memory-stress pilot. Do not fill VRAM uncontrollably or crash the host to create a demonstration.
2. Confirm reduced microbatch is retained on later chunks; no duplicate/missing result IDs, stale partial writes, changed condition order, or altered sample grouping.
3. Also test baseline/capture OOM and the identity-replay path; they are **not automatically covered** by PR #44's current patch-loop improvement. Fix or explicitly mark any gap.
4. Interrupt one resumable workload and resume it without recomputing committed rows. Inject corrupted cache data and require fail-closed integrity behavior.
5. Verify per-run budgets and quota/cancellation boundaries remain effective; leave public `POST /api/jobs` disabled.

**PASS:** finite retries, bounded VRAM, correct committed measurements after resume, explicit failure under unsupported conditions.

### GPU-04 — Measured acceleration, regression identification and tuning

1. Execute base and candidate on exactly the frozen small RTX 5070 pilot workloads with warmup and repeated timing. Do not add a 48-site expansion.
2. Sweep batch sizes, token lengths, number of feature edit conditions, and SAE width. Record observed best **safe** size, not just the largest batch.
3. Measure time spent in tokenizer/model forward, activation capture, SAE encoding, patch forwards, D2H transfers, metric reduction, I/O, checkpoint hashing and model load where instrumentable.
4. Prefer interventions/s and encoded rows/s over nominal batch size. Report old/new forward counts and effects of reusing cached activations.
5. Produce an honest per-model comparison table: where faster, where neutral/slower, where OOM, and exact numerical parity status.

**PASS:** a traceable quantitative result for every tested path. A speedup claim requires the predefined gate; otherwise deliver a measured limitation and a concrete fix.

### GPU-06 — REQUIRED real GPU tests of accelerated backends and steering

**GPU-06 is mandatory and follows GPU-04 directly; GPU-05 is not executed.** Run its actual experiments on the RTX 5070 after recording the GPU-00 environment and completing the targeted GPU-01 through GPU-04 baseline work. If an earlier candidate regression is unresolved, isolate GPU-06 using the verified reference environment instead of skipping it. The current PR does **not** yet contain an interp-engine adapter or KV-cached steering implementation: the agent must build and test the smallest safe compatibility prototype where feasible, not claim one exists.

**GPU-06A — Dependency and memory feasibility.**

1. In a separate clean Python environment, inspect the **current** compatible `interp-engine` and vLLM versions and their torch/CUDA requirements. Do not reuse SAEBase's pinned vLLM 0.27.1 installation if it conflicts. Save full package locks, GPU driver info and a runnable setup command.
2. Select a real, registered small-model SAE that fits a single RTX 5070 in both reference and experimental environments. If a requested backend is unsupported or exceeds 12 GB, record the exact package exception / memory reading and test the next smallest compatible configuration. Do not silently quantize an existing checkpoint to force a fit.
3. Measure peak resident VRAM including model, SAE, KV cache, temporary activations and graph buffers before raising batch size. Use bounded token lengths and one concurrent workload first.

**GPU-06B — Eager, vLLM and interp-engine functional comparisons.**

1. Actually load and run an **eager HF/TL reference** and at least one viable **accelerated vLLM/interp-engine path**. For the same frozen tokens, measure unmodified forward logits and generation, using equal precision and deterministic decoding.
2. Where each backend advertises capture or patch capability, test the exact registered layer/site, original activation capture, identity patch, one feature ablation and scaled edit. Compare tensor shapes, positions, masked inputs, logit differences and error metrics to the trusted reference.
3. Explicitly probe regular hooked vLLM, static-point capture and CUDA Graph replay **where supported**, checking declared-site restrictions and that missing hooks report unsupported rather than silent success. A generation-only backend is evaluated for text generation throughput but must **not** count as having verified feature capture/patch.
4. Verify cache isolation: a regular generation request must not reuse stale steered KV state; a capture request must not lose hook outputs due to a prefix-cache hit.
5. Repeat a small deterministic workload at least five times after warmup; record real tokens/s, time to first token, peak GPU memory, successful requests, failures and backend overhead versus reference. No claimed speedup without matched timings.

**GPU-06C — Cached-steering correctness test.**

1. Preserve the current full-prefix SAEBase steering implementation as the oracle. At each generation step it recomputes the **clean** prefix and edits only that step's current token/site. This differs from simply persisting modified KV entries.
2. Implement a minimal experimental cached decode or suitable backend adapter where technically supported, **with an explicitly named intervention protocol**. Compare the same supplied token continuations against reference **step by step**, before stochastic sampling. Check identity, patch location, baseline and edited logits, stopping conditions and cache isolation.
3. If protocol-equivalent cached steering is possible, execute at least five comparable GPU repetitions and report tokens/s, latency, memory, and maximum logit difference. If only persistent-steered-history generation is available, label it a **different protocol**, report its scientific divergence, and do **not** call it a speedup of the existing reference.
4. If a correct cached-steering prototype cannot be implemented within the bounded setup, still **run a real reference steering generation and a controlled unsteered vLLM generation baseline**. Mark the cached-steering parity and speedup cells **BLOCKED** with the exact technical reason; do not substitute paper design or hypothetical results for a test.

**GPU-06D — Deliverables and acceptance.**

Create `backend_matrix.csv`, `backend_parity.jsonl`, and `steering_steps.jsonl` with one row per backend/condition/step, plus a concise `gpu06_report.md`. Each attempted backend must be **PASS**, **FAIL** or **BLOCKED**, recording what actually executed. Include selected package pins, real runtime errors, GPU memory and comparable timing samples.

**PASS:** at least one real accelerated backend was actually run on the RTX 5070 and tested against a matched eager reference, all tested capture/patch claims satisfy their scientific parity gates, and the steering reference/accelerated comparison has explicit protocol and measured evidence (or a documented cached-steering blocker). A generation-only backend never certifies SAE intervention equivalence. If the accelerated backend itself cannot run due to the environment, GPU-06 remains **BLOCKED**, not complete.

**Keep new backends experimental until their parity and performance acceptance is met.** Do not enable CUDA graphs, quantization, alternative prefix caching, or KV-cached steering by default just to improve apparent throughput.

## 4. Implement fixes during the run, but preserve scientific contracts

On discovering a real problem, capture a tiny failing test first, fix it on the optimization PR branch (or a linked repair PR), run the related CPU tests, rerun the failing real GPU case, and record the exact before/after SHAs. Preserve external/public APIs and previous release tags. Do not update main or merge PR #44 merely because a microbenchmark improved.

Priority order for repairs: corrupted/incorrect outputs > OOM or resume errors > high-frequency CPU/GPU synchronization > model compatibility > tuning. Do not perform unrelated Research Lab UI redesign, branding or synthetic feature-label generation here.

## 5. Required file outputs and final report

Write results to a **private** experiment directory such as `experiments/performance/workplan09/<run_id>/` (gitignored, or keep it outside the checkout). Proposed structure:

~~~text
run_manifest.json           exact code/checkpoint/data identities, budgets, seeds
environment.json            verified GPU, CUDA, torch and installed dependencies
site_plan.json              frozen representative sites, groups and test modes
parity.jsonl                one row per real site/sample/condition with errors and status
benchmark_runs.jsonl        all warmed repetitions, GPU memory and counters
backend_matrix.csv          GPU-06 backend, feature support and execution status
backend_parity.jsonl        real accelerated-backend logits, capture/patch errors
steering_steps.jsonl        reference vs cached/prototype step-level comparisons
gpu06_report.md             required GPU-06 tests, timings and blockers
failures.jsonl              unsupported cases, errors, OOM and recovery outcomes
summary.csv                 comparable baseline vs candidate rows, not fabricated averages
report.md                   release decision, speed table, parity matrix and caveats
private_inputs/             raw restricted prompts/token-level arrays, never public
~~~

A public, **redacted** `harness/github_development_doc/performance/workplan09_results.md` may be committed only if it contains exact reproducible summary evidence without third-party source texts, private prompts, secrets, model weights or unauthorized raw datasets. Link to individual CI jobs and any public compatible evidence bundles with verified checksums; keep private artifacts private.

Every comparison row must contain: code SHA, model/SAE revision/hash, hook/layer, backend, precision, workload group and sample count, batch size, median wall time, throughput, peak GPU memory, error metric, result status and reason if blocked. Missing observations must never be converted to zero or silently dropped.

The last `report.md` must answer all questions plainly:

1. **Which representative real model–SAE sites were verified on the RTX 5070?** List each PASS/FAIL/BLOCKED outcome; the 48-site expansion is excluded.
2. **Did exact feature identity and numerical behavior remain valid?** Include largest errors and threshold/feature-selection disagreements.
3. **What improved in real workflows, and by how much?** Compare matched inputs, repetitions and VRAM on the RTX 5070.
4. **Where did batching fail or regress?** Which fallback should remain default?
5. **Did OOM, interruption and cache recovery preserve outputs?**
6. **What actually ran in mandatory GPU-06?** State interp-engine/vLLM backend variants, capture/patch support, stepwise steering parity, runtime, speed and specific blockers.
7. **What commits were made to the PR, and what CI/GPU runs verify them?**
8. **Final merge decision for PR #44 and experimental GPU-06 path:** PASS / BLOCKED / FAIL separately, with remaining blockers.

## 6. Final gates and stop conditions

| Gate | Required acceptance |
| --- | --- |
| G0 Environment | Frozen inputs/commits, real GPU and declared budgets |
| G1 CI | Python 3.11/3.12 offline, lint/format, package and regression checks green for final candidate SHA |
| G2 Numerical parity | All required representative real sites pass identity, capture, code, edit and logit checks |
| G3 Recovery | No silent row loss/duplicate; safe OOM fallback, resume and cache-integrity failure behavior |
| G4 Performance | Full per-run records and speed/memory comparisons; meaningful improvement labeled only where demonstrated |
| G5 Mandatory GPU-06 | Real RTX 5070 accelerated-backend execution and reference parity, steering reference/prototype comparison, GPU-06 output files; unsupported paths explicitly BLOCKED |
| G6 Release | Public-facing results redacted; no incompatible source/quantization/semantic claims; GPU-06 not silently skipped |

**If G1/G2/G3 fails, do not merge the acceleration path as a default.** Record the failure, fix what can be fixed, and rerun from the failing gate. If a checkpoint is unavailable, the GPU does not have enough VRAM, or the run would exceed the declared budget, mark that cell BLOCKED and continue other independent sites. Do not invent numbers or downgrade correctness to meet a speed target.

At the end, push tested fixes and the concise public-safe report to the appropriate GitHub branch, and return the PR URL, last commit SHA, CI link, number of real pilot-site passes, **mandatory GPU-06 backend/steering results**, speedups/regressions, open blockers and recommended merge decision. **Only describe actual executed work as completed.**

## 7. First instruction to the local GPU agent

**Read Work Plan 09, inspect the actual RTX 5070 and PR #44, protect local work and create base/candidate worktrees, check CI, run CPU parity, then execute the focused real GPU pilot. Complete GPU-01 through GPU-04 and MUST execute GPU-06 tests for interp-engine/vLLM and steering, documenting actual parity and performance or precise blockers. GPU-05 and the 48-site sweep are out of scope. Deliver `report.md`, `summary.csv`, `backend_matrix.csv`, `backend_parity.jsonl`, `steering_steps.jsonl` and `gpu06_report.md`, plus a PR update. Preserve private data and never publish raw inputs.**
