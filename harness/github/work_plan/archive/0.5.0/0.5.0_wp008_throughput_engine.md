# Work Plan 11 — Final High-Value SAEBase Acceleration

Updated: 2026-10-21
Status: Executable final optimization wave. This plan supersedes the previous broad Work Plan 11 scope.
Repository: OpenSAE/SAEBase
Base: Work Plan 10 PR #45 (perf/wp10-execution-reuse, bc9b1fb4638a01be19756d508dc0d1c89277e0b2), after verifying latest CI or merging its numerically validated core.
Hardware: one RTX 5070 with approximately 12 GiB VRAM. Inspect free capacity.
Work Plan 09 PR #44 is merged. Work Plan 10 PR #45 has 87 focused CPU tests, real GPU pilot and a green CI run at the noted SHA.

## Current execution handoff — use this instead of the older broad scope

The concrete Work Plan 11 implementation is now split across stacked PRs #45 -> #46 -> #47. **The current executable RTX 5070 handoff is:**

https://github.com/OpenSAE/SAEBase/blob/perf/wp11-complete-kv-vllm-20261021/harness/github_work_plan/11_final_gpu_validation_handoff.md

It supersedes older "skip vLLM/KV" or "mark BLOCKED without implementation" suggestions. Local agent MUST implement/debug and actually execute (1) final CI lint+CPU tests, (2) 25-row × 7-condition real BatchEngine end-to-end parity and performance, (3) Qwen clean-prefix KV 8/32-step full-prefix oracle and speed, and (4) real interp-engine vLLM SAE capture, position-scoped edit and complete downstream logit/reference parity. Keep incompatible experimental backends disabled; report actual failures instead of invented speedups.

At time of handoff the PR #47 Python 3.11/3.12 offline, package and analysis installation tests passed for commit 4bd66e45, while Ruff formatting failed on two edited Python files. Agent must check the latest CI rather than assume the old result persists. None of the new KV/vLLM GPU paths is yet certified. Do not alter public GPU endpoints or expand to 48 sites.

## 0. Decision: only implement performance work with real payoff

The user wants SAEBase fast, correct, simple to adopt and easy to maintain. This is an implementation plan, not a proposal to build a new inference framework or a new cloud product. Code and run experiments; repair failures; return measured results. The finish condition is a well-characterized efficient implementation on **defined workloads**, not the impossible promise that nobody can optimize it further.

**Mandatory work:**
1. Verify and preserve Work Plan 10's correctness fixes and safe fast-path boundaries.
2. Profile the genuine model-forward bottleneck and implement an exact, workload-relevant computation-reuse fast path **if it passes parity and the speed gate**.
3. Attempt a correct clean-prefix KV-cached SAE steering prototype, with a strictly bounded engineering budget and an objective pass/fail decision.
4. Conduct one bounded accelerated-backend investigation, retaining a safe reference fallback.
5. Run realistic speed/memory/reliability tests and finalize documentation plus a release decision.

**Do not implement unless profiling first proves a meaningful bottleneck:** arbitrary chunk-wide metrics rearchitecture, broad multi-site hook framework, new on-disk job database, multi-GPU scheduler, custom attention/Triton kernels, sophisticated caching layers, or wholesale refactor of API classes.

**Not required in WP11:** 48-site matrix/GPU-05, full Research Lab UI deployment, public GPU endpoint, multi-tenant queue productionization, new graphics/docs redesign, large SAR/RSS scientific reruns. Existing estimators and historical evidence remain unchanged.

## 1. First actions: protect code and freeze the reference (WP11-00)

1. Record current local working tree, SHA, outstanding experiments and uncommitted work. Never reset, overwrite or force-push prior GPU artifacts.
2. Inspect PR #45 and its 87-test + RTX 5070 report. It currently implements strict public sha256_state_dict, explicit VerifiedCheckpointSession, HF capture_with_logits, device-side scale/add, identity decode skip and conservative two-decode edits.
3. Merge the validated WP10 core or branch from the exact PR SHA after CI and numerical acceptance. **Do not run WP11 on a stale main lacking WP10**, and do not wait for nonessential WP10 extras to be invented.
4. Preserve SHA semantics: sha256_state_dict must really hash current bytes; leased_sha256 is an in-process identity shortcut, NOT a fresh cryptographic assertion. Keep trusted workers read-only, in eval/no_grad or inference_mode; do not let user-supplied mutation callbacks share a leased worker. Expose strict mode when arbitrary mutable modules are admitted.
5. Do not build a heavyweight whole-run staging transaction solely to protect against hypothetical within-run .data mutation. If a real untrusted-mutation use case appears, measure/document it separately and opt that workload out of leased execution until an appropriate publication boundary exists.
6. Existing ResultWriter shards/manifest atomicity, identity matching, cache corruption refusal, OOM halving and resume protections remain release requirements. Avoid replacing them.
7. Freeze the model/SAE checkpoint SHA, tokenizer, exact hook/layer/position, input token IDs, prompts, conditions, dtype, seed, execution backend and the immutable scientific protocol.

Target real sites: Qwen3-1.7B Base residual TopK L7 on HF hooks (primary); Pythia-410M L4 and Gemma-270M L5 (cross-backend/SAE architecture checks). R1-distill 1.5B is optional because it uses ~9 GB and does not add an independent performance bottleneck class. One loaded model at a time on the 5070.

**Deliver** environment.json, source_sha.json, workload_manifest.json and reproducible setup/command script. No fabricated speedups or private input strings in GitHub.

## 2. Three required performance workloads (WP11-01)

**W1 — Selected-site capture and SAE encoding.** Frozen 8 short plus 8 moderately longer real prompts, one site; optionally capture multiple layers only when real use actually requests them. Measure capture rows/s, actual forwards, SAE encode rows/s, cache hit/miss, I/O and VRAM.

**W2 — Repeated real feature interventions, primary.** Reuse the exact Work Plan 09/10 Qwen L7 work population (about 25 sample rows × 7 identity/ablate/scale/clamp/add/coalition/reconstruction conditions). Measure baseline+capture forwards, patch forwards, SAE decode calls, CPU/GPU synchronization if traced, throughput and peak VRAM. Record matched sample+condition identities; comparing patch-batch indices is invalid.

**W3 — Autoregressive SAE steering.** Run 8 and 32 new-token steps on the same frozen prompt/token continuation, first baseline full-prefix then experimental cached implementation. Record TTFT, per-token latency, full-prefix work vs cached work, original/edited logits and actual generated tokens. Use reference's separate clean/edited trajectories.

Optional microcases if W2 indicates material real demand: one small SAR coalition collection or RSS variant set using source-locked original estimators. No requirement to build six dashboards or reproduce entire source papers.

For W1/W2/W3, save one warmup + at least five timed repetitions, cuda synchronization around measured GPU runs, all individual timings, max torch allocated/reserved VRAM, CPU RSS, OOM retries, data volume, forward count, first-run and warmed worker time. Clearly separate model-only compute vs checkpoint verification vs disk writer. Compare against Work Plan 10 warm lease and strict mode as appropriate; no misleading extra 2.6x attributed to this wave.

Performance decision: an optimization must maintain exact input/feature IDs and the frozen numerical thresholds and improve its target steady-state runtime at least **1.15x median** to become the new fast default. For small low-risk cleanups below 1.15x, accept only if they remove clear complexity or a verified correctness problem; otherwise revert. A speedup on synthetic input only does not pass.

## 3. Largest likely opportunity: avoid repeating the full model for each edit (WP11-02)

Work Plan 10 reduced HF Qwen from roughly 225 to 200 forwards by merging capture and baseline. The remaining roughly 175 edit forwards dominate the W2 workload. This is the FIRST performance engineering target; do not spend the run optimizing 0.013s SAE encode or the already reduced hash overhead.

### 3.1 Precompute the unchanged early network once, then execute the dependent suffix

Consider a model-specific experimental `prefix_to_site / resume_from_site` primitive only for supported HF architectures:

1. Given one frozen prompt and selected layer L/hook, run the original model once up through the hook, retaining the correct **full sequence** layer-L output and original clean logits/feature snapshot. Maintain exact model load, dtype, RoPE, attention masks, and residual semantics.
2. For each requested edit condition, clone the layer-L output, change only the selected token/site using SAEBase's exact residual-preserving activation delta, and run **only layers after that hook** through final norm and lm_head. Do not recompute layers 0..L. Reuse the clean pre-hook state only across edits of the SAME prompt, tokens, model identity and layer.
3. When a patch is at an earlier token but logit readout is later, downstream attention may depend on it. Recompute every downstream sequence position required by the causal dependency; simply applying the edit to the readout token or restoring only a last-token vector is WRONG.
4. For final-token readouts, use suffix-only incremental execution only where architecture and cache semantics are independently verified. Never assemble a patched output from incompatible cached KV for a different condition.
5. Support tuple-return blocks and model-specific norms/residuals; refuse unsupported architecture rather than silently returning plausible logits. If the required model decomposition is too fragile across upstream Transformers versions, stop implementation and mark this fast path BLOCKED; retain original hooks.
6. Place the experimental path behind a named opt-in and capability check. Public ModelAdapter.capture/patch APIs remain backward compatible. Do not patch Torch or vLLM internals globally.

### 3.2 Prove both correctness AND net benefit

Use a 3-stage pilot, stopping immediately at failure:

- **Stage A, CPU small model:** implement exact block-suffix replay; compare reference logits and edited logits for identity/ablate/reconstruction on final and EARLY patch/readout.
- **Stage B, real Qwen 5070:** 8 frozen prompts × all seven operations. Compare original activation, SAE nonzero IDs (zero disagreement), patched logits and metrics for identical site/condition. The vLLM/HF fused bf16 batch must remain disabled; SAME kernel and numerical boundary are required. Verify LN buffers and all hooked modules cleaned up. Tolerance stays frozen (Qwen bf16: atol/rtol 1e-2), no ad hoc relaxed pass.
- **Stage C, full W2:** one warmup + five repeats, same outputs, bound memory to actual free 5070 VRAM, measure full original vs prefix/suffix. Achieve >=1.15x median for W2 and no material peak memory/p95 regression before enabling a tested fast path. If not, revert prototype, report bottleneck and stop work on it.

The theoretical saving is proportional to the skipped early layers only if there is no large staging/KV penalty. Measure real kernel, copy and Python overhead. Treat fast path as unsupported for untested model families.

### 3.3 Cheap alternatives only when Stage C identifies them

If suffix replay cannot pass, profile whether:
- one model forward can safely evaluate MULTIPLE independent edits in a batch using the same numerical kernel;
- fixed site and same prompt conditions can be grouped without changed TopK support;
- edit tensors/SAE decoder outputs can be coalesced only for mathematically row-independent decoders;
- baseline and original reconstruction cache reuse is safe at exact precision.

The prior Qwen fused bf16 batch had feature-ID flips and remains disabled. Do not redo the same unsuccessful generic CUDA/SDPA flag sweep without a new technical reason.

## 4. Second opportunity: scientifically equivalent KV-cached steering (WP11-03)

### 4.1 Existing contract is nonnegotiable

Read saebase/interventions/steering.py. On each steered token step, it recomputes the **clean entire current prefix**, captures the LAST token's SAE feature, edits only that token/layer, and computes next-token logits. Next step uses the chosen tokens but does NOT reuse edited history KV.

Therefore:
- `clean_prefix_stepwise_edit_v1` is the existing protocol.
- `persistent_edited_history_v1` is a DIFFERENT scientific protocol, not a faster implementation of the same one.

### 4.2 Minimal working prototype, no new framework

1. Start with one real Qwen model and fixed 8-token continuation plus a tiny CPU transformer test. Prototype maintaining **separate clean-history KV** for the growing steered sequence.
2. For each step, obtain the clean current-token activation, apply the same SAE edit and calculate edited next-token logits by redoing the dependent model suffix. Advance the clean cache along the sequence's chosen tokens, not with old edited KV.
3. Preserve positional encodings, attention mask, layer residual convention, RoPE, correct cache ownership, EOS/seed/greedy semantics, baseline vs steered distinct trajectories and prompt token indexing.
4. Compare EACH step's clean and edited logits with full-prefix reference on the same fixed tokens. Check SAE selected feature IDs exactly. Do not claim equivalence from identical generated words.
5. Only after 8-token parity, test 32 tokens and at least two prompts, including early vs late intervention layer, 5 repeated timings, VRAM and TTFT. Gate >=1.15x median and frozen numerical tolerance.
6. Timebox engineering to a defined GPU/wall-clock budget in run_manifest; if backend cannot support correct clean/edited cache separation and suffix replay, mark BLOCKED and preserve reference. Do not create a new full transformer engine for this.

This is a **mandatory bounded implementation attempt**, NOT a requirement to somehow make an unsafe KV algorithm pass. A documented semantic blocker is a legitimate outcome; it cannot be called an accelerated SAE-steering release.

## 5. vLLM/Neuronpedia decision: bounded, only valuable paths (WP11-04)

Work Plan 09's vLLM 0.28 failed original SAEBase Qwen L7 residual parity (difference originates in block 0; attention/norm discrepancies accumulate). Eager interp-engine + SDPA matched the reference across all 28 layers. Those findings must be treated as evidence, not as an excuse for endless compatibility tuning.

1. Choose ONE supported current, isolated interp-engine/vLLM environment only when a documented upstream change plausibly resolves the relevant discrepancy. Verify package pins, CUDA compatibility and 5070 VRAM before loading weights.
2. Run 8 frozen Qwen prompts, same clean logits and site, identity/ablate/scale where supported, source-compatible BF16 precision, and 5 timed repetitions only after parity.
3. A backend failing frozen original feature support or activation/logits tolerances must NOT serve scientific SAE capture/patch data. Stop there and mark it EXPERIMENTAL/NOT_REFERENCE_EQUIVALENT, not PASS.
4. Separately run a **generation-only** vLLM throughput comparison only if W3 or Research Lab actually needs fast ordinary text generation; it has no SAE hooks and MUST NOT be used as evidence of feature intervention parity. If useful and >1.15x faster, offer an explicit generation-only route; otherwise retain the existing implementation.
5. For any prefix cache serving a capture/edit request, validate it does not bypass required hooks or reuse stale edited KV. If unable to prove isolation, disable prefix caching for that request.
6. Do not design an all-purpose ExecutionProvider subsystem. Add a small capability declaration and explicit routing only for backends actually shipped and verified.

A correct but slower eager interp-engine backend is worth retaining as an optional compatibility alternative only when it saves concrete maintenance effort. No claim of faster SAE interpretation unless measured.

## 6. Minimal reliability, no extra infrastructure (WP11-05)

Already implemented: manifest-committed ResultWriter, SHA-256 verified caches, OOM microbatch fallback, strict public hash and trusted session, baseline OOM recovery, model-site identities. Reuse these.

**Required checks on the changed paths only:** injection of OOM on new suffix/KV allocations, finite retry/abort, 12 GiB budget, hook release on exception, no retained CUDA cache across independent jobs, resume with no duplicate measurement IDs, no deleted historical evidence, and explicit errors for unsupported backends.

Do **not** build a second queue, redesign Research Lab, add multi-tenant server, GPU pool, new object store or continuous background worker. Its public GPU POST endpoint remains disabled. If no scheduling/serving capability is introduced, no service load test or new worker lease subsystem is required.

WP10 note: existing lease enables fast reads only in a **trusted read-only worker**. Lack of a whole-run transaction does not establish vulnerability in the current static evaluator without a mutating code path. Keep current ResultWriter and enforce strict hashing for untrusted/mutable usage; if an actual mutation during execution is demonstrated in a supported workload, create the smallest separate correctness fix before claiming a strong release guarantee. Do not mislabel a lease signature as a fresh checksum.

## 7. Final evidence, stop rule, and user deliverable (WP11-06)

Report only three primary workload families W1/W2/W3 (plus an optional real SAR/RSS microcase if actually run):
- pinned code/model/SAE/input/precision/backend identities and VRAM;
- individual warm runs, median / p95 with sufficient observations, throughput, allocated/reserved memory, forward counts, cold/warm/verifying modes;
- numeric parity by sample+condition, max_abs/max_rel, ZERO feature-support flips when a scientific equivalent is claimed;
- change-by-change measured speed and maintenance complexity, honest FAIL/BLOCKED rows;
- major phase time fractions and Amdahl headroom: if a remaining measured phase cannot plausibly produce 1.15x overall, do not spend a new project on it.

Stop when:
- strict public checksum and the trusted-session distinction remain correct;
- all enabled fast paths pass frozen numerical and feature identity tests, and missing capabilities are refused explicitly;
- the major genuine forward/suffix and steering bottlenecks have been attempted with results; failed attempts remain disabled;
- VRAM/OOM/recovery/CI are clean for the selected RTX 5070 workloads;
- no proven high-return optimization with realistic available implementation is left untried; explain unresolved bounds rather than pretending universal optimality.

Do not reopen completed WP09 trials or require exhaustive 48-site coverage. Publish compact, evidence-grounded harness/github_development_doc/performance/workplan11_results.md and update harness/github_development_doc/performance.md to describe actually enabled defaults. **Do not publish new stable version without tested gates.**

Minimum commands from root (adapt only after inspecting repository state):

~~~bash
git fetch origin
git status --short
# If WP10 PR #45 is merged, branch from origin/main.
# Otherwise branch from the approved, exact WP10 PR SHA on a separate worktree.
git switch -c perf/wp11-final origin/main  # only after WP10 is merged
python -m pip install -e ".[dev,quality]"
python -m pytest -q
python -m ruff check saebase/ models/ saebase/tools/
# New tests written by agent: do not run nonexistent test paths until created.
python -m pytest -q tests/test_suffix_replay.py tests/test_clean_kv_steering.py
~~~

Private output directory:
~~~text
experiments/performance/workplan11/<run_id>/
  environment.json
  manifest.json
  parity.jsonl
  timings.jsonl
  memory.jsonl
  errors.jsonl
  throughput_table.csv
  report.md
~~~

Keep private prompts and weights out of GitHub. Commit source tests and public-safe redacted report on a dedicated PR. The final report must state precisely: what is faster, by how much for which workload, what remained numerically incompatible, and which bottlenecks were deliberately skipped because evidence showed little payoff.

## 8. Copy-ready execution instruction to the local GPU agent

Read Work Plan 11 at harness/github_work_plan/11_high_throughput_engine_and_final_acceptance.md and the current WP10 PR #45 and GPU evidence.

Implement the high-value tasks in sections 1–7, starting with a matched RTX 5070 forward/patch profile and a minimal exact suffix-replay prototype. Prioritize real model-forward reduction and reference-equivalent clean-prefix cached steering. Never disable feature-identity validation to make a benchmark faster. Cap GPU memory and experimentation budgets, fix real discovered bugs, and do not replace existing API/queue infrastructure.

Skip low-return synchronization rewrites, multi-site frameworks, heavyweight output transactions, public GPU serving and 48-site sweeps unless a new actual workload demonstrates they are required.

Create tests, run real-checkpoint parity, benchmark old vs new with 5+ repetitions and per-phase counters, commit accepted changes, open a PR, and deliver throughput_table.csv plus report.md and exact blockers. Only advertise acceleration that actually passed.

Complete a truthful engineering final acceptance, rather than returning another strategy document.
