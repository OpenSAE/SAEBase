# Work Plan 11 — High-Throughput SAE Execution, Generation, and Final Engineering Acceptance

Created: 2026-10-21
Status: Second and FINAL intended optimization implementation wave, to start after Work Plan 10's P0/parity gates.
Repo: OpenSAE/SAEBase. Input: Work Plan 10 code/benchmarks and the merged Work Plan 09 reference.
Hardware: RTX 5070, ~12 GiB. Record actual capacity and limit all GPU workloads; 48-site expansion and GPU-05 remain out of scope.
Mission: make performance production-quality for the defined research workloads, with an evidence-based end state. This plan does NOT promise mathematical global optimality or that future hardware/models cannot admit further speedups.

## 0. What counts as FINISHED

The result must be runnable and measured: capture, SAE coding, multi-condition intervention, SAR/RSS replay, and user-facing steering/generation have a correct default path, an efficient optional path where justified, bounded GPU memory, deterministic source identities and honest capability discovery.

A speedup matters only when:
(1) the same intended scientific protocol is measured,
(2) exact feature support and baseline/edited effects match its reference within pre-registered tolerance,
(3) actual model/SAE weights run on the target GPU,
(4) warmed throughput/latency improve with all relevant memory and retries recorded,
(5) installation/resume/CI gates succeed.

Work Plan 09 established:
- Qwen fused masked bf16 batching changed SAE TopK feature IDs; it is DISABLED by default. Row-wise same-kernel capture/patch matches but does not accelerate.
- Eager interp-engine with SDPA matches Qwen registered residuals across all 28 layers. This is a valid alternative reference candidate, not evidence of faster generation.
- vLLM 0.28 bf16 produces residual differences starting inside block 0; Qwen L7 is outside frozen SAEBase parity tolerances, some deeper residuals diverge strongly. DO NOT relabel it scientifically equivalent to HF.
- The previously reported ~2.36x-2.91x workflow gains mostly eliminated repeat state-hash overhead. Work Plan 10 fixes the integrity hazard and tunes REAL forwards.

## 1. Evidence-first system architecture — WP11-00

Do not rewrite the public SAEBase API. Build a narrow execution-provider boundary exposing only real capabilities. Suggested (adapt to existing ModelAdapter):

~~~text
ExecutionProvider
  capabilities()          {backend_id, supported_points, capture, edit,
                           multi_site, batch, reference_clean_kv,
                           generation, dynamic_hooks, static_graphs, dtype}
  prepare(config, weights_identity, budget)
  capture(tokens, sites, selection)
  capture_and_logits(tokens, site, patch_position, readout_position)
  evaluate_edits(tokens, site, edit_spec[], metric_spec)
  generate(tokens, generation_spec, intervention_spec?)
  close()
~~~

Create an immutable ExecutionProtocolIdentity in run manifests, including model+SAE state hashes, tokenizer/chat-template, architecture, backend, exact revision/dependencies, dtype, attention implementation, cache policy, hook convention, edit operator, decoding strategy and seed. A changed backend/dtype/attention kernel is a changed execution protocol even if the upstream checkpoint is identical.

Explicit capability tiers:
- REFERENCE_EXACT: same model/site/edit semantics and frozen numerical parity demonstrated.
- BACKEND_NATIVE: internally reproducible backend with its own declared numerical protocol but not claimed equivalent to the HF/SAEBase scientific population.
- GENERATION_ONLY: no SAE feature capture or interventions allowed.
- UNSUPPORTED/BLOCKED: reject explicit request; never quietly fall back to a scientifically different backend under the same experiment identity.

Backend matrix must drive routing and error messages. No generic "supports vLLM" boolean; declare real site/dtype/operation availability and which workload it was tested on. Preserve the prior strictly identified activation stores.

## 2. Task-specific performance baselines and hard budget — WP11-01

Freeze at least six workloads and assign each a reference:
- W1: selected-token activation capture from a real registered site; 8 short and 8 longer prompts, 1/2/4 sites where defined.
- W2: a fixed 25-sample × 7-condition intervention benchmark (same exact Work Plan 09 Qwen site).
- W3: SAR K16 and 384 coalition masks for ONE real prompt/site if source-pinned masks and allowed inputs are available; otherwise choose a smaller K and state it; distinguish measurements from fitted estimator time and do not imply full SAR paper replication.
- W4: RSS variants from a frozen public-safe input with selection and held-out variants explicitly separated; do not fabricate "meaning-preserving" labels or conflate variants with independent groups.
- W5: unmodified generation, same 32-token and 128-token greedy continuations where VRAM permits, one vs multiple concurrent requests.
- W6: reference SAE steering of 8/32 new tokens on the SAME generated-token path and with two separate baseline/edited continuation traces.

Measure cold load, first verified run, warmed evaluator, capture, actual model forward, decode, metric reduction, cache I/O, time-to-first-token, decode tokens/s and throughput at concurrency 1/2/4/8 only as memory permits. Use 5+ timed repeats after warmup, randomized AB/BA order, synchronizations for compute timing, and a new output directory for each run. CPU overhead versus device kernel time MUST be distinguishable.

Add a lightweight tracing helper and a source-pinned JSON schema for benchmark results; compare model-only improvements separately from avoiding repeated checkpoint hash work. Track allocated AND reserved VRAM and process RSS, token count, forward count, cache hits/misses, OOM attempts and result-journal commits.

Establish measured *headroom*: for each workload report stage costs and a simple Amdahl bound. If a stage consumes fraction p of observed time, eliminating JUST that stage yields theoretical speedup <= 1/(1-p). Do not add independent maximum-speed gains as though they all combine linearly.

## 3. Full, honest accelerated backend investigation — WP11-02

### 3.1 Choose reference and experimental engines

REFERENCE: SAEBase original loaded HF/TL model, matching attention and hook definition, plus interp-engine eager with SDPA where proven to match. Do not compare two different model revisions or new quantized checkpoints as identical.

EXPERIMENT: newest tested compatible interp-engine vLLM, vllm-static or vllm-generate in isolated dependency environments. The exact vLLM/Torch/CUDA version pin must be recorded and validated against current upstream release notes. Do not mix incompatible pinned extras with SAEBase's existing generation stack. Never silently install one CUDA toolkit into another working environment.

On RTX 5070 ~12 GB, load one model/SAE combination per process, stop if memory budget cannot fit graph buffers and the advertised SAE cache. Use small frozen Qwen checkpoint first. Size experimental VRAM using actual model weights, SAE residency, staging activations, KV blocks, graph pool and a safety allowance.

**Upstream reference:** Neuronpedia currently uses separate eager / dynamic vLLM / static-tap vLLM / generation-only backends. Static capture needs DECLARED read/write points at graph recording, and a generation-only graph serves NO arbitrary SAE hooks. Do not equate normal vLLM generation throughput with intervention throughput.

### 3.2 Numerical diagnosis: end the endless kernel-tuning loop

Work Plan 09 already established Qwen:
- identical layer-0 input;
- differences start in block 0, with attention-kernel and normalization effects;
- vLLM residuals outside frozen bf16 tolerance;
- different attention/norm controls tried.

Do not blindly repeat more variations of the same test. Give this investigation a BOUNDED matrix:
- one compatible pinned vLLM release/environment, one independently built version if a known relevant fix is documented;
- two registered sites (one Qwen site + one architecturally distinct small site if accessible), with 8 frozen prompts and patched and unpatched cases;
- eager vs vLLM dynamic vs static if available, plus generation-only as separate task;
- record layer 0, early, target and final layer capture, clean logits, identity, one-feature ablate/scale, nonzero SAE feature support, streaming generation;
- vary numerical kernels/dtype ONLY one at a time with a fixed scientific contract. No posthoc tolerance adjustments.

If residual parity is still outside frozen tolerance after this bounded investigation, mark vLLM SAE capture/steering as NOT REFERENCE_EQUIVALENT and STOP trying to force it into the same scientific identity. It can remain GENERATION_ONLY or explicitly BACKEND_NATIVE with distinct reproducibility ledger if usable. No bogus success. Do not modify scientific metrics to compensate for kernel differences.

### 3.3 Independent generation-only acceleration

Where vLLM generation-only fits and runs, benchmark prompt prefill, greedy continuation, single stream vs concurrency. If it beats the reference while preserving the generation protocol (tokens and stop behavior or appropriately labeled numerical deviation), expose it behind an explicit generation-only provider. Its output must NOT be used as SAE downstream intervention evidence.

Prefix cache only when appropriate: unmodified, stable input prefixes may share KV. For any capture or edit request, enforce isolation using proven provider-specific salt/disable-prefix-cache controls; a cache hit cannot skip a required hook. Test alternating original, edited, original requests and verify no state leakage. Do not enable caching by default for scientific intervention paths without this test.

**WP11-02 acceptance:** real accelerated paths attempted with binary PASS/FAIL/BLOCKED, matched timings and memory. A numerical failure is recorded and disabled by default, not silently merged.

## 4. Exact or explicitly different KV-cached steering — WP11-03

### 4.1 Freeze the scientific reference

Read saebase/interventions/steering.py. It generates two independent continuations. At each step of the steered trace, it recomputes the *CLEAN FULL PREFIX*, applies a feature edit only at the current last-token site, propagates it through downstream layers, samples next token, then repeats clean-prefix recomputation for the new token sequence. Therefore, the reference DOES NOT persist edited KV for past positions.

Define two protocols and never conflate:
- clean_prefix_stepwise_edit_v1: the reference above; preferred target.
- persistent_edited_history_v1: a different research question, opt-in only; cannot count as reference-accurate acceleration.

### 4.2 Engineering of a clean-prefix-compatible cache

Implementation idea, subject to backend capability:
1. Maintain an independent **clean-history KV cache for the steered sequence's own token history** (not the baseline sequence, which can diverge).
2. To prepare step t, use the clean-history KV from positions [0,t) plus the current token, with all past tokens unedited. Obtain current-layer activation and clean next-token logits as the reference would.
3. Compute SAE encode and the exact residual-preserving patch on the current layer/token.
4. Obtain edited logits by injecting at that current token and re-executing the DEPENDENT suffix of the network with the **correct clean past-key/value for downstream layers**. A patched activation at layer L must not rewrite older positions. Do not reuse downstream current-token cache created from the unpatched state when computing edited logits.
5. Separately advance the CLEAN history state for the chosen next-token sequence. The KV retained for subsequent steps must be the clean-state KV, never the current edited response's downstream cache.
6. Use the actual architecture's RoPE positions, causal masks, attention layout, GQA/MQA tensor dimensions, dynamic sliding window and last-token conventions. Reject unsupported families rather than approximating. Clamp/truncate context consistently with reference.
7. Do not claim exact equivalence from token agreement alone; validate logits, activation and SAE feature support every step.

An initial prototype can use a custom HF adapter and small model, without full vLLM graph machinery. Avoid adding a new generic inference engine if an existing backend safely exposes clean-prefix cache and selective suffix replay.

### 4.3 Acceptance and fallback

Construct reference tests with supplied fixed token continuations (force identical steps rather than depending on greedy decoding). Include:
- 8 and 32 steps, short and long prefixes; 2 distinct groups; middle-layer and later-layer interventions;
- identity, ablate, scale, add, coalition where supported; EOS, greedy, and temperature >0 using frozen RNG;
- distinct clean and steered continuations, repeated prompts, cache hit isolation and cancellation;
- per-step original/edited logits and SAE selected-feature identities, plus final token trajectories.

Run the reference WITHOUT KV caching and prototype on same GPU/precision with identical exact weights; compare per-step tensors under the frozen bar and zero discrete-feature selection flips. If no backend can provide safe selective suffix recomputation, leave reference steering unchanged, label cached-steering BLOCKED, and still finish the independently useful generation-only acceleration.

Measure TTFT, per-token latency, peak VRAM and output tokens/s at 8/32 tokens, with 5+ repeat series. A genuine cached-steering speedup requires >=1.15x median at a scientifically equivalent protocol with no loss of numerical integrity; otherwise keep reference as default.

## 5. SAE-specific algorithmic hotspots — WP11-04

These experiments depend on Work Plan 10's correct snapshot/decoder operations.

A. SAR coalition workloads: benchmark time per 384-mask response collection, count actual model forwards, cache decode(unchanged) and baseline once per original prompt, compute only the edited SAE contributions per coalition. Do not switch to a linear additive approximation of MODEL BEHAVIOR; that is exactly what SAR evaluates. For decoder proven affine, an explicit decoder-only delta shortcut may be tested as a separate fast path, but use the same full downstream model after each coalition patch. If multiple edit conditions are independent, batch where the real kernels preserve feature/logit parity; forbid Qwen fused bf16 by default.

B. RSS variants: tokenize each actual variant once, freeze selection and held-out groups, stream capture only at measured target positions, reuse full prompts/cache only with compatible identities, and keep responsive-support estimator unchanged. Do not treat candidate wrapper modifications as meaning-preserving without review.

C. Feature exploration: avoid materializing all d_sae codes for display when a verified sparse representation already exists. Display indices, amplitudes and source-backed text from paginated/verified stores. Do NOT truncate the stored scientific codes to UI top-k: query/display top-k only.

D. Add grouped tasks by model checkpoint, hook/site and SAE to reduce repeated weight loading. Cache one verified activation store and reuse across multiple compatible SAEs; bounded LRU for SAE residency only when model+SAE+KV+temporary tensors fit 12 GiB. On the reference 5070, single loaded model is the safe default.

E. No bespoke Triton or CUDA kernel just because it exists. Only implement if profiling proves a specific kernel/group consumes >=20% of W1-W6 steady time, and a minimal prototype plus parity test demonstrates substantial benefit. Otherwise external engines are the preferred place for kernel maintenance.

## 6. Queue, memory, and concurrency architecture — WP11-05

Do NOT turn Research Lab's read-only public API into an unauthenticated inference server. Its private JobQueue already supports idempotency, per-owner quota, worker leases, retries and attempts. Add only verified worker interfaces and budget enforcement, retaining existing public POST /api/jobs=503 until separate deployment authorization.

One owner process per GPU:
- job spec with immutable model/SAE/site/protocol identity and opaque input reference;
- admission computes upper bound of prompt tokens, #conditions, output tokens, estimated temporary memory; refuse overload before loading model;
- queue model/site-local work to retain model weight residency; fairness prevents starvation;
- separate read-only CPU UI from trusted GPU worker; unauthorized requests may not download checkpoints or schedule unlimited compute;
- cancellation checked at sample/chunk boundaries, worker lease fencing checked before result commit;
- pinned worker output identity, atomic journal and cleanup on cancel/OOM;
- bounded admission for concurrency: experiment at 1/2/4/8 simultaneous requests only if VRAM headroom exists; distinguish queued latency from execution time;
- health checks: liveness separate from an authenticated small real inference readiness probe, with timeouts and no public GPU open endpoint.

GPU memory budget should record: model/SAE resident, KV, hook/capture staging, metric scratch, CUDA graph pool if used, PyTorch reserved cache, and headroom. OOM failures reduce batch/concurrency and keep that safe state until resource conditions change; never restart infinitely at a known-unsafe width.

Test kill -9 simulated worker failure (not host GPU driver), duplicate/retried job, expired lease, cancelled task, race for same idempotency key, and disk full. All committed metric entries must remain durable and exactly once under measurement_id; pure retry may repeat compute but not publish duplicate identities.

## 7. Measured efficiency envelope and FINAL STOP RULE — WP11-06

An honest near-final engineering claim is about a FROZEN WORKLOAD, HARDWARE, SCIENCE PROTOCOL and supported execution backends. "No top engineer can optimize it" is unprovable. Replace that slogan with falsifiable exit criteria.

For W1-W6 provide:
- best parity-verified candidate vs original reference, including actual forward counts, warm/cold wall time, throughput, peak memory, time-to-first-token, p50/p95 and failures;
- three run modes: strict-current-state verification, trusted immutable session, and pure model-only execution;
- measured phase breakdown summing approximately to total (unattributed time reported, not swept away);
- a lower-bound discussion: minimum necessary model forwards under the accepted API, and Amdahl headroom for removing each remaining measured component;
- limits of concurrency/scheduler: first-token/p95 cost and resource safety, not just best aggregate tokens/s;
- baseline and new result artifacts pinned to SHAs and environment.

Try only plausibly high-value alternatives identified by profiling, not random tuning:
1. correct single-forward HF capture+baseline from WP10;
2. sparse/bounded SAE patch vectorization from WP10;
3. exact clean-prefix stepwise cached steering;
4. interp-engine eager/dynamic/static backend where parity proven;
5. generation-only vLLM where it is actually the requested task;
6. maximum safe concurrency, batch buckets and model residency.

Definition of 'performance converged' for the evaluated surface:
- all P0 integrity/correctness and recovery checks pass;
- each public API operation uses a tested default path, fails explicitly for unsupported acceleration and keeps old callers working;
- each W1-W6 has a measured best candidate and headroom decomposition; if one cannot run or has a scientific blocker it MUST remain BLOCKED and cannot justify a global "all accelerated" claim;
- on supported hot paths, no remaining UNTRIED change with an Amdahl upper bound >=1.15x is ignored without a documented safety/engineering reason;
- for any attempted change yielding >1.15x measured improvement, implement and revalidate; for alternatives consistently <1.15x, document bounded residual headroom and close;
- no rising OOM, tail latency, wrong feature IDs, unverified private source leakage or unresolved weight-hash trust boundary.

This is a defensible terminal milestone for the selected configuration, not a universal speed ceiling for future architectures/hardware.

## 8. Release gates, files, and verification

### Release gates

G0: WP10 merged or its reference-safe code pinned with all P0 gates passing.
G1: Actual backend capability registry, provenance and backend-specific scientific contract tests complete.
G2: Reference-exact SAE capture/patch behavior and feature IDs PASS for every backend offered as equivalent.
G3: Cached steering semantics demonstrated stepwise or explicitly NOT AVAILABLE, with reference path preserved.
G4: Memory/scheduling bound verified for single RTX 5070, no unbounded GPU job entry.
G5: End-to-end W1-W6 table, repetitions, CPU/GPU split, p95, peak VRAM, model-only and verification costs all recorded.
G6: Packaging/CI clean; no fake speedup claims; public reports omit private inputs and secrets.
G7: Each advertised acceleration has a supported workload and a tested on/off fallback; anything scientifically blocked is not default.

### Artifact structure

~~~text
experiments/performance/workplan11/<run_id>/
  environment.json
  backend_capabilities.json
  workload_manifest.json
  numerical_parity.jsonl
  generation_step_parity.jsonl
  benchmark_repetitions.jsonl
  profiler_breakdown.json
  gpu_memory.jsonl
  failure_recovery.jsonl
  throughput_table.csv
  residual_headroom.md
  acceptance_report.md
~~~

Public redacted documentation: docs/performance/workplan11_results.md and an updated docs/performance.md, reflecting verified capabilities with a user-facing CPU/eager/accelerated backend selection guide.

### Minimum regression commands

~~~bash
git fetch origin
git switch -c perf/wp11-high-throughput origin/main
python -m pip install -e ".[dev,quality]"
python -m pytest -q
python -m ruff check saebase/ models/ scripts/
# Add focused backend and GPU tests with clear CUDA skip:
python -m pytest -q tests/test_execution_backends.py tests/test_steering_kv_parity.py tests/test_gpu_admission.py
~~~

Run a real RTX 5070 pilot before each speed claim; save actual commands, CUDA/torch/interp-engine/vLLM pins, weights and SAE identities. Repeat reference vs optimized execution with aligned prompt groups, patch conditions and seed. Never compare different model kernels as if they reproduce the same SAE feature experiment when the validated residual diverges.

Implementation must be a sequence of small commits with a PR and final decision. Do not publish a new stable version until verified correctness, benchmark and release tests are green. Do not retroactively change old Stage D archive or Work Plan 09 measurements.

**First command for the local agent:** read Work Plan 10 report and current main, freeze exact RTX 5070 workload matrix W1-W6, implement ExecutionProvider capability-based routing, measure vLLM generation-only and interp-engine backed capture/patch in isolated environments, prototype reference-equivalent clean-prefix cache steering, and perform a final six-workload release review. Run experiments, commit tested fixes, and return measured successes, BLOCKED paths and exact remaining headroom.

**Final report must say explicitly:** what is fast today, what remains slow and why, which accelerations were rejected for science, and whether the defined reference hardware/workloads have reached the performance-converged stop rule. Work Plan 11 is final only after that evidence, not merely after creating this document.
