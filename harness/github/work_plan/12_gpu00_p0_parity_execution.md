# WP12 GPU-00 — P0 Real-Weight Parity Test Handoff

Date: 2026-10-21
Status: **GPU AGENT REPORTS 4/4 LEGACY FP32 SITE PASSES; INDEPENDENT VERIFICATION AND P0 ACCEPTANCE PENDING**
Branch: feat/wp12-p0-foundation (PR #50, not merged into main)
Parent: [WP12 master blueprint](12_unified_interpretability_master_blueprint.md)
Protocol: [WP12 GPU agent handoff](12_gpu_agent_handoff_protocol.md)
Source audit: [WP12-00 inventory](12_p0_source_inventory_20261021.md)
Frozen authored inputs: [eight prompt pairs](12_gpu00_prompt_pairs.json)
Agent-reported execution and next independent checks: [GPU-00 report](12_gpu00_agent_report_20261022.md)

> Execution update: on source SHA `490b18c`, the GPU agent reports Pythia-70M L3/L5 and Gemma-3-270M L5/L9 all PASS using legacy TransformerLens 3.9 / SAELens 6.54 (fp32), with 85 CPU tests passed, 283 matrix forward equivalents and no relaxed tolerances. The raw run directory is not accessible to the independent reviewer, so this does **not** close GPU-00. TL4 was not tested. See the linked report for preserved failure and negative-control details.

## 0. Authority, boundary and result we need

The owner requests that a **GPU agent execute the next small WP12 test**. That authorizes only C0 host checks plus one bounded C1 GPU-00 smoke on real, publicly available model/SAE weights. It does not authorize C2/C3 runs, full 48-site repetition, long training, external API spending, renting hardware, changes to deployed Hugging Face Spaces, release publication, or data redistribution.

**Goal:** establish that current SAEBase's declared legacy-capable backend can reproduce the same underlying native TransformerLens/SAELens operations for model/SAE/site identification, tokenization, raw activation capture, SAE encoding/decoding, identity/no-op patching, clean-to-corrupt patching, one actual feature ablation and cache identity. A successful existing legacy worker is **not** TransformerLens 4 compatibility proof. WP12/P0 remains open until the independent CPU reviewer also signs its other gates.

The GPU agent is the **runner**, not the independent reviewer. Keep unrounded raw measurements and failures. Do not change SAR, RSS, Input Localization, cached experiment records, the public website, or prior Work Plans.

## 1. Site and sample matrix

Use only the following registered sources from the current SAEBase registry. Load and unload **one model/site at a time**, not four weights at once.

| Case | Model key | SAE profile key | Layer | Exact expected hook | Why |
| --- | --- | --- | --- | --- | --- |
| A1 | pythia70m | pythia70m_resid_sm | 3 | blocks.3.hook_resid_post | Small established Pythia M0 reference |
| A2 | pythia70m | pythia70m_resid_sm | 5 | blocks.5.hook_resid_post | Second registered Pythia site; new WP12 coverage |
| B1 | gemma270m | gemma270m_scope_resid | 5 | blocks.5.hook_resid_post | Official Gemma Scope 2, prior Stage D L5 evidence |
| B2 | gemma270m | gemma270m_scope_resid | 9 | blocks.9.hook_resid_post | Official Gemma Scope 2, prior Stage D L9 evidence |

- A uses EleutherAI's Pythia-70M deduped base and SAE Lens release pythia-70m-deduped-res-sm. B uses Gemma-3-270M and the public google/gemma-scope-2-270m-pt SAE assets. Resolve the **actual** model/SAE source repositories, immutable revisions, loaded weight state SHA256, tokenizer revisions and SAE IDs. A named SAE Lens release alias is not itself an upstream commit SHA.
- Verify that registry-selected hook, model dimension, SAE input dimension, normalization and data type agree with the native checkpoint. A failed pair is BLOCKED/FAIL with original error; **do not silently replace** the requested checkpoint, family, hook or representation.
- Use the **eight** fixed clean/corrupt pairs from the adjacent JSON: 8 independent groups x 2 input variants = 16 strings per model. Each of the four sites reuses the identical eight text groups but their **model-specific token IDs** differ. This is engineering-controlled text, not user data or benchmark ground truth.
- Tokenization gate before a CUDA forward: per-model records must include exact token IDs, BOS/chat-template handling, group and variant IDs, token count, decoded sample, revision and SHA256. Confirm equal token sequence length inside each pair for each model. If a pair mismatches, report it and stop that model's paired-patch route pending a revised reviewed manifest. Do not drop, retokenize differently, or silently replace a pair.
- Last **valid** prompt-token activation is selected; last valid next-token logits are the teacher-forced readout. No generated-token sampling is part of GPU-00. For native clean/corrupt patching, compare the same named hook, selected final token position and same full prefix semantics on both paths.

**Frozen sample selection:** eight groups, seed 17, batch size 1, sequence length at most 128 tokens, no hidden API calls. Freeze tokenization in a CPU-only manifest before any GPU measurements. If public checkpoint access is restricted, record BLOCKED_ACCESS and do not bypass it.

## 2. C0 preflight on the GPU host (mandatory before C1)

The agent may use an existing clean checkout or a separate new checkout/worktree. Never reset an owner's dirty working tree. Suggested checkout:

    git clone --branch feat/wp12-p0-foundation --single-branch https://github.com/OpenSAE/SAEBase.git SAEBase-wp12-gpu00
    cd SAEBase-wp12-gpu00
    git rev-parse HEAD
    git status --porcelain
    python3.11 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[dev,analysis]"
    python -m pip check
    saebase-check-env --mode analysis

Set the approved run root **outside Git-tracked source**, for example /workspace/wp12_gpu00_runs if it exists and has sufficient capacity. All logs and measurements must go there. Do not install every P1–P7 optional environment or download 48-site weights.

**Record before launch** in preflight.json and a log:
- source commit, dirty-tree state, UTC time; Python/pip/torch/TL/SAELens/transformers/HF Hub and CUDA versions, exact freeze/lock, upstream native library versions or Git SHAs;
- actual nvidia-smi and torch device properties, free GPU memory, resident processes, CPU/RAM, df -h for scratch/root, and the cache folder with remaining disk reserve;
- model/SAE/tokenizer source revisions, resolved named source IDs, rights/access status, real-dtype and weight-processing mode. Preflight must fail if the loaded model or SAE identity is ambiguous;
- manifest and prompt file SHA256; per-model token IDs and lengths; exact operations, invocation count prediction, max input tokens, expected output files and tests;
- implementation support matrix for model/site/backend and the exact numeric thresholds in §4.

**Compatibility trap:** the source uses legacy HookedTransformer while pyproject.toml currently accepts broad transformer-lens>=2.0 and sae-lens>=4.0. First record the resolver outcome. If the installed TL lacks HookedTransformer or loading semantics disagree, report this as a compatibility failure. A **separate pinned legacy worker** may be used to collect a clearly labeled LEGACY_ONLY reference without modifying or silently downgrading the primary environment; save its complete lock and source versions. Do not label the new TL4 backend PASS unless a separately implemented and independently matched TL4 bridge was actually exercised. Do not change default dependency constraints as an incidental GPU run.

**CPU tests** in the environment that will run CUDA, before downloading model weights:

    python -m pytest -q tests/test_m0.py tests/test_m1_store.py tests/test_m2_interventions.py tests/test_sae_model_registry.py tests/test_capture_with_logits.py tests/test_weight_identity_modes.py

If install, tests, artifact storage, tokenizer identity, license check or resource preflight fails: produce failure logs and stop before spending GPU. Missing package access and network failure are BLOCKED, not numerical FAIL.

## 3. C1 test sequence (real native operations and SAEBase; no mocks)

**Pilot A1 first.** Verify a real model and real SAE can load and that both sources agree. Existing Pythia M0 smoke is an extra baseline and should be attempted in the **same exact frozen environment** if workable:

    python saebase/tools/m0_reference_smoke.py --device cuda --seed 17 --n-generate 160 --min-kept 16 --n-prompts 8 --data-dir "$RUN_ROOT/legacy_m0_data" --output "$RUN_ROOT/legacy_m0_reference.json"

The built-in M0 script has **its own generated SimpleCopying inputs**, not the frozen eight clean/corrupt groups. Label it LEGACY_M0_SMOKE and never count its eight examples as the eight GPU-00 paired-patch groups. If the M0 script fails, preserve its traceback and decide whether the issue blocks A1. Do not reuse prior M0 output as evidence of a new run.

For all A1, A2, B1 and B2, use an isolated GPU-00 harness placed under the run directory, using the existing API and native upstream APIs without changing SAEBase implementation. Before GPU execution, save the harness source SHA256 and an operation-level spec. Compute actual tensors and full logits in **two distinct code paths**, not simply two calls to SAEBase:

1. **Native reference:** use the same verified checkpoint's original TransformerLens hook/cache/patch primitives and its original SAE Lens encoder/decoder. Record native clean and corrupt last-position logits, clean/corrupt selected hook activations, native sparse code, reconstruction, and native clean-to-corrupt patched logits. Use one fixed dtype/processing mode. Importing a precomputed figure/JSON does not count as a native run.
2. **SAEBase path:** use load_registered / ModelAdapter / capture_features / edit_features / patch_logits on the same exact checkpoint/model processing and frozen token IDs. Record its corresponding full-precision tensors and logits. Its weights, SAE site and tokenizer SHA must equal the native reference. If native and SAEBase model instances are loaded separately, fingerprint both **full** state dictionaries before comparing.
3. **Identity and negative controls:** native and OpenSAE forward(no hook) vs no-op identity hook on identical input; clean->clean patch, and incorrect hook or stale-cache identity must reject clearly. Compare both full logits, not only first 12 tokens or a norm.
4. **Clean/corrupt patch:** patch the last valid clean activation into the corrupt run at the exact declared site/position, compare the full target logit vector from both paths. Keep clean, corrupt, native-patched and SAEBase-patched arrays; record per-group effects, even if some edits produce no material change.
5. **SAE feature edit:** native SAE encoding at the declared clean selected activation vs SAEBase feature snapshot, full decoder reconstruction and original reconstruction residual. Choose the *highest-absolute nonzero* feature from the **clean** input for each group, set that code to zero using the declared residual-preserving edit, and patch the edited activation. Run an equivalent native SAE-code ablation with the same algebra, target and dtype. Log selected feature IDs, original code values, L0, reconstruction MSE, edited activation and edited logits. If no feature activates, do not fabricate an effect.
6. **Cache:** key captured activations by model state / tokenizer / site / token identities, and feature code by SAE state. Demonstrate same-key reuse (no fresh model forward) and an intentionally mismatched key rejected without returning stale results. Reuse existing SAEBase store and original tests rather than build another cache.
7. Repeat with the remaining site(s), **sequentially**, only after A1 parity is acceptable or the discrepancy is diagnosed. A2 and B2 are separate capabilities checks; a pass at L3/L5 does not certify L5/L9 by assumption.

Use existing source contracts and compare unrounded arrays. Distinguish teacher-forced single-token logits from generated trajectories. Do not claim TL4 parity, NNsight gradient support or pyvene support from legacy TL tests.

## 4. Frozen acceptance and stop thresholds

Thresholds apply only to fp32 execution with exactly matched checkpoint, tokenization, site and processing. **Do not relax them after seeing results.**

| Measure | Required rule | Record |
| --- | --- | --- |
| Model / SAE identity | Same immutable source revisions and same complete state hashes across reference and SAEBase paths | Exact SHAs, tokenizer identity, model-processing, shapes |
| Tokenization / positions | All eight pairs pass model-specific equal-length test; same IDs used by both paths; valid last position | Exact token IDs and BOS policy |
| Native vs SAEBase activation | elementwise allclose with atol 1e-4, rtol 1e-4, float32 | max abs/rel discrepancy and sample failures |
| Native vs SAEBase SAE codes + reconstruction | elementwise allclose with atol 1e-4, rtol 1e-4, float32; matching shapes and feature indices where not at a threshold boundary | L0, active indices, max errors, threshold ties |
| Baseline, no-op, clean/corrupt and edit logits | elementwise allclose with atol 1e-2, rtol 1e-4, float32; same full vocab and readout position | max abs/rel errors, per-pair pass flags |
| Genuine feature change | finite real edited activation and logits; selected clean feature was nonzero; record edit delta without requiring each text/logit sample to flip | per-sample raw deltas, at least one nonzero feature selected per site |
| Negative controls | no-op clean->clean accepted; wrong model/SAE/hook/tokenizer/cache identity rejected rather than silently rerouted | negative-control test output |
| Integrity and resources | complete manifests, raw arrays, checksums, resource log and error log; no over-budget step | runtimes, peak VRAM/RAM, disk, observed throughput |

For bf16/fp16, quantized, processed-vs-unprocessed or different backend execution, **no fp32 PASS is transferable**. Record a separate protocol and propose its tolerance before launching; do not silently claim equivalence because allclose happens under looser post hoc rules. The legacy M0 script independently uses atol=rtol=1e-4; its own declared check applies to its own output.

**Limits:** C1 only; no new generation; max 128 input tokens, microbatch 1; at most **320 total model forward equivalents** for the four-site WP12 prompt-pair matrix, excluding measured one-time checkpoint loading; stop after a total elapsed **7200 s cap** for this testing session or before exceeding **20 GiB new scratch records/cache**. Preserve at least **20% scratch free disk** and keep peak memory within preflighted actual available GPU headroom. If your actual machine cannot satisfy these caps, report a smaller pilot rather than overrunning them. No 48-site sweep, 500-prompt archive re-run, gradients/circuits, training, distributed run or automated C2 escalation.

**Immediately STOP that route** on failed source identity, token mismatch, missing rights, unapproved remote credential, wrong site, nonfinite arrays, no-op mismatch, native parity failure, CUDA OOM, or cap breach. Log observed values and stage. A temporary smaller microbatch is permitted only with a new preflight; skip no failed cases and do not overwrite failures.

## 5. Required immutable handoff artifacts

Under a run directory outside Git, e.g. runs/wp12/P0/gpu00-UTC-SHA/:

    manifest.json
    preflight.json
    environment.txt
    requirements-resolved.txt
    tokenization.json
    harness.py
    native_reference/           # raw arrays per model/site/group/operation
    opensae_results/            # matching unrounded raw arrays
    parity.json                 # predeclared test, tolerance, each PASS/FAIL/BLOCKED
    cache_identity.json
    cpu_preflight.log
    resource_usage.json
    failures.json               # [] if clean; no deletion of failed attempts
    checksums.sha256
    summary.md

Required manifest data: audited source commit, native upstream exact versions/commits, model/SAE/tokenizer revisions and state digests, hook/position/dtype/processing identity, frozen text input file hash, per-model token hashes, group IDs and unit of analysis, seed, operation, status and capability IDs F06–F09/F11/F14 and F41/F42 where applicable.

Do not include secret tokens, private corpora, third-party model weights or copyrighted checkpoints in PR attachments. Return a small rights-approved evidence archive or durable reference with checksum; preserve larger raw arrays outside Git, available for an authorized second reviewer. Screenshots are supplemental, not numeric evidence.

**A valid completion report must have** a four-row site PASS/FAIL/BLOCKED table, each failed-case denominator, native-vs-SAEBase checks, exact group and input counts, observed forward/work and wall time, peak VRAM/RAM/disk, source commits and hashes, known unsupported routes, and the next smallest repair test. Legacy-only PASS leaves TL4 pending. All results remain **GPU agent reported / pending independent review**. Only the local independent reviewer updates the master 50-capability acceptance ledger to verified statuses after inspecting raw output.

## 6. Deliverable decision tree

- **All four cases pass the frozen native parity and controls:** deliver source-bound evidence and request **independent P0 signoff**. Do not start GPU-01 automatically; remaining CPU/deployment/compatibility gates are separate.
- **Only legacy backend passes:** deliver legacy evidence, keep F06/new TL4 bridge marked PARTIAL and propose the minimal TL4 bridge test (WP12-01); do not silently change dependencies.
- **One or more model sites blocked by weight access/loader:** mark each exact combination BLOCKED, identify access and the smallest scientifically valid alternative for human review. No silently substituted model.
- **Numerical mismatch:** stop expansion, preserve both raw tensors and difference at first diverging operation, diagnose dtype/token/hook/processing and propose one restricted repair. A failed case is useful evidence, not grounds to skip it.
- **Resources insufficient:** complete and return preflight + partial C1 output, with no C2/C3 escalation.

Acceptance authority: the owner/independent CPU reviewer. This handoff updates **planned work and test inputs only**; no GPU numerical result has been produced by this document.
