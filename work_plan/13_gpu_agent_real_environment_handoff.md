# WP13 — Local GPU Agent Real-Environment Handoff

Date: 2026-10-22
Read first: [WP13 operating plan](13_real_environment_integration_and_deployment.md).
Machine-readable queue: [13_live_execution_matrix.csv](13_live_execution_matrix.csv).

**2026-10-22 actual GPU Agent progress:** LIVE-00/01/02 and the **teacher-forced portion of LIVE-03** report PASS_WORKING on an RTX 5070. LIVE-02 computed 20 complete Gemma L5/L9 traces and locally indexed 5,804 features, with **zero approved excerpts**. The small source files from PR #55 and #56 are now merged into WP13, with all raw Trace/SQLite/logits retained privately. **Do not rerun these GPU tasks.** Follow the [LIVE-03 measured-result integration and next-step plan](13_live03_review_and_gallery.md). Public gallery release and fresh-prompt private queue execution remain separate tasks. Existing raw traces stay on the GPU host.
Inherited implementation: [WP12 core assessment](12_core_function_assessment.md) and [PR #53](https://github.com/OpenSAE/SAEBase/pull/53).

**One instruction:** execute the specified original model/backend operation and make the **result actually usable in OpenSAE**. The objective is not a second GPU-00 audit, a 48-site benchmark, or 50 standalone folders stamped “PASS”. Start with LIVE-00 and the first functioning Prompt→SAE Feature→Intervention user journey. Do not run all rows blindly.

## 0. Working rules for the local agent

- Use the exact GitHub `docs/wp13-real-world-integration` branch, currently based on WP12 code head `1054cf6d449b29b91b39578bbbd495b99141f1e9` plus WP13 documents. Record actual checkout SHA before every computation because it can change when the Integration Agent fixes a bug.
- If GPU-00 assets already exist from `gpu00-20261022T031830Z-490b18c`, reuse their known-good model/SAE environment and cached weights, not its 576 output tensors as new runtime evidence. The agent previously reported 4/4 legacy TL3.9 + SAELens6.54 fp32 PASS; no default rerun.
- Use a **separate environment per native library**. Preserve the tested legacy venv. TL4, NNsight, pyvene, SAEBench, SAELens training, Sparsify and circuit-tracer may have mutually incompatible dependency pins. Do not install every extra into one environment.
- No model checkpoint, third-party corpus text or token is committed to public GitHub, HF dataset or Space without explicit provenance/rights review. Don't disable SSL or use unauthorized mirrors to make downloads work.
- One GPU job at a time; check the **actual** device, VRAM and free disk. Previously mentioned 32 GiB RTX 5090 is a planning hint, not a guarantee of available hardware. Do not run a two-GPU simulation and call it F50.
- If a model is too large, use an independently supported *smaller authentic checkpoint* documented by the Integration Agent. Do not silently substitute another checkpoint inside the original target case or pretend the source identity matches.
- Keep a failed run's first traceback and applied patch diff. We need **one actionable failure**, not a 150 MB ZIP every time.
- GPU Agent **does not publish** the production HF Space or open unauthenticated GPU ports. Its handoff ends in private results, a small reviewed public-bundle candidate and a status summary.

## 1. Environment setup without disturbing previous work

Source is not on `main` yet; do NOT do `git checkout main` and assume it contains the new WP12 functions.

```bash
# Existing clone; first inspect before creating a separate worktree.
git fetch origin
git status --porcelain
git rev-parse HEAD
git branch --show-current

# Never reset an existing dirty tree.
git worktree add /workspace/SAEBase-wp13 origin/docs/wp13-real-world-integration
cd /workspace/SAEBase-wp13
git rev-parse HEAD
git status --porcelain

mkdir -p /workspace/wp13/private/runs
mkdir -p /workspace/wp13/private/snapshots
mkdir -p /workspace/wp13/candidates
mkdir -p /workspace/wp13/public-bundle

python3.12 --version
nvidia-smi
df -h /workspace
free -h
```

Use an existing Python 3.12 interpreter if available (the old GPU-00 host reported 3.12.3); otherwise select a supported installed 3.11/3.12 version. Do not modify the system Python or uninstall the previously working environment.

**Legacy isolated environment** (example for a new venv, only if a tested existing one is unavailable):

```bash
python3.12 -m venv /workspace/wp13/venvs/legacy
source /workspace/wp13/venvs/legacy/bin/activate
python -m pip install -e ".[dev,analysis]"
python -m pip check
python -m pytest -q tests/test_wp12_core_status.py tests/test_wp12_scope2_native.py \
  tests/test_wp12_clt_intervene.py tests/test_wp12_nnsight_generate.py
python scripts/wp12_p8_acceptance.py \
  --ledger work_plan/12_capability_acceptance_matrix.csv \
  --output /workspace/wp13/private/source-report
python -m pip freeze > /workspace/wp13/private/requirements-legacy.txt
```

This installation uses the project dependency resolver; **it is not automatically equivalent** to the reported frozen TL3.9/SAELens6.54 environment. Record actual package versions, and if incompatible, restore the originally resolved GPU-00 versions **in the isolated legacy worker only**. Do not make the P8 CLI's 50 pending signoffs a blocker for real product smoke.

Separate native venvs suggested under `/workspace/wp13/venvs/{tl4,nnsight,pyvene,saebench,training,circuits}`. Reuse a compatible environment when its exact resolved dependencies agree; no need to maintain seven redundant identical copies.

### LIVE-00 preflight facts

Write a compact `preflight.json` listing: `code_sha`, Python+torch+CUDA+GPU+driver, resolved library versions, one actual source identity per selected model/SAE, cached weight locations, rights/access, scratch/root free space, current resident GPU processes, and whether launch scripts import. Inspect `lsof` or NVML processes if VRAM is already in use. Check `OPENSAE_LAB_DATA` path and Docker availability for the CPU branch.

**PASS_WORKING** for LIVE-00 only means exact source/host and a runnable small job route have been established. Do not declare any numerical new backend correctness from a hardware inventory.

## 2. Queue and exact execution instructions

Tasks after LIVE-01 are selectively triggered by the Integration Agent. Do not run LIVE-05 training/circuits until the primary 10-prompt user journey is actually working.

### LIVE-01 — One real registered model → SAE feature → edit

Owner: GPU Agent.
Prerequisite: LIVE-00 environment with original checkpoint and tokenizer source resolved.
Target: Pythia-70M `pythia70m_resid_sm` L3, or Gemma-3-270M `gemma270m_scope_resid` L5; both have historical GPU-00 reported support. Use one model first.

```bash
python scripts/wp12_p1_smoke.py \
  --sae-key pythia70m_resid_sm --layer 3 --device cuda \
  --output /workspace/wp13/private/runs/LIVE-01-small-smoke.json
```

This is a real 3-prompt checkpoint test and returns feature identities, shapes, reconstruction and one edit logit delta. It is not a generated-text steering test. If it fails at the first real import, save a short traceback and environment lock. **Fix the adapter**, don't spend effort revalidating all four GPU-00 sites.

Success: real checkpoint loaded at exact hook, nonzero feature selected, no-op unchanged, edited execution completed with finite numbers. Keep model and SAE digests, dtype, peak VRAM and actual runtime.

### LIVE-02 — Make 10 authored prompts into real browse-ready records

Owner: GPU Agent produces data; Integration Agent approves/publishes.
Prerequisite: LIVE-01 valid source identity for **two selected compatible sites** or a documented one-site fallback.

First run the documented one-prompt capture on actual checkpoint:

```bash
python scripts/wp12_p2_capture.py \
  --sae-key gemma270m_scope_resid --layer 5 --device cuda \
  --prompt-file /workspace/wp13/private/prompts/demo-01.txt \
  --sample-id wp13-demo-01-L5 --group-id wp13-demo-01 \
  --split discovery \
  --output /workspace/wp13/private/runs/LIVE-02-demo-01-L5.json
```

Use the [10 original candidate prompt groups](13_demo_prompt_candidates.json) as the default fixed source text after a human reviews public-text rights; preserve its discovery/evaluation split and exact bytes. A selected feature is not guaranteed to have positive/negative held-out examples, so do not invent activation labels. Then add a small batch controller that loads a model/SAE/site once and loops through up to 10 approved short prompts. The existing CLI currently loads weights for every invocation, so repeating it 20 times is *not* the desired final implementation. Use `saebase.explore.capture_prompt_trace(pair=loaded_pair, ...)` and preserve exact prompt bytes and tokenizer SHA.

Target: first 1 good trace and feature page, then 10 approved short texts × L5/L9 = **up to 20 traces**, within existing 32-trace limit. 6 distinct prompt groups for discovery, 4 for held-out or another predeclared disjoint arrangement. Content semantics must be observed, not invented. No random synthetic sparse activations in a public bundle.

Each trace: exact model/SAE full state SHA256, tokenizer fingerprint, site ID, original prompt and SHA256, sample/group ID, split, complete nonzero sparse features per token, token text and token ID, legal review status. Texts **remain private** until human-approved for release.

Handoff: private raw traces, one compact preview JSON, a limited rights-review manifest and one screenshot of actual feature values. Integration Agent owns converting to `backend.feature_store` SQLite and `backend.explore.publish_exploration` checked public bundle. If imported source hashes don't match registry, don't forge or alter them.

### LIVE-03 — True baseline/edited outputs and a local fresh-prompt task

Owner: GPU Agent (compute) + Integration Agent (UI/public curated display).
Prerequisite: LIVE-02 real feature IDs.

Required operations on 2–3 actually active features from one selected prompt:
- source run, identity/no-op, actual code ablation, one scale or add;
- teacher-forced full-vocab logit delta, actual sample/token/feature ID;
- optional native generated-continuation edit via `saebase/interventions/steering.py` and/or `NNsightLocal.generate_edit`. **Do not call logits-only execution generated steering.**

For one private fresh prompt, use the existing service flow, after starting the **authorized local backend** on loopback. The following are locally operator-scoped examples, not public service enablement:

```bash
# Secret values must come from local secret manager, not these documents.
# Set distinct 32+ character OPENSAE_PRIVATE_OPERATOR_TOKEN /
# OPENSAE_PRIVATE_WORKER_TOKEN, enable private jobs and use owner-only data root.
export OPENSAE_ENABLE_PRIVATE_JOBS=1
export OPENSAE_PRIVATE_DATA=/workspace/wp13/private/service
mkdir -p "$OPENSAE_PRIVATE_DATA"

# In an isolated private terminal:
uvicorn backend.api:app --app-dir apps/research-lab \
  --host 127.0.0.1 --port 8000

# Use the documented /api/private/inputs -> /jobs flow from an authenticated
# operator client, with operation "inspect" (feature_ids=[] and strength=0).
# After submission, execute one queued operation:
python scripts/wp12_private_worker.py --device cuda

# Operator retrieves /api/private/jobs/<actual job id>/result with its token.
```

If the current private worker reloads weights for each job and browsing becomes slow, record measured model-load vs encode latency. **Integration Agent** develops a persistent model cache/supervisor; GPU Agent verifies it with two sequential requests and a restart. Don't publish secrets or full private prompt text in logs.

Success: a **previously unseen** owner-supplied input runs a real model and returns a token-feature record with actual source IDs; authorized owner can retrieve it; wrong bearer token cannot. A public CPU visitor must still get a recorded-only message.

### LIVE-04 — Local explanation model (small, real, reviewed)

Owner: GPU Agent.
Prerequisite: LIVE-02 authentic discovery/held-out activation data and local instruct checkpoint that fits.

```bash
python scripts/wp12_autointerpret.py \
  --local-model /workspace/wp13/private/snapshots/local-instruct \
  --examples /workspace/wp13/private/feature-interpretation-example.json \
  --output /workspace/wp13/private/runs/LIVE-04-candidate.json \
  --device cuda
```

The input JSON needs `site_id`, `feature_id`, `discovery`, `heldout`. Each example needs a distinct `group_id`, actual `activation` and `text`, with both activating and non-activating held-out groups. The CLI uses original local language-model generation plus a separate binary held-out prediction pass. Scores are activation-prediction accuracy, **not** proof of semantic interpretation. Result remains `approved=false`; review candidate text, bad examples, source and rights before UI publication.

If no compatible local instruct checkpoint is available, do not fetch large models or incur paid API usage without permission. The Feature card can still truthfully say “no approved explanation”.

### LIVE-05 — Original SAEBench and a matched two-SAE comparison

Owner: GPU Agent.
Prerequisite: source-pinned original evaluation environment, real rights-approved dataset and exactly compatible model/SAE.

Read `saebase/wp12/benchmarks.py` and `custom_saebench.py`. One valid invocation model, in a Python environment with the **original pinned upstream source**:

```python
from pathlib import Path
from saebase.wp12.benchmarks import NativePilot, native_pilot

job = NativePilot(
    repo_root=Path("/workspace/SAEBase-wp13"),
    upstream=Path("/workspace/SAEBase-wp13/upstream"),
    output=Path("/workspace/wp13/private/runs/LIVE-05-native"),
)
print(native_pilot(job))
```

**Do not run this exact snippet unless its expected `repo_root/upstream` original SAEBench checkout and pinned revision are actually present**. It invokes the existing small Core + Sparse Probing upstream script, not the full benchmark. Inspect actual native dataset/model mapping, API dependencies and results; propose a focused patch if upstream types differ.

For a native custom checkpoint use `run_custom(...)` rather than inventing its CLI public semantics. Before plotting two SAEs, require same exact model weight SHA, hook, dataset/splits, native commit and metric. If no second real same-site checkpoint exists, leave the comparison empty, or complete LIVE-06 first to create one.

Success: one real native evaluator output has original source/metric IDs, can be opened in OpenSAE Benchmark UI and is visibly distinct from a recorded JSON import. Two-SAE comparison only on genuinely comparable sources.

### LIVE-06 — Tiny native train, checkpoint load and real resume

Owner: GPU Agent.
Prerequisite: compatible local model and small licensed dataset snapshots, native trainer version pinned. No F50 distributed request.

Code entry: `saebase.wp12.training.TrainPlan`, `run_training`, `register_checkpoint`, `resume_sparsify_training`.

Experiment sequence:
1. Select smallest genuinely compatible model and one named hook. Check actual `d_in`, SAE `d_sae` and `k`; record official model and dataset revision SHA. Avoid choosing a 7B model simply because it is available.
2. Native SAELens: instantiate a small `TrainPlan(backend="saelens", ..., architecture="topk")`, approve a small budget, run original trainer. Save original checkpoint with source IDs.
3. Native Sparsify: separate venv, original `TrainPlan(backend="sparsify", ..., architecture="topk")` with `max_examples <= steps * batch_size`. Use original checkpoint and record actual trained steps. Training on-the-fly is performed by upstream native hookpoints.
4. Convert or register resulting weight/metadata source in SAEBase and **actually reload/encode/decode**. A checkpoint file alone does not mean OpenSAE supports its new architecture.
5. Test native `--resume` only after an actual `checkpoints/{run_name}/state.pt` exists and the original plan matches. Do not substitute `--finetune`. One interrupted/restarted short continuation is sufficient for product readiness.
6. Optional native `--transcode` short run if the upstream model supports actual different input/output hooks. A transcoder must not be presented as same-site SAE.

Stop if source representation incompatible, checkpoint reload fails, dataset license unclear, OOM or approved compute budget reached. Fix actual adapter, not training objective/metric definitions.

Success: a *fresh* native checkpoint loads through OpenSAE and returns finite real codes/reconstruction; at least one real resume attempt completed and recorded (or a concrete native blocker is reported).

### LIVE-07 — Actual native circuit and cross-layer feature edit

Owner: GPU Agent; UI repair and asset approval by Integration Agent.
Prerequisite: original compatible circuit-tracer checkout, source model/transcoder, bounded prompt and source rights.

Native execution first, with pinned upstream CLI (as in `run_native_circuit_tracer`), one short prompt and compatible original `--transcoder_set`. Record actual output graph and one source SHA. Do not import a tensor pickle in the public web process.

Import the native v1 frontend `{metadata,nodes,links}` JSON via `import_native_graph_file`. Confirm a real feature node, error node and signed target edge. Run a **supported** actual intervention:

- Single-layer transcoder: `saebase.wp12.circuit_intervene.execute_transcoder_node` using matched actual model/representation SHA.
- Multi-layer CLT: `saebase.wp12.clt_intervene.execute_clt_feature` with actual official source-layer shards and downstream hook mapping; the identity must match the trace/model exactly.

Then open the published graph in Research Lab and demonstrate selection, filtered signed edges, pin/group/note/export, and a link to the measured before/after result. A graph generated from another model family is not an eligible intervention target.

If circuit-tracer cannot run on the available model/VRAM, label `BLOCKED_ENV` for **native attribution**, but still test the browser converter with an official permitted real graph file. Do not call an upstream fixture newly computed attribution.

### LIVE-08 — Private worker reliability and real browser round trip

Owner: both Agents, Integration Agent leads.

Run the same owner job path twice to measure load vs compute latency. Check source-identical reuse and restart after a deliberately interrupted worker. Test 401 with missing/wrong role tokens, 503 when jobs disabled, cancel/timeout/error and original source mismatch. Never print tokens or full private text to stdout or screenshots. Submit results from real GPU worker, then follow the exact owner-only API to retrieve their SHA256-matched results.

Deliver `service-smoke.json` with `worker_started`, `backend_version`, `job_id`, `status`, `output_type`, `model_load_seconds`, `operation_seconds`, `retry_outcome` and `no_public_gpu_access=true`. The Integration Agent can then prioritize persistent model reuse and UI feedback.

### LIVE-09 — Public CPU Docker staging and remote browser

Owner: Integration Agent. GPU Agent only supplies approved compact data.

Build from repo root, using exact source checkout and frozen bundle:

```bash
docker build -f apps/research-lab/Dockerfile -t opensae-lab:wp13 .
docker run --rm -p 7860:7860 \
  -v /workspace/wp13/public-bundle:/workspace/runs/research-lab:ro \
  opensae-lab:wp13

# In a separate terminal or safe HTTP client:
curl -fsS http://127.0.0.1:7860/api/health
curl -fsS http://127.0.0.1:7860/api/summary
curl -fsS http://127.0.0.1:7860/api/explore/traces
curl -fsS http://127.0.0.1:7860/api/wp12/circuits
```

If `/workspace/wp13/public-bundle` doesn't contain the exact `features/`, `explore/` and `published_wp12/` directories expected by the backend, do not treat 200 OK from the empty UI as success. Staging must show actual approved records and interactive navigation; test restart with same data bundle.

HF deployment uses owner-approved staging Space first. Save existing production Space revision before any proposed change. Operator/worker secrets stay out of the public CPU runtime.

### LIVE-10 — Product operation smoke and operator feedback

Owner: Integration Agent + at least three external researchers.
GPU Agent may help reproduce one local bug, no broad rerun.

Three tasks: real prompt→feature, recorded vs live intervention distinction, native benchmark/circuit inspection/export. Record actual completion/blockers and one requested UI improvement each. Check understandable error labels and browser accessibility with keyboard navigation. Source the screenshot from the running staging Space, not a mocked component.

This stage produces a usable preview release candidate, not a P8 science certification.

## 3. Per-job hard budgets and pragmatic scope

The [CSV queue](13_live_execution_matrix.csv) lists initial limits; adjust only if the hardware preflight and actual source constraints require it. These are **work-control defaults**, not promised runtime estimates.

| Class | Inputs | Default new computation / wall cap | Notes |
| --- | --- | --- | --- |
| Small model/SAE smoke | 3–10 prompt groups, max 128 tokens | ≤30 minutes / ≤10 GiB new scratch | Reuse known model cache; abort on model identity mismatch |
| Interactive prompt capture | 10 approved prompts × up to two sites | ≤60 minutes / ≤15 GiB | Load each model/site once, output complete sparse token rows |
| Controlled steering and explanations | 2–3 features × 1–2 prompts, ≤16 generated tokens when supported | ≤60 minutes / ≤15 GiB | Distinguish source-only logits and text generation |
| Native benchmark pilot | One Core + one Sparse Probing small run | ≤90 minutes / ≤20 GiB | No default full dataset sweep |
| Native training/restart | Tiny genuine checkpoint, bounded max_examples/steps | ≤90 minutes / ≤30 GiB | One GPU, no F50 multi-GPU claim |
| Native circuit | One short compatible prompt/target | ≤90 minutes / ≤20 GiB | Stop if original source cannot execute |
| Private worker | Two tiny owner jobs plus interruption | ≤30 minutes / ≤5 GiB | Keep durable result and protect source text |

Do not run hours of extra experiments simply to fill a matrix. When a job fails, report the smallest actual blocking operation and hand the Integration Agent a tiny reproducer.

## 4. One small handoff per wave, no repeated huge evidence ZIP

Record under `/workspace/wp13/private/runs/<LIVE-ID>-<utc>-<source-shortsha>/`:

```text
request.json             # fixed model/site/operation and owner scope
preflight.json           # actual hardware, lock, source identity
status.json              # PASS_WORKING/PARTIAL_RUNTIME/BLOCKED_ENV/FAIL_FUNCTIONAL
summary.md               # exact user operation + result or first blocker
results_preview.json     # limited nonsecret, real values; no rights-uncleared text
resource_usage.json      # observed wall/VRAM/RAM/disk, no guesses
failures.json            # first failure and fixes, [] if none
source_versions.txt      # pip freeze / original tool refs
private_raw/             # private unrounded computations, checkpoints
publishable_candidate/   # only human-approved and rights-reviewed assets
```

For routine integration handoff, provide GitHub source SHA, `status.json`, `summary.md`, one small `results_preview.json` and any focused fix branch. Large raw assets can stay at their stable private path until genuinely needed. An archive is needed only when moving an approved public release or when an actual numeric bug cannot be diagnosed from a focused small sample.

Minimal final status vocabulary:
- `PASS_WORKING`: a user-relevant real operation ran and its output reached intended consumer
- `PARTIAL_RUNTIME`: real operation partially ran but UI/backend integration is broken
- `BLOCKED_ENV`: authenticated sources, rights, checkpoint availability, NDIF or real GPU hardware absent
- `FAIL_FUNCTIONAL`: executable attempted but computation/output materially wrong; create an actionable fix

Never mark `PASS_WORKING` for an import-only test, simulated output, a backend wrapper with no actual invocation, or a page with no real content.

## 5. Agent-to-Agent repair protocol

1. GPU Agent reports `LIVE-ID`, source SHA, exact method/source and the first failing operation; attach one minimal reproducible command and traceback.
2. Integration Agent reproduces on CPU when possible; patches only the adapter, input schema, UI or version lock, preserving underlying original estimator and historical data.
3. The fix goes to a short named branch `fix/wp13-LIVE-ID-<cause>` and a PR against the active WP13 integration branch. No force-push to `main`.
4. GPU Agent fetches the **specific accepted fix commit** and reruns only the failing real operation and directly affected follow-up. It does not repeat earlier successful 10 prompts or GPU-00.
5. Integration Agent confirms the output can be consumed by the Lab and updates the operational queue. A new task begins only if it provides the next useful user-visible capability.
6. Document any **rights/credential/multi-GPU** blocker without requiring the GPU Agent to work around it.

One GPU Agent session can progress through several already approved small LIVE jobs as source compatibility permits; handoffs should be compact and results-oriented, rather than repeating a large independent certification after each small step.
