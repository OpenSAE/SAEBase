# Work Plan 13 — Real-World Integration and Deployment

Date: 2026-10-22
Repository: OpenSAE/SAEBase
Source baseline: `feat/wp12-p3-p8-platform` at `1054cf6d449b29b91b39578bbbd495b99141f1e9`, CI [39820276523](https://github.com/OpenSAE/SAEBase/actions/runs/39820276523) successful.
Plan branch: `docs/wp13-real-world-integration`, stacked on [WP12 PR #53](https://github.com/OpenSAE/SAEBase/pull/53).
GPU execution companion: [13_gpu_agent_real_environment_handoff.md](13_gpu_agent_real_environment_handoff.md).
Job inventory: [13_live_execution_matrix.csv](13_live_execution_matrix.csv).

**Mission:** Take the 48 source-level core-PASS implementations and the two external blockers from Work Plan 12 into **actual working research workflows**, local model execution, a useful publicly browsable Research Lab, and repeatable, secure deployments. This work plan deliberately **does not** initiate 50-feature formal P8 certification, reconstruct a 576-tensor audit, or re-run the 48-site historical sweep.

**Status now:** planning and interface preparation only. No new GPU job, Hugging Face deployment, public data release or native model run is claimed by creating these documents.

## 1. What changes from WP12

WP12 proves that executable routes exist in the codebase. WP13 asks different, user-visible questions.

| Question | What counts as working |
| --- | --- |
| Can a researcher install it? | Clean supported Python environment launches the selected backend, with pinned dependencies and a meaningful failure when unsupported |
| Can a researcher run a model? | Actual model and pretrained SAE load, use matched tokenizer/layer/weight-processing identity, capture real activations and respond to a real prompt |
| Can a researcher explore it? | A real prompt → actual per-token features → detail page → evidence-linked examples; labels have a stated origin and no invented “interpretation” |
| Can a researcher intervene? | Source/original and edited executions both actually run; visible measured logit or generation changes, plus no-op and nonfireable-hook safeguards |
| Can a researcher evaluate or train? | Native SAEBench/SAELens/Sparsify tools actually run against compatible sources; new checkpoint or comparison is loaded back into OpenSAE |
| Can a researcher trace a circuit? | Native circuit-tracer produces a graph; OpenSAE opens its real output; clicking a node can trigger a compatible actual intervention |
| Can someone else visit the Lab? | A deployed browser app contains reviewed, meaningful real examples and loads/restarts without depending on the author's GPU being online |
| Can an authorized researcher supply a fresh prompt? | The **private** worker runs a real model and returns the job/result for its owner. Public users see accurately labeled recorded examples unless a separate authorized GPU service exists |
| Does it recover? | A failed task can be retried or diagnosed; a deploy can be rolled back; a model not available does not become an empty fake PASS card |

**No false equivalences.** Recorded results ≠ live inference; teacher-forced logits ≠ generated text; attribute graph ≠ causal proof; native benchmark imported JSON ≠ native evaluation; source implementation PASS ≠ deployed operation. Keep native SAR/RSS/Input Localization estimators and historical Stage D / 500-prompt records unchanged.

**WP12 source baseline:** [Core verdicts](12_core_function_assessment.md) record 48 core PASS, 0 core PARTIAL and 2 core BLOCKED (F20 NDIF, F50 real multi-GPU). All independent P8 signoffs remain PENDING. A real environment failure is a *deployment/runtime bug to fix*, not an excuse to relabel a truly missing operation PASS.

## 2. Intended operating architecture

### 2.1 Public tier: source-backed CPU Research Lab

- Docker/FastAPI/JS runs on CPU without torch, model weights, remote NDIF credentials or the private queue being enabled.
- Pages: Results, Features, Intervene (preflight/recorded evidence), Analyze SAR/RSS/Localization, Datasets, Reports, Benchmarks, Training, Circuits and Jobs/availability.
- A first release must contain **real, operator-approved** source-bound feature cards and at least one navigable Prompt→Feature journey. Pages with no published data must say so instead of showing generated pretend results.
- Public `POST /api/jobs` intentionally returns 503; owner-only `/api/private/*` is not reachable from public browsers. “View recorded output” and “Compute in local GPU” remain separate controls.
- Browser and API should show build SHA, data bundle digest, scope of recorded/live features, model and SAE IDs and updated time, as appropriate. Do not reveal private prompt text or access tokens.

### 2.2 Private computation tier: local single-GPU agent/worker

- Native model checkpoints, SAE/transcoder/CLT parameters, training archives and arbitrary input text remain in private storage.
- Worker supports one bounded real request at a time initially: `inspect` for all-token SAE codes, `ablate/scale/clamp/add/identity` for controlled teacher-forced edits. Expand toward native generation/graph/training as separately scoped jobs.
- API access requires distinct operator/worker bearer tokens and durable private data; listen on `127.0.0.1` or an explicitly authenticated private network. Do **not** open a public unauthenticated GPU port.
- One-shot `scripts/wp12_private_worker.py --device cuda` exists. A persistent queue consumer with model reuse, status, timeout, heartbeat, deduplication and graceful shutdown is a **WP13 development task**, not already a deployed service.
- The GPU Agent's historical single-card capacity is a planning assumption; actual available CUDA device, VRAM and scratch must be rechecked before running. A single GPU cannot satisfy F50.

### 2.3 Data, storage and release tier

Three separate roots, never conflated:

```text
/workspace/wp13/
  private/
    snapshots/             # pinned model, SAE, tokenizer and native upstream checkouts
    inputs/                # consented original texts, protected
    runs/LIVE-*/           # source logs, raw tensors, checkpoints, traces
    queue/                 # encrypted/protected or owner-only worker storage
  candidates/
    approved-prompts.json # only original, explicitly reviewed sharable examples
    feature-index/         # SQLite, exact source identities
    explore/               # manifest.json + records.json
    benchmarks/            # native metric summaries
    circuits/              # converted real graph previews
  public-bundle/
    manifest.json          # immutable release digest, git/weights/data hashes, license
    features/
    explore/
    published_wp12/
    screenshots/
```

- The CPU Docker image is presently built from the **repo root**, copies the app and historical `conf/coverage/stage_d_evidence` and sets `OPENSAE_LAB_DATA` to `/workspace/runs/research-lab`.
- New public records must be **mounted or copied into that actual runtime data root**, not merely present in the developer's local Git checkout. Test this explicitly, including process UID 1000 and restart behavior.
- Avoid raw model weights, unauthorized third-party data, secret files, raw 500-prompt archives and copyrighted contexts in public bundles.
- The public deployment should use a separate **staging Space or isolated staging deployment** first. Keep the existing `OpenSAE/SAEBase` Space untouched until its current revision and successful rollback path are established.
- Hugging Face Docker Space disk may be ephemeral across restarts; production browse assets must come from the image or an explicitly mounted read-only versioned volume/bucket. Buckets are a viable storage route but are **not** automatically available or permissioned. Official references: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [Storage](https://huggingface.co/docs/hub/spaces-storage), [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets).

### 2.4 Architecture decision for launch

**Launch 1:** public *recorded-real* experience + private *local-live* capability. No live GPU endpoint for general web visitors. This gives visitors a reliable working product even if the single GPU is shut down.

**Launch 2 (optional):** authenticated on-demand remote GPU service only after runtime permissions, networking/TLS, worker isolation, job quotas, live queue/load behavior, cost ceilings and deletion controls are implemented. It must not become a hidden dependency of Launch 1.

## 3. Outcomes the work must produce

### User journey A — Real feature exploration (highest priority)

1. Select a supported official model and exact SAE/layer from visible registry.
2. Select one of at least **10 useful, human-reviewed original short prompts**. At least two verified sites from one compatible small model is preferable for the first public bundle; use the Gemma-3-270M L5/L9 sites if their registry and loaded weight digests really match.
3. See the original tokens, chosen token and **measured** activation/feature ranking; every number comes from the model and SAE.
4. Open feature detail, inspected activation examples, hypothesis/source (or “not interpreted”), verified site/hashes, adjacent feature links.
5. Follow links from prompt ↔ token ↔ feature ↔ back to prompt; export standalone HTML. A user must not see “empty” because the 20 actual example traces were never imported.
6. Show at least one honest group-level correlation view if ≥3 independent complete-code groups meet the source contract. Do not turn a top-k-only partial record into “all absent codes = zero”.
7. Arbitrary unseen prompt in the **public CPU browser** says “requires local/private compute”, rather than faking inference; private owner can run it via authorized worker and inspect a real token result.

For 10 prompts × two sites, publish up to 20 reviewed traces, respecting the present `backend/explore.py` limits: **32 traces total, 128 tokens each, 256 nonzero features per token and 8 MiB release**. If any limit fails, repair the intended format or reduce *approved* scope explicitly; do not silently truncate sparse nonzeros and mark complete.

### User journey B — Real feature intervention

1. Start from a measured active feature, not a random index.
2. Execute matching same-prompt baseline + no-op + ablation + at least one scale/add experiment.
3. Show the actual edited token logits/delta and (on supported generation backends) baseline vs edited decoded continuation. Label **teacher-forced logit-only** and **generation-time** separately.
4. Record selected feature, token position, model/SAE/hook digests and actual effect. A genuine near-zero effect is allowed and accurately shown.
5. Invalid hook, stale checkpoint, invalid feature and unavailable generation backend produce understandable rejection. Do not silently show a saved old response as current.

### User journey C — Native research workflow

1. Run one original native **SAEBench Core and Sparse Probing** bounded job, with actual datasets/weights, rather than just parsing JSON.
2. Produce a fair two-checkpoint comparison only if both checkpoints share model, hook, dataset/split, native protocol and metric; otherwise explicitly refuse comparison.
3. Run tiny native **SAELens** and **Sparsify** training on genuine local snapshots, save a checkpoint and load an actual new SAE back into an Explore or encode/decode task.
4. Show genuine **resume** vs **finetune** difference. Resumed Sparsify must use original trainer state in `checkpoints/{run_name}`. No two-GPU certification on one device.
5. Run one genuine original **circuit-tracer** attribution on a compatible model/transcoder set; import its real frontend JSON, navigate an error node and a signed edge, select a feature and perform a compatible actual model edit. If original upstream graph generation is incompatible with the available single GPU/asset, show the real blocker and still make the graph importer usable with a separately sourced authorized native artifact.
6. Load at least one official pretrained Scope 2 transcoder; test CLT on a genuinely compatible smaller official release only if the actual shard/config/site/VRAM conditions permit.

### User journey D — Deployed, trustworthy Lab

1. Start a fresh local CPU Docker container from a frozen source SHA, find the bundled real feature/prompt/benchmark/circuit records, and follow all links.
2. Restart and see identical records without writing to ephemeral state.
3. Browser: successful prompt selection and token clicks, feature-to-intervention preflight, graph pan/filter/pin/export; error and empty states remain comprehensible.
4. Staging HF Space: build from the correct repo root and asset bundle, capture real build/runtime revision and smoke `/api/health`, `/api/summary`, `/api/features`, `/api/explore/traces`, `/api/wp12/circuits`, and `/api/wp12-compute`.
5. Promote the tested image/data artifact to the organization Space only with explicit owner access/approval and a recorded previous working revision to roll back to. Do not infer “Space deployed” from a GitHub commit.
6. Ask three researchers to complete a real journey without developer intervention, record where they get stuck, repair blockers, and label the available capabilities plainly. This is **product usability testing**, not a formal scientific study.

## 4. Work order and owners

Work is grouped by a **usable vertical slice** rather than by arbitrary feature IDs. Jobs and actual handoff payloads appear in [the GPU companion](13_gpu_agent_real_environment_handoff.md) and [CSV queue](13_live_execution_matrix.csv).

| Wave | Primary owner | Work and dependency | Small deliverable | Exit condition |
| --- | --- | --- | --- | --- |
| R0 Source/rollout baseline | Integration Agent | Freeze WP12 stacked PR order and deployment baseline; no existing assets overwritten | Canonical working SHA, checkout/install matrix, current Space revision if accessible, staging strategy | One source SHA and exact paths for every operator; clean CPU build |
| R1 Local backend reality | GPU Agent + Integration Agent | Pythia/Gemma real loads, TL4/legacy isolated, Scope2 native, pyvene & NNsight short probes | Model/site support table with real outputs and first failure | At least one complete, real load→tokens→codes→edit path |
| R2 Golden feature corpus | GPU Agent produces; Integration Agent curates/publishes | 10 reviewed prompts × 1–2 genuine sites; index+traces+source labels | A compact reviewed public data bundle and interactive Feature/Prompt paths | Real feature URL and token TopK navigable end-to-end |
| R3 Steering and authoring | GPU Agent computes; Integration Agent makes interface intelligible | Baseline/no-op/ablate/scale/generate where supported, local auto-interpretation | Real before/after gallery with provenance, actual editable local job | Measured changes and clear unsupported-mode behavior |
| R4 Native benchmarks and training | GPU Agent | Small original SAEBench, SAELens train/load, Sparsify TopK/transcoder/resume | Native result, real checkpoint, source-linked comparison or honest mismatch | Outputs can be reopened with SAEBase, not just tool-specific logs |
| R5 Circuit and cross-layer | GPU Agent + Integration Agent | One upstream graph; official Scope assets, original JSON→OpenSAE graph; actual feature edit | Interactive saved circuit with signed/error edges and visible edited readout | At least one native graph and compatible node edit |
| R6 Private worker and public staging | Integration Agent + GPU Agent | Job queue end-to-end, result retrieval, public CPU Docker, staged HF deploy with real bundle | Private runnable job plus stable publicly browsable staging Space | Restarted browser works; no unauthenticated GPU route |
| R7 Usability, performance and release | Integration Agent + 3 external users | Repair user blockers, record observed load/latency, publish preview and rollback path | Launch/readme/screenshots, operator guide and issue backlog | A stranger can finish the recorded-real task unaided |

**Critical dependency:** R0 → R1 → R2 → R6 public staging → R7. R3 starts once R1 works. R4 and R5 can proceed in parallel with CPU UI preparation when their source assets are ready. **Do not delay the useful public recorded-real deployment waiting for native training, a multi-GPU allocation, or NDIF credentials.**

### R0 — Source, CI and dependency control

**Integration Agent owns:**
- Verify PR stack `#50 → #51 → #52 → #53` remains open and correctly stacked. Keep WP13 on a new branch from the *actual* WP12 head, not main. Preserve P0 evidence and old Work Plans.
- Work from an immutable source SHA. All PR fixes must be made via a small branch or against the active integration branch with recorded commit; never modify a running training checkout in place.
- Create isolated and documented environments: CPU app (FastAPI only), legacy TL3.9/SAELens6.54 baseline, TL4 Bridge worker, local NNsight, native pyvene, native SAEBench, native SAELens trainer, Sparsify trainer, circuit-tracer. Each needs `pip freeze`/`pip check`, no dependency soup or retroactive changes to GPU-00.
- Run the existing GitHub `CI` and targeted `apps/research-lab/tests`. Note current generic CI does not by itself prove TL4/NNsight/CLT with real checkpoints or deployed browser behavior.
- Confirm whether the current HF Space is really built from the same source and what historical evidence is visible. The publicly accessible Space status was not established during WP13 authoring; treat as UNKNOWN until inspected.
- Add an operator-facing version/availability panel and build manifest, deploy scripts, data publication recipe and short user journey smoke.

**GPU Agent owns:** device/driver/disk/preloaded-checkpoint inventory with exact versions, not a rerun of the old four GPU-00 parity cases.

### R1 — Real checkpoint/environment compatibility, not exhaustive parity

Recommended first routes (only if sources actually resolve on the host):

- **Legacy known-working anchor:** Pythia-70M with `pythia70m_resid_sm` L3 (and L5 only when needed); Gemma-3-270M with `gemma270m_scope_resid` L5/L9. Prior GPU-00 *reported* parity here using TL3.9/SAELens6.54 fp32. Do **not** spend hours repeating it.
- **New route:** separate TL4 `TransformerBridge` environment, one chosen model/SAE/site. Compare that run's token count, hook firing, code shape, no-op, one ablation; it is acceptable to mark TL4 incompatible and continue the product demo on pinned legacy while the integration agent repairs the adapter.
- **Official Scope2:** run `load_official_scope2()` with exact official repo SHA, directory/config and safetensors; inspect actual JumpReLU, affine skip, input/output hook site against the model. For CLT use `load_official_scope2_clt()` with all required shards only when source model/site and VRAM fit.
- **NNsight/pyvene:** one genuine small local model with capture→edit→result, one multi-input/gradient as appropriate, one native generation-step edit. Mock-based CI is insufficient to claim a real site works.
- Freeze a `working_routes.json`: `backend`, `python`, `model_sha`, `sae_sha`, `revision`, `dtype`, `site`, `operations_supported`, `actual_seconds`, `peak_vram`, `failure`. Any route lacking exact coordinates stays UNSUPPORTED, not “passed by equivalence”.

### R2 — Build the first authentic 10-prompt experience

**GPU Agent:**
- Use ten short prompts authored specifically for the demo or otherwise rights-cleared, varied enough to show distinct feature behavior (numbers, geography, negation, syntactic structure, code, etc.). No invented guarantee that one specific feature matches a semantic pattern.
- Record exact prompt bytes and token IDs, 1–2 **compatible** registered sites; record discovery vs evaluation groups separately for candidate explanations. Reuse a loaded model across prompts for efficient capture; build a batching driver if the existing CLI reloads weights on every call.
- Generate `backend.feature_store`-compatible rows, `backend.explore`-compatible *complete* sparse token traces and optionally honest candidate interpretations. The existing `scripts/wp12_p2_capture.py` is a **one-prompt** CLI; a 10-prompt run controller is an integration task, not an implicit feature.
- Return a small rights-reviewed candidate bundle and source coverage report; no 500-group rerun, no third-party mass-text export.

**Integration Agent:**
- Confirm exact `site_id`, model/SAE weight digests and tokenizer SHA against the current verified site registry. If new real weights do not match a historical site, create a proper new source identity registration, never edit hash fields to make an import succeed.
- Convert only operator-approved raw records with `create_feature_store` and `publish_exploration`, place them under `OPENSAE_LAB_DATA`, run API and browser smoke.
- Prefer **one excellent functioning sample** before processing all 10. Do not confuse a single correct feature page with all 10 prompts being imported.
- Build 2-way Feature↔Prompt links, clear descriptions for “reviewed example”, “candidate explanation” and “recorded vs new input”, and image/screenshot/HTML export tests.
- Publish at most 20 reviewed site/prompt traces within the existing 32-trace/8 MiB constraints. If those caps are too small, fix format/index deliberately while retaining integrity safeguards.

### R3 — Real steering and analysis, grounded in measured results

- Work on 2–3 actually activated features from R2, choose 1–2 prompts per feature.
- Teacher-forced: baseline, identity, ablate, scale at same prompt/position with exact code and source identities. Prefer demonstrating meaningful differences but do not cherry-pick away no-effect controls.
- Generated response: test only backends that actually support edited generation. Record generated-token IDs and readable before/after text. Use the existing `saebase/interventions/steering.py` and native NNsight adapter; first repair API semantics in real environment when needed.
- Add `LocalAutoInterpreter` run on a small already-downloaded instruct checkpoint; require at least one activating and one inactive held-out group and clear “hypothesis, not proof” description. If no compatible local instruct model fits, retain genuine observed feature cards without fabricated descriptions.
- Export a clear `interventions` result: baseline, edited state, measured delta, source, operation, token location and human-readable explanation. A static public screenshot must never imply an arbitrary live input was processed.

### R4 — Original native benchmarks and short training

- Do **not** run a native evaluator until its genuine original model, SAE, dataset and custom representation API are supported. The existing `m3_native_saebench_smoke.py` and `saebase/wp12/custom_saebench.py` are first pilot entries.
- Start with ≤one Core and one Sparse Probing bounded native evaluation on one compatible public model, then use a second distinct checkpoint at the exact same model/site/dataset only if it exists. Do not mix different layer, hook or eval protocol to invent a two-SAE Pareto line.
- Train a small TopK SAE using original native SAELens or Sparsify on an actual owned/approved local dataset snapshot. Take a checkpoint, load/encode/decode with SAEBase, demonstrate resume of the native training state if original API supports it; distinguish fine-tune/new-run.
- New representations get registered as new identities; expose them in the Training and Benchmarks sections with source metadata. No overwriting existing official SAE assets.
- If source API has changed or original `--resume` fails, the GPU Agent files a *targeted reproducer + proposed patch*, not a 1000-line GPU report or unsupported success assertion.
- F50 remains blocked on one GPU. A 2-GPU request is not part of this stage's default resource budget.

### R5 — One actually measured circuit

- Ensure the original pinned circuit-tracer CLI, compatible model and source official transcoder set are available. Check actual CLI flags, not just stored README commands.
- Run exactly one small native attribution with a short preselected prompt/target and save its original graph JSON/export. No claims that “a displayed graph means original gradients ran”.
- Convert via `import_native_graph_file()`, keep token/feature/error/logit node types and edge signs. Show the graph inside the Research Lab and retain original source SHA.
- Select a real feature node and run `execute_transcoder_node()` with actual model and representation identities; for CLT try `execute_clt_feature()` only with real pinned shards, matching native hook names and available memory. Show computed output differences and reject mismatched source sites.
- Annotate/pin/group/export; open the export in a fresh browser session. Unsupported cross-layer scenarios should clearly say why, not quietly use a same-site SAE.

### R6 — Actual deployment and recovery

**Private integration first:**
- Verify genuine `/api/private/inputs` → `/api/private/jobs` → worker claim/execution → `/api/private/jobs/{id}` → authorized `/result`. Start with `inspect` and one genuine feature edit.
- Existing worker currently loads the model on each job. Add **persistent process / checkpoint reuse** when repeated tasks make latency unacceptable, with source-aware cache and explicit load/unload policy. This is a product improvement, not a scientific test.
- Test no-token/wrong-token (401), malformed/new-vs-stale source, user cancellation before claim, retry after worker interruption, max queue and large input limits. Logs must not print private prompt text or tokens. Do not expose port 7860/8000 publicly with private jobs enabled without owner-approved TLS/auth perimeter.
- Observe actual queue wait, model load, first-token/response latency, GPU memory peak and failure count; only publish values measured on the actual machine.

**Public staging:**
- Build the **actual repo-root** Docker context using `apps/research-lab/Dockerfile`. Include verified historical Stage D evidence plus new approved `features/`, `explore/` and `published_wp12/` bundle. Source and data manifests must describe what was mounted and displayed.
- Use a staging Space or isolated URL; capture build/runtime logs, Space revision and screenshots. Avoid overwriting `OpenSAE/SAEBase` before confirming its current working revision and owner approval.
- Smoke exact browse and data links, startup from clean image, fresh browser, restart, read-only `POST /api/jobs` = 503, accurate compute availability and no private endpoints.
- Hugging Face storage volumes are optional; verify permissions and revision immutability. Do **not** rely on the ephemeral container filesystem for persistent user work. If a public dataset release is desired, separately review license, privacy, model/token provenance and stable dataset path.
- Promote only after staging works. Roll back to captured prior Space revision on blank data pages, changed science metrics, secret leakage, wrong role authorization, repeated 5xx or broken navigation.

### R7 — Lightweight, actual usage tests and launch

Recruit three real researchers who have not been coached through the application. Give each one a **single concrete task**:
1. Find a feature from a particular recorded prompt and explain what the measured token activation shows.
2. Find and compare baseline vs intervention on that same real prompt, recognizing whether it is recorded or live.
3. Open a real circuit/benchmark or training result and export a usable source-linked artifact.

Capture only task completion, UI blockers, short user feedback and publicly safe screenshots. No extensive controlled statistical study. Repair broken navigation or confusing scientific labels before a preview announcement. Keep usage/latency observations honest; publish only the supported routes, not 50/50 “scientifically certified”.

## 5. Concrete interface/product repairs required

These are **not** repetitive compliance paperwork. They are likely real product blockers visible from current source inspection.

| Priority | Current code reality | Repair expected from Integration Agent | How to know it works |
| --- | --- | --- | --- |
| P0 | PRs #50–#53 are stacked, `main` is still behind | Freeze a current integration checkout, run clean install/CI, prepare merge/promotion plan | Clean clone of selected SHA starts CPU Lab and relevant local Python entrypoints |
| P0 | CPU Docker installs only FastAPI/uvicorn and copies old evidence | Add versioned reviewed public data bundle mount/copy with correct paths; no torch requirement in CPU app | Docker shows real WP13 prompt/feature cards after restart |
| P0 | `capture_prompt_trace` and `wp12_p2_capture.py` run one prompt and load weights | Bounded multiple-prompt capture job that **reuses one model** | 10 prompts collected without ten model downloads/reloads |
| P0 | Features/Explore depend on source hashes and manually approved records | Site registration/import and human-reviewed publication helper | Exact source hashes imported, user-facing card opens; no forged identities |
| P0 | Private worker has one-shot command and often one model load/job | Run supervisor, model cache, leasing/retry, better per-job stages | Second request reuses compatible loaded model and succeeds after interrupted worker restart |
| P0 | P3 intervention results are raw tensors/CLI reports | Publish reviewed before/after examples with explicit text vs logits labels | User can identify effect and reference the original prompt/source |
| P1 | New `tl4`, NNsight, pyvene, Scope2 and CLT adapters have mostly CPU fixture tests | Isolated genuinely installed native runtime probe + small fixes | At least one actual compatible model site executes each claimed adapter subset |
| P1 | P4/P5 runners may depend on upstream changed configs and datasets | Worker-specific native source pin, run one original job, fix only exact changed call signatures | OpenSAE consumes a new authentic native artifact |
| P1 | Circuit graph viewer can load typed saved JSON, but live graph provenance needs real upstream | Genuine circuit-tracer run, JSON import, graph and real node edit | Measured graph available in browser and edited model readout recorded |
| P1 | Existing HF Space deployment revision is not established by code | Staging launch, accurate manifest & status, explicit backup/rollback | Browser publicly reachable without private GPU |
| P2 | Public fresh-input inference / NDIF / multi-GPU are not available | Keep off by default; propose authorization/hardware only if actual user need | No misleading UI or fake service availability |

## 6. Compute efficiency and task discipline

Default resource tiers are practical controls, **not strict parity-certificate gates**.

- **CPU:** documentation, UI, site contract, public bundle preparation, FastAPI tests, Docker build, rights review and staging release. Integration Agent can do this without GPU.
- **Small GPU:** one official site or small model, 3–10 real prompts, short ≤8–16 generated-token trials, bounded checkpoint loader or trainer pilot. Begin with the previously observed small Pythia/Gemma model route. Budget one task at a time and record the actual throughput; avoid repeated checkpoint load across prompts.
- **Moderate GPU:** a curated 10–32 prompt corpus, a small original SAEBench pilot, one tiny training/restart, one original circuit. Only use if a useful end-to-end feature needs it.
- **Expensive/multi-GPU:** broad 48-site or full 500-prompt experiments, many dense graph gradients, large multi-run training, real 2+ GPU. Not part of this plan's default execution. Ask for explicit authorization and additional hardware only when required.

The GPU Agent checks free VRAM/disk before every new source family, reuses local HF cache and immutable model revisions, records peak memory and actual wall time, and stops if an asset has incompatible hooks, missing authorization, OOM, or other materially wrong identity. For **runtime failures**, preserve one representative traceback and fix the lowest failing operation; do not package 576 raw tensors for every feature.

## 7. Agent cooperation and handoff rules

### Integration Agent (ChatGPT/GitHub, CPU/UI)

Owns: frontend, public API, data ingestion/publishing, source registry, source contracts, staging Docker and Space rollout, tests, documentation, merge and release branch. Maintains a canonical task queue and decides which GPU task gives the highest immediate product value.

For every GPU job, supplies a single frozen card:
`job_id, code_sha, exact model/SAE/hook, input source, 3–10 prompt IDs, operation, output path/schema, max GPU/disk budget, success UX, fallback if unsupported`.

When output comes back:
1. Read **the small summary / real preview / one failure** first, not force the author to move huge evidence ZIPs.
2. Fix bugs on a separate code branch and push a focused PR; never modify the GPU Agent's stored raw run.
3. Turn measured results into approved browser assets and verify API/HTML correctness.
4. Schedule only the next necessary GPU job; don't rerun an already successful job solely to re-certify it.
5. Update **operational** status and source links in WP13. Do not change WP12 P8 verification state without its separate standards.

### GPU Agent (local hardware and original libraries)

Owns: actual model/SAE weights and tokenizer source, CUDA/driver/dependency resolution, downloaded official checkpoint load, real NNsight/pyvene/SAEBench/SAELens/Sparsify/circuit-tracer execution, checkpoint snapshots, raw run outputs and screenshots of its local use.

Does not: overwrite main, rewrite upstream estimators, publish private prompts or weights, open public GPU services, install everything into one uncontrolled environment, declare source-only mocks a live model success, or repeat GPU-00.

**Code ownership split:** Integration Agent edits `apps/research-lab/`, interface contracts and release scripts on WP13 integration branch. GPU Agent edits only isolated worker-adapter fixes through a separate branch/PR (`fix/wp13-<job>-native`). The Integration Agent reviews, merges and then updates the runner's base SHA for the next job. If both need one file, reserve ownership explicitly first.

**Reporting:** one concise result per job, with `PASS_WORKING / PARTIAL_RUNTIME / BLOCKED_ENV / FAIL_FUNCTIONAL`, exactly what the user can do, observed performance and the next fix. Do **not** mark “PASS” merely because a module imports or a mock test passes. Run directories and raw source objects stay on GPU host.

## 8. Lightweight acceptance that serves deployment

This is intentionally **not** the P8 independent numerical certificate.

For a **real operation**, PASS_WORKING requires:
- Genuine original model / official SAE/trainer or native service executed, at correct source identity, in a pinned environment.
- Selected user action produced a measured observable output and was not a silent no-op. Capture of actual token codes, edited logits/response, trainer checkpoint, graph or benchmark counts.
- No blocking runtime failure along that user journey. One simple relevant negative case rejects correctly (e.g. unknown hook, mismatch or unauthenticated request).
- Result can be loaded or displayed by the intended consumer, not stranded in a private stdout log.
- A reproducible command/reference and minimal run manifest exist. No demand for exhaustive float-array comparisons.

For **public deployment**, PASS_WORKING requires the actual staging/production app to load in a fresh browser, display its approved real records, support navigation/export and survive restart; public GPU submission must remain disabled without an intentionally authorized service.

**Evidence per job:** original SHA, environment info, 1–2 actual examples, source-file/link, runtime/VRAM measurement and one error/fix if any. No new 150 MB transfer requirement. Reserve dense raw tensors locally for scientific problems.

WP13 task states: `NOT_STARTED | WORKING | INTEGRATED | STAGED | DEPLOYED | PARTIAL_RUNTIME | BLOCKED_ENV`. Do not confuse these operational states with `current_status` and `p8_status` in the WP12 ledger.

## 9. Merge, deployment, privacy and rollback

1. Keep the source PR stack intact until base integration is tested. Merge source PRs in dependency order only after targeted tests pass; ensure a final new `main` CI run before any public Space promotion. No rebase or history rewrite of user-owned commits.
2. Freeze a manifest pairing `git SHA + library lock + model/SAE source + data bundle SHA + Space revision`; the UI shows recorded/current scope.
3. Deploy to staging first; verify local Docker and staging HF asset equality. Only then replace an approved organization Space revision.
4. Store operator/worker keys exclusively in a secret manager or authorized Space secrets, never in README, URL query strings, GitHub logs or example bundles. Private APIs must not be hosted publicly inside the CPU-only Space.
5. Keep root data private until rights reviewed. HF Storage Buckets/volumes should be mounted with the minimum access needed; staging production artifacts must be immutable/versioned separately even if bucket storage is mutable.
6. Roll back at once on 5xx browse failures, blank feature/circuit data after deployment, wrong model/SAE identities, leaked input text, role auth failures or broken old results. Restore previously recorded Space revision and previously approved data bundle without rerunning GPU.
7. Maintain a small public `Supported now / Recorded examples / Needs local GPU / Not available` legend so users do not infer universal coverage.

## 10. Completion definition and direct next action

**WP13 first milestone:** a genuinely working public **recorded-real** Research Lab backed by at least ten approved short inputs and one real end-to-end source-bound feature/intervention example, plus an authorized **private-live** prompt capture route. At that point public usability and integration matter more than multiplying arbitrary model rows.

**WP13 broader completion:** a native SAE comparison and checkpoint, a real circuit, robust private queue and a staged/approved public deployment with user testing. F20/F50 remain separately marked external blockers if NDIF or two GPUs aren't available.

**Start with the companion's LIVE-00 preflight and LIVE-01 tiny real model+SAE route**, but have the Integration Agent prepare the staging CPU deployment and data-ingest repair **in parallel**. **Do not rerun GPU-00** or launch expensive training/circuits before one complete user journey works.

This is a new operational work plan. It supplements WP12's scientific final acceptance and does not change the historical P0/GPU-00/Stage D observations, original algorithms or P8 ledger.
