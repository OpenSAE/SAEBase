# WP17 — Replace Seven SAEBase Playground Demos with Real Compute

**One execution handoff. Updated 2026-10-23.** Do not create another roadmap. This supersedes the web-interaction steps of WP16; it does not replace historical work or require repeated WP12/13/15 experiments.

## Decision

The public site at https://atheavenma.github.io/opensae/workspace.html is **now a sample-first, seven-module playground**, built on GitHub Pages. It requires no server to explore its interface. Every primary module already has (a) a filled example, (b) working controls, and (c) a displayed result. The user never has to guess a model ID or API URL just to try a page.

**Important:** any computed values synthesized in the browser are labeled `Demo`. Real WP15 metrics and recorded original continuations are labeled `Measured`. Never present a simulated value as model inference. The GPU agent must **replace the demo implementations with existing genuine model operations, not redesign or rebuild the front end.**

Website code: `AtheavenMa/AtheavenMa.github.io/opensae/workspace.html`, `opensae/workspace-play.js`, `opensae/workspace-play.css`; PR #5. Retain all visual/layout, labels and interaction decisions unless an actual API contract requires a minimal revision.

Existing compute code: `OpenSAE/SAEBase` branch `feat/wp15-six-workflows`, PR #57; its WP13/WP12 stack, GPU-00/01/02/03 records, `/workspace/wp15/staging/rc1` and original model/SAE weights. **Do not rerun** G1–G5 to manufacture another screenshot, regenerate the previously approved trace, or create new UI libraries, framework migrations or architecture plans.

## What is already done — GPU agent must not redo

- All **seven** frontend modules, controls, navigation, example presets and response rendering: **done**.
- Desktop 1440 and mobile 390 browser tests: **done** (no JS error, no horizontal overflow).
- Each module's primary action produces a result. Cross-module **Explore → Intervene → Generate** works. Circuit edge-strength slider, clickable/keyboard-accessible nodes, evaluation metric switch, training parameter switch, interpretation evidence selection work.
- Web API disclosure is optional. Existing private owner flow with `/api/jobs/preflight`, `/api/private/inputs`, `/api/private/jobs` and result polling was exercised through a **test-only mock API**, not a real hosted service; no model results fabricated as successful server execution.
- Static original WP15 `benchmark.json`, `training.json`, `generation.json`, `circuit.json`, `explanation.json` already publish source-bound numeric data in GitHub Pages. G4 graph counts 155/748. G3 one-SAEBench-run L0 376, MSE ~0.025, sparse probing accuracy 0.59375. G5 genuine 50-step Sparsify checkpoint reload. G2 measured source-approved 16-token continuations are used only when the prompt and operation match the original source record.
- Contact/Vision/homepage copy and Python API documentation were handled in earlier website PRs.

## What the GPU agent must execute

**Goal:** replace one simple demo action per module with a real running operation using existing assets. Keep all presets. The interface should choose a known working model and feature by default; a visitor should *never* need to look up hook names before seeing a result. Model-dependent tasks may run through a private operator-authenticated service, not free public anonymous arbitrary-prompt GPUs.

### 1. Explore — real prompt → tokens → activated features

Current browser `runExplore` uses a seeded illustrative feature map for any prompt. Replace this only when the real backend is reachable.

Reuse `saebase.explore.capture_prompt_trace` (Gemma-3-270M / `gemma270m_scope_resid` / L5) or already approved `WP13-DEMO-01` (`/workspace/wp15/staging/rc1`, 639 nonzero rows). On preset example, return the matching token rows, feature IDs and measured activations; for a new prompt, enqueue an authenticated fresh-input inspect job. Preserve token-to-feature selection and reusing selected Feature in Intervene.

Example existing `scripts/wp12_p2_capture.py`; private worker `apps/research-lab/backend/private_worker.py`. Do not expose unreviewed prompt text or private trace DB publicly.

### 2. Intervene — real edit from an active feature

Current `runIntervene` uses the recorded Gemma F1819 ablation max logit change ~9.802836 for the exact boats prompt; other parameter combinations are `Demo`.

Connect selection, operation (ablate / scale / clamp / add), and strength to the real native `compare_sae_edits` via existing `private_api.py` and `private_worker.py`. Return source/model/site, measured baseline/edited next-token IDs, logit delta and user-visible error. Existing **private** WP15 edit proved HTTP 401 for bad tokens and nonzero logit changes: reuse that worker. For a target feature inactive at the selected position, show a meaningful message rather than a fabricated edit.

### 3. Generate — actual autoregressive feature edit

Current `runGenerate` shows existing **measured** original WP15 continuations for exact approved G2 prompts; other prompts use clearly marked demos.

Current private queue disallows `max_new_tokens>0` and only has teacher-forced edits. **Extend that exact queue/worker**, not a new framework, to wrap the already working `saebase.interventions.steering.compare_steered_generation` / `scripts/wp15_steer.py`. Add bounded operator-authenticated `generation` job with 16–32 new tokens, strict model/SAE identity and no-op/control/edited comparisons. Return decoded baseline and edited outputs with per-step logit effects. Preserve the original WP15 measured values and do not auto-publish private input texts.

### 4. Circuits — real graph browsing and optional source-consistent tracing

Current browser loads the original recorded, sanitized native G4 Gemma-2-2B graph (155 nodes, 748 browser edges). Working graph filter, zoom via browser, click/keyboard node selection already exists.

Keep the first working capability as **open and inspect a real graph**. If there is compatible native circuit-tracer execution already available, add a bounded operator job calling `run_native_circuit_tracer` and `import_native_graph_file` with matching original transcoder; otherwise **do not block the working graph viewer** or invent a circuit from the G1 Gemma-3 SAE.

### 5. Evaluate — real native benchmark results and optional new evaluation

Current browser metric selector reads original G3 native SAEBench results. Do not rerun this for a demo. For a user-authored evaluation, reuse `saebase.wp12.benchmarks.native_pilot` and original fixed SAEBench upstream `8042bb3828c6340da8d12062324e92b2077c571c`; make one authenticated queued operation with compatible model/SAE/dataset config and source-linked report. Otherwise the existing measured metric exploration **already constitutes one useful playable page**.

### 6. Train — real existing checkpoint plus optional short new run

The exact preset TopK / K32 / 50-step option uses the original WP15 Sparsify run; alternative `steps`, `K` and architecture currently generate an explicitly **simulated** response. Use `saebase.wp12.training.TrainPlan` and `run_training` for an operator-authenticated, bounded original native run, with nonblocking job progress, checkpoint registration and **actual** checkpoint reload + encoding test. The existing native 50-step checkpoint is proof of integration, not proof of high SAE quality. Do not train a big SAE solely for site UX.

### 7. Interpret — real candidate and separately evaluated evidence

The exact F1680 preset displays original diagnostic score **0.50** on 3 positive and 1 negative held-out prompts and correctly says no approved explanation. Other feature hypotheses and scores in the browser are **Demo**.

Reuse original local Qwen2.5-0.5B-Instruct and `scripts/wp12_autointerpret.py`, with actual distinct positive/negative activation contexts from G1 capture. Return a candidate, held-out score, evidence counts and review state; never publish candidate semantic text without authorization. A weak or unreviewed candidate remains hidden rather than giving a false feature label.

## Shared API contract / deployment

- Existing browser supports `GET /api/health`, `POST /api/jobs/preflight`, `POST /api/private/inputs`, `POST /api/private/jobs`, `GET /api/private/jobs/{job_id}`, `GET /api/private/jobs/{job_id}/result`. Keep these paths and add new operation types to existing queue, rather than redesigning all seven screens.
- If the GPU host is not 24/7 available, start it only for operator tests. One HTTPS origin from an approved existing host or temporary **authenticated** tunnel is sufficient. Keep `OPENSAE_CORS_ORIGINS=https://atheavenma.github.io`. CORS is **not** authentication. Restrict access to GPU job routes with strong operator secrets, request limits and no raw prompt leakage. Never put secrets in GitHub or static HTML.
- Small portable result schema for each completed operation: `{status, model, site_id, feature_id, operation, input_sha256, result, runtime_seconds, source_commit}` plus operation-specific token features or generated outputs, or a job ID with progress. Reuse existing Pydantic records and small JSON adapters, don't invent parallel backend schemas.
- Real processed input and results must be displayed where the current Demo result appears. On no compatible backend, the public sample play mode still works and stays marked Demo. Network/remote failures get explicit error messages, not a counterfeit measured result.
- Work in one feature branch on top of current WP15 development, commit source/tests, and give the specific existing website PR or follow-up patch required to replace demo computations. No new branding/visualization work.

## Acceptance for GPU Agent

1. A visitor opens all seven tabs, selects the supplied example and gets a result **without providing an unknown model ID**.
2. For authenticated operator tests, a **new** prompt runs through real Explore and Intervene; a real generation job yields decoded results; live native benchmark/training/interpretation may use their existing checked original runs unless a missing worker contract requires a tiny targeted run. Circuits shows the original source graph correctly.
3. When a real backend is connected, the browser clearly distinguishes live measured output from demo mode. No public source contains credentials, giant raw logits, unreviewed prompt text, model weights or private SQLite.
4. Deliver exact backend host URL or approved local/HTTPS tunnel URL (only if actually running), asset paths, GPU versions and run receipts for any new execution, endpoint-response examples, one mobile and desktop browser screenshot, API command and relevant PR links.
5. Mark each item **REAL LIVE / REAL RECORDED / DEMO ONLY** in this file. Update it in place. **Do not generate another plan.**

## Execution order

First expose the already approved compact data for normal browsing, then complete private fresh-input Explore + Intervene, then generation, then optional per-task native jobs. Existing result pages need no further GPU runs. **Use the fewest operations required for a genuinely working product.**
