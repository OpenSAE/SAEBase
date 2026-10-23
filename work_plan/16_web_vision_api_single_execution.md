# WP16 — OpenSAE Website and Working SAEBase API

**Date:** 2026-10-23  
**One execution plan.** This document is the only new work plan for this release handoff. Do not produce a new roadmap or repeat WP12–WP15 GPU experiments.

## Decision

The **OpenSAE organization website** is the primary front end. Hugging Face Research Lab is not the product homepage. Preserve the existing HF Space and its likes/revision; do not replace or delete it.

Public website: **https://atheavenma.github.io/opensae/**  
Website source: `AtheavenMa/AtheavenMa.github.io`, `opensae/`, merged website PR #2.  
SAEBase source: `OpenSAE/SAEBase`, `feat/wp15-six-workflows` (PR #57), stacked onto WP13/WP12.  
Do not merge WP12/13/15 into `main`, modify the HF Space, buy compute or claim hosted GPU serving without an actual verified deployment.

### Product identity and text

- Rename the **Foundation** navigation and page to **Vision**. `/opensae/vision.html` is the canonical page; `foundation.html` redirects to it.
- Contact Q&A uses the exact copy: **“An open research non-profit organization building tools for mechanistic interpretability. SAEBase is our main software platform.”** No other status footnotes or decorative explanations.
- Preserve Source Serif 4 / Inter, website palette and extreme simplicity. No meaningless small labels, card walls, fake readings, invented functions or decorative API descriptions.

## A. Work already completed — DO NOT REDO

1. **Frontend published in GitHub source:** Vision navigation + URL, Contact text, standalone `/opensae/api.html`, actual `/opensae/workspace.html` API inputs and `workspace-live.js`/CSS. Website PR #2 is merged into `main`. Preserve the personal academic homepage unchanged.
2. **Workspace no longer hardcodes feature/benchmark/training numbers.** Connecting to a real compatible API fetches genuine sites, approved prompt traces, token features, feature search, recorded WP15 results and real circuit nodes/edges. An unavailable API shows errors instead of a fabricated answer.
3. **Intervention UI:** Existing `POST /api/jobs/preflight` checks parameters without pretending to execute; actual operator-only `POST /api/private/inputs` → `POST /api/private/jobs` → `GET /api/private/jobs/{id}` and `/{id}/result` runs a supported private edit, with token entered by operator and never embedded into site source or storage.
4. **Python and HTTP usage:** `/opensae/api.html` documents installation, `/api/health`, `/api/sites`, `/api/explore/traces`, `/api/explore/search`, edit preflight/private worker, and actual G2 generation / G3 benchmark / G4 graph import / G5 training / G6 interpretation code entry points. Refers to the WP15 branch until merged.
5. **Backend browser support:** `apps/research-lab/backend/api.py` on WP15 has an explicit `OPENSAE_CORS_ORIGINS` allowlist with `https://atheavenma.github.io` as default, GET/POST + Authorization/Content-Type headers, no credentialed wildcard. A CPU TestClient preflight test was added to `test_platform.py`. No API secrets in public JS.

The frontend is operational against a **reachable compatible FastAPI instance**; it is not yet a fully functioning **public hosted compute service** because no such endpoint has been connected to the production site. Do not claim public inference is available before end-to-end verification.

## B. GPU Agent's required actions — execute, do not redesign

### B1. Deploy the already-built API

- Use the existing `apps/research-lab/backend/api.py` FastAPI app and Dockerfile. It is a **read-only CPU API** by default, separate from the GitHub Pages website. Choose an existing approved deployment environment with HTTPS (existing owned staging host or a no-new-cost supported host). Do not convert the front end into Gradio or rebuild the API.
- Deploy exact source from `feat/wp15-six-workflows`. Set `OPENSAE_LAB_DATA` to a read-only mount containing reviewed permitted data. Configure `OPENSAE_CORS_ORIGINS=https://atheavenma.github.io`.
- Prove GET `/api/health`, `/api/sites?limit=5`, `/api/explore/traces`, `/api/wp15/benchmark`, `/api/wp15/circuit`; POST `/api/explore/search` and `/api/jobs/preflight`. Verify CORS OPTIONS requests from the exact website origin and rejection of unapproved origins.
- Return the **real HTTPS base URL** and its deployment revision. Operator must then set that URL in the live Workspace's API field; if using a canonical public API URL, add it as a default in the front end after verifying health. No guessed hostname.

### B2. Connect authentic WP13/WP15 data — no new runs

Reuse the existing private assets and source records:

- **G1**: one previously rights-reviewed LIVE-02 `WP13-DEMO-01` trace already staged locally at `/workspace/wp15/staging/rc1`, 639 nonzero codes. Point the CPU service to this bundle; verify Prompt → Token → Feature. Do not regenerate codes.
- **LIVE-03**: original two measured L5 F506 / L9 F95 teacher-forced comparisons: use `scripts/wp13_stage_live03.py` and existing source-bound approved records *only when* the original approved source files are available. Do not invent new records or mix separate runs.
- **G2**: two reviewed original generation strings + numerical record in the WP15 local staging data. Keep exact approved text, original prompt hash and ablation/clean readouts.
- **G3**: native SAEBench result already recorded (Core L0/MSE and Sparse Probing). **G4**: Gemma-2-2B native graph of 155 nodes, displayed 748 edges. **G5**: 50-step Sparsify training and actual reload. Use existing WP15 import/record adapters; don't rerun their expensive native work.
- **G6**: prior held-out score 0.50 is not an approved interpretation. Keep empty/clearly unsupported interpretation view until a meaningful candidate is actually generated and reviewed. Never invent a semantic label.

### B3. Make operations truly executable from the Workspace

The public read-only site is already wired. For an **owner-authorized** GPU service:

- Start with the existing durable, token-authenticated `backend/private_api.py`, `private_worker.py`, `jobs.py`, `scripts/wp12_private_worker.py`, and model/SAE caches. Previous G1 owner queue demonstrated inspect and ablate with meaningful logit change. **Do not retrain/recalculate G1–G5.**
- Make a secure **HTTPS API endpoint** reachable by the browser, or use a trusted secure operator gateway. Never expose loopback HTTP directly to the HTTPS page; never place operator or worker secrets in GitHub Pages source, URLs, logs, or public examples.
- Verify one real new operator prompt using the Website: connection → model selection → feature capture/inspect → preflight → authenticated ablate/scale → refresh result. If a model site is absent from `/api/sites` due to historical 48-site identities, expose a separate exact verified live-capability catalog; **do not** silently pretend mismatched site IDs are interchangeable.
- Then integrate real **Generate** using the existing `scripts/wp15_steer.py` or `compare_steered_generation`; **Train** using `saebase.wp12.training` (Sparsify/SAELens native); and **Interpret** using `scripts/wp12_autointerpret.py`, each through **bounded authenticated tasks** using the existing queue architecture, not a new worker system. For **Evaluate** and **Circuits**, retain real recorded-data browsing and documented local-native commands; add GPU task buttons only if a compatible worker can execute the original upstream methods.
- Complete the smallest working user journey for each module. Do not promise public anonymous arbitrary-prompt inference, paid hosting, new training quality or automatic publishing of model-generated explanations. Give meaningful failure messages and working input validation.

### B4. Finish site/API integration

- Update `opensae/workspace-live.js` and `api.html` **only** where the exact deployed API or new job-contract requires it. Preserve the existing V5 visual structure and the minimal text standard.
- Provide one actual **desktop browser** and **mobile browser** test from the live public site with HTTPS API, including screenshots of real G1/G2/G3/G4/G5/G6 records (G6 may show no approved interpretation if still low quality), plus a completed live ablation.
- Check origin allowlist and worker authorization (invalid token must return HTTP 401), failed preflight, restart/persistence, repeat requests, no raw private prompt leakage, and hosted API health.
- Deliver code commits and PRs, the actual public/API URLs, tested commands, deployment revision/rollback path, and a compact final result report. No fake success, no giant raw ZIP.

## C. Acceptance and finish line

**Already finished by integration:** static site, Vision, Contact, API examples, API-aware Workspace layout, FastAPI CORS code.  
**Agent remains responsible:** reachable HTTPS API, reviewed data mounts, actual browser/worker end-to-end operation for each supported method, remote browser evidence, additional live job routes where absent.

A feature is complete only when a new user can see its real data or run its supported compute action. A page merely loading is not a compute acceptance.

**Final site:** `https://atheavenma.github.io/opensae/` until OpenSAE-owned domain/Pages is configured. The HF Space is secondary. Do not change it.

**No new plan.** Fix issues in code, add tests, and update the existing WP16 status section only. Do not repeat the WP12–WP15 architecture work or create another planning phase.

## D. Current status (update in place, do not create a new plan)

- Website: **MERGED AND DEPLOYED**, GitHub Pages build+deploy SUCCESS (site commit `1fef49acb27ad3b66dbd40da3d1b84fd0512ca4a`; Pages run `39910392095`). Live front end includes Vision, Contact, API and the connected Workspace assets.
- Frontend API wiring: **IMPLEMENTED**; actual remote backend health check still required.
- Backend CORS: **IMPLEMENTED ON WP15 BRANCH**, tests require CI execution.
- Public HTTPS CPU API: **NOT VERIFIED**.
- Authenticated browser→GPU worker: **NOT VERIFIED**.
- G1–G5 private original source runs: **AVAILABLE FROM WP13/WP15**, reuse.
- Reviewed G6 explanation: **NOT AVAILABLE**.
- Public remote end-to-end UX: **NOT VERIFIED**.
