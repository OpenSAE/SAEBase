# WP12-00 — Source-based capability inventory and execution baseline

Date: 2026-10-21
Source baseline: `main` at `f9c1f4e07f14bda4ab4724ae399138249cc7b3ba`
Status: **CPU source inventory complete. P0 gate NOT PASSED.** No fresh CPU test, GPU run, or public deployment acceptance is claimed.

## Scope and method

This audit reviewed the repository tree, selected current Python API, registry, experiment-store, intervention, native-result importer, Research Lab backend/frontend, test code, release notes, Work Plans, GitHub CI/PR/release metadata, and the 50-feature roadmap. The existing records remain immutable. It does **not** certify functional parity against ten upstream tools or change scientific estimators. The acceptance matrix now contains a conservative state for each feature plus an entrypoint and missing evidence.

Audit statuses: `PARTIAL` = related code or recorded evidence exists but WP12 acceptance is incomplete; `NOT_STARTED` = no WP12-specific executable end-to-end route identified during this source audit, even if a prerequisite/library exists; `BLOCKED` = the necessary external credentials/hardware/route are not evidenced. Neither `CPU_VERIFIED`, `GPU_VERIFIED`, nor `RELEASED` was assigned from unrerun historical evidence.

| Status | Count |
| --- | ---: |
| PARTIAL | 24 |
| NOT_STARTED | 24 |
| BLOCKED | 2 |
| CPU_VERIFIED / GPU_VERIFIED / RELEASED | 0 |
| **Total** | **50** |

See [machine-readable ledger](12_capability_acceptance_matrix.csv). These counts describe the **WP12 comprehensive capability requirements**, not the functionality of prior SAEBase releases.

## Repository state and prior work

- Current audited `main`: `f9c1f4e`. No open PRs were found at the audit snapshot. Prior PRs 43–49 are closed, but closed is not a sufficient substitute for checking merge ancestry. The Work Plans 01–11, both Work Plan 11 files, and historical Stage D paths remain in place.
- GitHub lists releases `v0.2.0`, `v0.3.0`, `v0.4.0` and separate Stage D raw/evidence releases. Current `pyproject.toml` is v0.4.0.
- The latest observed `CI` workflow on the baseline SHA completed successfully (Actions run `39709423555`). Other contemporaneous release-dispatch runs showed both failures and later successes; do not collapse those into a global deployment claim. The aggregated commit-status API returned no separate status entries.
- `.github/workflows/ci.yml` defines Python 3.11/3.12 offline tests, an analysis-extra install/import check, distribution builds, Research Lab API/JS checks, CPU Docker smoke, Localize and quality jobs. These are **GitHub-reported** checks, not tests independently executed in this audit environment.
- `.github/workflows/m3-saebench-native.yml` is manual and runs original pinned SAEBench Core/Sparse Probing. This prior upstream workflow is not the same as a unified public native-SAEBench runner inside SAEBase.

## Confirmed reusable foundation

| Area | Current source and evidence | Reuse decision |
| --- | --- | --- |
| Model + hooks | `saebase/api/model.py`, `models/hooked_lm_common.py`, `models/hf_hooks_backend.py` | Retain; audit TransformerLens legacy `HookedTransformer` use and design TL4 bridge before extending |
| SAEs + registry | `saebase/sae/loading.py`, `saebase/sae/registry.py`, `saebase/sae/sparsify_adapter.py` | Retain existing real SAE registry and native checkpoint adapter; do not claim training/transcoder support |
| Integrity and caching | `saebase/store/identity.py`, `saebase/store/collect.py`, `saebase/api/features.py` | Retain hash-bound caching and residual-preserving edits |
| Interventions and generation | `saebase/interventions/engine.py`, `saebase/interventions/steering.py`, `saebase/generate/nnsight_worker.py` | Preserve existing distinct HF/KV/vLLM/NNsight execution protocols; no universal equivalence assumption |
| Benchmarks and research | `saebase/benchmarks/runner.py`, `saebase/benchmarks/adapters/saebench.py`, `saebase/integrations/{sar,rss,localization}.py` | Preserve SAR/RSS/Localize native scientific semantics; SAEBench metric import is **not** execution |
| Research Lab | `apps/research-lab/backend/api.py`, `backend/feature_store.py`, `frontend/app.js` | Reuse read-only results, feature indices, source-approved examples, datasets, analysis, preflight and reports |
| Prior GPU records | `docs/verified-model-matrix.md`, `docs/performance/workplan11_final_results.md` | Preserve 18 configuration / 48-site Stage D records, WP11 protocol-specific timings, and their limitations |

## Research Lab as actually inspectable

Source: `apps/research-lab/backend/api.py`, `apps/research-lab/frontend/app.js`, `apps/research-lab/README.md`, `docs/research-lab/cpu-deployment.md`.

- Recorded feature index and `GET /api/features` perform identity/label filtering over approved stored entries. They do **not** implement fresh prompt inference, per-token top-k queries, candidate auto-interpretation, or a complete feature dashboard.
- `POST /api/jobs/preflight` validates requested operation/site and explicitly returns `submitted: false`, `ran_model: false`. `POST /api/jobs` intentionally returns HTTP 503.
- Existing CPU Lab displays verified historical 48-site/64-group evidence when the data is bundled. The newer reported 500-prompt raw archive is not part of the public Docker image. Rights and source-access review are outstanding.
- The repository includes a Hugging Face Docker Space recipe, but `docs/research-lab/cpu-deployment.md` explicitly states that this is **not** evidence of the currently deployed Space. The actual live Space could not be independently inspected from this execution environment; its deployed revision and user-path verification remain **UNVERIFIED**. Do not equate GitHub's source tree with production deployment.

## Important compatibility / correctness blockers

1. `models/hooked_lm_common.py` imports legacy `HookedTransformer` while `pyproject.toml` allows `transformer-lens>=2.0` and `sae-lens>=4.0` without upper limits. Current package combinations require pinned-environment inspection and a TL4 compatibility decision. Successful prior imports do **not** prove new API parity.
2. `saebase/benchmarks/adapters/saebench.py` is an intentionally strict Core/Sparse Probing JSON importer; no automatic original upstream runner exists at the user entrypoint.
3. `saebase/sae/sparsify_adapter.py` loads native TopK checkpoints and expressly rejects transcoders/skip connections. It is not Sparsify training.
4. The NNsight worker contains narrowly scoped NNsight/vLLM operations, not a general gradient-capable trace service or NDIF remote integration.
5. No complete `circuit-tracer` graph-construction, graph interaction or cross-layer transcoder integration was identified in `main`.
6. `F50` requires **two or more real GPUs** and cannot pass from a single-GPU machine; `F20` requires actual authorized NDIF deployment/access and an executed trace.
7. The repository's `docs/research-lab/release-acceptance.md` contains unclosed raw-data rights, independent reproduction, Space deployment and external research-user gates. Preserve those gates.

## Evidence limitations

This environment cannot clone `github.com` directly due DNS/network restrictions and lacks the project's installed `hydra-core`, `transformer-lens` and `sae-lens`. Therefore no fresh clean install, `pip check`, local full `pytest`, Docker smoke, native upstream comparison, GPU parity, or live-Space acceptance was performed. GitHub API inspection is the evidence source; source inspection is not an independently executed test. No C1/C2/C3 computation was authorized or initiated.

The matrix `cpu_evidence`, `gpu_evidence`, `native_parity_evidence`, `artifact_link` and `reviewer` cells are deliberately empty until actual proof is attached. Links under `open_sae_entrypoint` are **code/document paths**, not assurances of a public runnable API. `upstream_reference` values are reference documentation starting points, not a completed frozen ten-tool product study.

## P0 gate checklist

| Gate | State | Required next step |
| --- | --- | --- |
| Source inventory, 50 classified items, preserved historical files | DONE (review pending) | Independent reviewer confirms every classification |
| Main/PR/release/CI metadata snapshot | DONE | Recheck merge ancestry and CI after PR merge |
| Deployed Research Lab real revision and browser journeys | NOT VERIFIED | Inspect actual HF Space and routes with authorized browser or live network |
| Clean core install / CPU test suite / Research Lab Docker smoke | NOT RUN HERE | Execute pinned reproducible clean environment and record logs |
| Frozen 10 upstream UI journeys, licenses and source commits | NOT STARTED | WP12-01 dependency and rights freeze |
| Frozen v1 model/SAE/input/intervention/schema contracts | NOT STARTED | WP12-01 |
| Exact TransformerLens 4 and SAELens/NNsight worker compatibility | NOT VERIFIED | WP12-01, followed by native parity |
| New two-model/two-site real GPU smoke | NOT RUN | GPU-00 only after CPU preflight passes |
| **Overall P0 gate** | **NOT PASSED** | Complete open checks before P1 |

## First handoff order

1. Independently inspect the current HF Research Lab deployment and confirm its source revision, API health, feature and search functionality, and read-only execution boundary; capture dated evidence.
2. In a clean Python 3.11/3.12 environment, install the core + isolated extras; run existing offline test suite, `apps/research-lab/tests/test_platform.py`, `node --check`, CPU Docker / API smoke, and `pip check`. Capture raw outputs.
3. Review each ledger row against native-upstream expectations, freeze the actual versions, model/SAE/hash/tokenizer/site metadata, supported operation flags, sample manifest and numerical tolerances.
4. Only then hand GPU-00 a C1 two-model/two-site × 8–16 prompt experiment specification, run the real source-bound parity check, and submit results for independent review.

No upstream code, historical result, UI, scientific estimator or inference endpoint was modified by this audit.
