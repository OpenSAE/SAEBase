# Engineering Plan Archive

## Work Plan 14 — Active release-first plan (takes precedence for launch)

**Current priority: ship useful, working SAEBase features.** This supersedes WP12's full 50-feature certification and WP13's eleven-job completion list as **launch requirements**. Existing work plans and data remain historical references, not prerequisites. Optional missing capabilities should be hidden/deferred rather than block a usable release.

- [**14 — Ship-first release plan**](14_ship_first_release_plan.md). Required real Feature Explorer → Prompt/Token → Feature → measured Intervention, working Python quick start, polished public CPU Lab, staged deploy/restart, GO/GO_WITH_CUT criteria.
- [**14 — GPU Agent release handoff**](14_gpu_agent_release_handoff.md). Bounded G1–G6 genuine smoke runs, exact feature-specific SHIP/FIX_ONCE/HIDE/DEFER reports, existing GPU asset reuse and no raw archive transfers.
- [**14 — Release scope**](14_release_scope.csv). Launch-required, optional and explicitly cut features with a user-visible minimum demo and owner.
- [**14 — GPU receipt review**](14_gpu_receipt_review_20261022.md). Agent-reported G1 accepted for local live; optional G2–G6 hidden/deferred; public staging remains the next release gate.


This folder holds plans and execution records used while building SAEBase. Start with the [project README](../README.md) and [documentation](../docs/index.md) for current installation and research tasks.

- Early plans: model/SAE acceptance, GPU coverage and Research Lab rollout.
- Performance plans: model-forward optimization, checkpoint integrity, suffix replay and KV-cached steering.
- Final measured performance, including Qwen3-1.7B intervention and generated-token results, is reported in [Work Plan 11 results](../docs/performance/workplan11_final_results.md).

The two Work Plan 11 files reflect an original broad plan and the later GPU execution handoff; they are historical records, not competing user-facing installation instructions. Existing paths remain intact so external references and experimental records do not break.

## Work Plan 12 — Unified OpenSAE research infrastructure

The new long-term master blueprint integrates the **50 functional capabilities of 10 established interpretability projects** through a single SAEBase API and Research Lab, retaining all historical plans and experimental evidence.

- [**12 — Master blueprint**](12_unified_interpretability_master_blueprint.md). Nine phases P0–P8, upstream feature-to-UI mapping, API architecture, 50 exact acceptance tests, local/GPU split, C0–C3 compute tiers, scientific boundaries and release policy.
- [**12 — GPU agent handoff**](12_gpu_agent_handoff_protocol.md). Exact phased GPU/CPU responsibilities, preflight, run artifacts, rate measurement, stop rules and independent signoff.
- [**12 — 50-capability ledger**](12_capability_acceptance_matrix.csv). Core implementation and final release are tracked separately: 48 source-level PASS, 0 core PARTIAL, 2 external BLOCKED, with all independent P8 verification pending. The historical WP12-00 audit remains a dated baseline.
- [**12 — P0 source audit**](12_p0_source_inventory_20261021.md). Current code, Research Lab, CI/release and compatibility inventory; P0 gate remains open.
- [**12 — GPU-00 runbook**](12_gpu00_p0_parity_execution.md). First bounded C1 real-model numerical parity test with environment preflight, exact model/SAE/site choices, operation definitions, tolerances, resource stop rules and immutable handoff outputs.
- [**12 — GPU-00 frozen input pairs**](12_gpu00_prompt_pairs.json). Eight small original paired prompts, site list and tokenization acceptance policy.
- [**12 — GPU-00 agent report and independent review**](12_gpu00_agent_report_20261022.md). Reports four passing legacy fp32 model/SAE sites, preserved failures and resource counts, while explicitly leaving raw-artifact verification, TL4 compatibility and overall P0 acceptance open.
- [**12 — P1 implementation and short smoke**](12_p1_model_backend_implementation.md). Version-aware TransformerLens loading, missing-hook protection, residual component inspection, local NNsight tracing, Gemma Scope 2 metadata/variants, lightweight tests and limited real-weight smoke.
- [**12 — P2 feature explorer**](12_p2_feature_explorer.md). Real-only feature dashboards, explanation/recorded-prompt/token searches, local sparse prompt capture, correlations, hypothesis provenance and portable offline HTML; no fake online inference.
- [**12 — P3–P8 consolidated implementation**](12_p3_p8_consolidated_implementation.md). Typed interventions, bounded original SAEBench, native training/restart, Circuit Tracer graph conversion/UI, private worker and independent release checks. Scientific and public release acceptance still remains open.
- [**12 — Core functionality verdicts**](12_core_function_assessment.md). A strict 50-item implementation assessment: **48 core PASS, 0 PARTIAL, 2 BLOCKED**, with per-item executable criteria, source/test references and specific remaining gaps. Independent P8 scientific validation is separate.

Work Plan 12 now has developed core source routes but is **not** a claim that all functions have been executed with real official checkpoints or deployed. Follow older WP04–WP11 experimental protocols and accepted findings.

## Work Plan 13 — Real-environment integration and deployment

Turn WP12's source-level implementations into actual runnable research workflows and a useful deployed Research Lab, prioritizing genuine model/prompt experience over repeated formal numerical certification.

- [**13 — Real-environment integration and deployment**](13_real_environment_integration_and_deployment.md). Architecture, 10-prompt real-feature browser, genuine interventions, native training/benchmarks/circuits, private local GPU queue, source-reviewed dataset publication, HF staging/rollout, usability and recovery.
- [**13 — Local GPU Agent execution handoff**](13_gpu_agent_real_environment_handoff.md). Exact LIVE-00 → LIVE-10 task sequence, commands, isolated library environments, bounds, return artifacts and Integration/GPU agent repair ownership.
- [**13 — Machine-readable LIVE task queue**](13_live_execution_matrix.csv). Eleven execution tasks with dependent prerequisites, resource caps, measured success definitions, minimal handoffs and honest external blockers.
- [**13 — Ten candidate demo prompts**](13_demo_prompt_candidates.json). Original engineering text candidates with fixed IDs and 6/4 discovery/evaluation split, all explicitly unreviewed; real model activations and release approval must be added separately.
- [**13 — LIVE-00/01/02 agent report**](13_live_00_02_report_20261022.json). Trusted reported runtime results and exact GPU/database identities, kept separate from source verification and public approval.
- [**13 — Small LIVE-02 handoff and staging publication**](13_live02_small_handoff_and_publication.md). Metadata-only GPU report, source-hash-bound local review, compact Research Lab staging bundle; no huge ZIP transfers.
- [**13 — LIVE-03 measured intervention integration**](13_live03_review_and_gallery.md). PR #55/#56 merged; L5 F506 and L9 F95 no-op/ablation/scale with separate edited-token and final-token logits, optional approved CPU gallery and remaining private job work.

WP13 changes **operational readiness and deployment**, not the untouched WP12 P8 scientific release verdicts. Start with one complete real Prompt → Feature → Intervention workflow, publish a useful reviewed CPU Lab, then extend to native training and circuits.
