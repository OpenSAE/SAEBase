# Engineering Plan Archive

This folder holds plans and execution records used while building SAEBase. Start with the [project README](../README.md) and [documentation](../docs/index.md) for current installation and research tasks.

- Early plans: model/SAE acceptance, GPU coverage and Research Lab rollout.
- Performance plans: model-forward optimization, checkpoint integrity, suffix replay and KV-cached steering.
- Final measured performance, including Qwen3-1.7B intervention and generated-token results, is reported in [Work Plan 11 results](../docs/performance/workplan11_final_results.md).

The two Work Plan 11 files reflect an original broad plan and the later GPU execution handoff; they are historical records, not competing user-facing installation instructions. Existing paths remain intact so external references and experimental records do not break.

## Work Plan 12 — Unified OpenSAE research infrastructure

The new long-term master blueprint integrates the **50 functional capabilities of 10 established interpretability projects** through a single SAEBase API and Research Lab, retaining all historical plans and experimental evidence.

- [**12 — Master blueprint**](12_unified_interpretability_master_blueprint.md). Nine phases P0–P8, upstream feature-to-UI mapping, API architecture, 50 exact acceptance tests, local/GPU split, C0–C3 compute tiers, scientific boundaries and release policy.
- [**12 — GPU agent handoff**](12_gpu_agent_handoff_protocol.md). Exact phased GPU/CPU responsibilities, preflight, run artifacts, rate measurement, stop rules and independent signoff.
- [**12 — 50-capability ledger**](12_capability_acceptance_matrix.csv). One row per feature. WP12-00 source audit currently classifies 24 PARTIAL, 24 NOT_STARTED and 2 BLOCKED; independent CPU/native/GPU evidence fields remain empty until independently verified.
- [**12 — P0 source audit**](12_p0_source_inventory_20261021.md). Current code, Research Lab, CI/release and compatibility inventory; P0 gate remains open.
- [**12 — GPU-00 runbook**](12_gpu00_p0_parity_execution.md). First bounded C1 real-model numerical parity test with environment preflight, exact model/SAE/site choices, operation definitions, tolerances, resource stop rules and immutable handoff outputs.
- [**12 — GPU-00 frozen input pairs**](12_gpu00_prompt_pairs.json). Eight small original paired prompts, site list and tokenization acceptance policy.

Work Plan 12 is a *planned expansion*, not a claim that all capabilities have been implemented, publicly deployed or tested on GPU. Follow older WP04–WP11 experimental protocols and accepted findings when implementing the new work.
