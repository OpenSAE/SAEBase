# Work Plan 12 — GPU Agent Execution Handoff

This file is an executable **handoff protocol**, not authorization to run expensive jobs. Read [master blueprint](12_unified_interpretability_master_blueprint.md) and [50-feature ledger](12_capability_acceptance_matrix.csv). Begin at P0; preserve historical Work Plans 01–11 and existing evidence.

**GPU-00 execution update:** [Agent-reported outcome and independent review queue](12_gpu00_agent_report_20261022.md). The agent reports 4/4 real-site legacy TransformerLens 3.9 / SAELens 6.54 fp32 passes following the [bounded P0 real-weight parity runbook](12_gpu00_p0_parity_execution.md) with [eight fixed prompt pairs](12_gpu00_prompt_pairs.json). **The raw evidence has not yet been transferred or independently reviewed. Do not repeat C1, promote capability statuses, claim TL4 compatibility, or automatically launch GPU-01.** The [P0 source audit](12_p0_source_inventory_20261021.md) still has unmet CPU/deployment gates.

## Assignment boundaries

**Local strategy/CPU agent**
- Audit existing code, PRs, dependency versions, model/SAE registry and Research Lab's real deployed source.
- Build typed experiment contracts and check unknowns with explicit errors. Develop visual navigation, feature indexing, data and license audit, reports and unit tests.
- Choose exact models/SAEs/sites, source groups, held-out controls, tasks, budget and stop conditions before GPU execution.
- Analyze GPU raw results independently, reject unjustified equivalence and update the CSV ledger with independently verified evidence.
- Never replace original SAR/RSS/Input Localization estimators or historical measured assets.

**GPU agent**
- Check actual allocated GPU/VRAM, RAM, available disk, drivers, conda/pip environment and current processes first; do not infer availability from prior machine descriptions.
- Run native upstream references and matching OpenSAE operations against exact pinned models/SAEs, and write unrounded outputs plus source hashes.
- Prove model-specific logits/activation/feature code, identity and edited-run parity with a predeclared tolerance.
- Execute tiny pilots first, report timings/peak resources, and stop on mismatch or resource/budget violations.
- Maintain resumable immutable run directories outside tracked code and upload only rights-approved summaries/artifacts.

## The four compute tiers

| Tier | Work | Execution limit |
|---|---|---|
| C0 | No new model run: manifests, reports, UI, tests, interactive saved-feature/circuit exploration, existing licensed evidence | Local CPU by default |
| C1 | One real smoke/pilot: 8–32 prompts, selected 1–2 small models/sites, short generation, tiny trainer or graph | One GPU after CPU preflight; mandatory for any claimed live model operation |
| C2 | 2–6 sites, up to 100–500 groups, small set of paired interventions, native benchmark subset | Only after C1 numerics match and justified need; logged quota |
| C3 | 48-site suite, full evaluation/training, many gradients/graphs, distributed 2+ GPUs | Explicit approval including compute/storage/time budget and recovery; no automatic execution |

If only a single GPU is available, F50 real distributed training remains blocked. If remote NDIF/model deployment access is absent, F20 is blocked for that route.

## Prescribed task queue and gates

| Job ID | Phase | Local CPU preparation | GPU run required | Stop / success |
|---|---|---|---|---|
| GPU-00 | P0 | Follow [GPU-00 runbook](12_gpu00_p0_parity_execution.md): exact packages, registered A1/A2/B1/B2 sites, eight clean/corrupt groups, frozen per-model token IDs and C1 preflight | Four selected sites across two small models, 8 groups / 16 variants per model; genuine native-vs-SAEBase baseline/no-op/patch/encode/edit/cache measurements | Exact identity + source parity; mismatch stops expansion; preserve raw evidence and all failures |
| GPU-01 | P1 | Implement validated TL4 bridge, NNsight adapter, Gemma Scope asset registration | Official small SAE load/encode/decode, native comparisons, patch clean/corrupt | No adapter promotion without exact hash/shape/tolerance check |
| GPU-02 | P2 | Create real feature cards/search over existing archive | 8–32 custom prompts, target features/activations; limited interpretation evaluation | Search results match token-level sources; generated explanations have provenance |
| GPU-03 | P3 | Typed InterventionSpec; visual before/after workflow | No-op, ablate, scale, compose, generation-step edit; compare full-prefix baseline | No unsupported equivalent silently substituted |
| GPU-04 | P4 | Pin native SAEBench and metric schema; source/split/rights review | Core + Sparse Probing small real run; a valid two-SAE comparison | Original native outputs preserved, fair comparison and paired controls pass |
| GPU-05 | P5 | Implement config validation, checkpoint import, atomic recovery tests | Native SAELens short train + resume; Sparsify TopK and transcoder; optional moderate | Fresh loaded checkpoints reproduce native encoding; 2+ GPUs needed for F50 |
| GPU-06 | P6 | Graph importer/renderer using official fixture; typed transcoder registry | One native compatible attribution graph and node intervention; one official cross-layer asset if feasible | Node/edge/error hashes and intervention target match source |
| GPU-07 | P7 | Auth+queue, privacy and network failure tests | An authorized actual remote trace and one worker interruption/retry | Missing credentials → BLOCKED, no simulated remote success |
| GPU-08 | P8 | Independent verifier reads complete ledger and artifacts | Only missing source-bound cases; rerun frozen reference sample | Feature acceptance requires independent signoff and honest supported scope |

## Preflight JSON minimum

```json
{
  "run_id": "wp12-p1-smoke-example",
  "phase": "P1",
  "capability_ids": ["F06", "F07", "F11"],
  "model_repo": "<real exact model>",
  "model_revision": "<commit SHA>",
  "representation_repo": "<real exact SAE/transcoder>",
  "representation_revision": "<commit SHA>",
  "site": "<real backend hook>",
  "dataset_id": "<licensed source ID>",
  "dataset_revision": "<exact version>",
  "prompt_group_ids": ["<frozen ID>"],
  "native_tool_commit": "<git commit>",
  "open_sae_commit": "<git commit>",
  "backend": "<backend & version>",
  "device": "<actual hardware>",
  "precision": "<dtype & quantization>",
  "operation": "<exact type & target>",
  "pass_fail_tolerance": "<frozen numeric protocol>",
  "estimated_forward_count": 0,
  "estimated_storage_bytes": 0,
  "max_wall_seconds": 0,
  "max_disk_bytes": 0,
  "compute_tier": "C1",
  "rights_review": "pending",
  "operator_approval_for_c3": false
}
```

This is an example schema, not a runnable job. Placeholders must be replaced and preflight accepted before launch. Never leave unknown tokens/weights or an arbitrary 0 budget in an actual approved execution.

## Required artifact structure

```text
runs/wp12/Px/<run-id>/
  manifest.json
  preflight.json
  native_reference/
  opensae_results/
  parity.json
  resource_usage.json
  failures.json
  checksums.sha256
  summary.md
```

`native_reference` contains untouched original-tool raw outputs or licensed hash-verified pointers. `opensae_results` preserves raw numerical outputs; do not generate only rounded charts. Private input/source data never goes to a public repository by default.

## Required phase completion report

For each phase, report:
1. Stable capability IDs, status **before/after**, exact Git commits and tests.
2. User actions available on web / Python / CLI, including what remains recorded-only or backend restricted.
3. Upstream native comparison and numerical tolerance failures, preserving negative controls.
4. Runtime/peak VRAM, RAM, disk and observed throughput for each real run.
5. Evidence manifest and screenshots for shipped UX.
6. Open problems and next smallest GPU job if necessary.
7. Independent CPU-reviewer signoff, then code review / merge. No self-certified full phase success.

## Fast stop rules

- Any mismatched model/SAE/hook/tokenizer or inconsistent raw code positions: stop and correct identities before running more.
- Missing or ambiguous license, pretrained access, remote credentials or publication rights: record BLOCKED and avoid workaround.
- Unvalidated TransformerLens 4, NNsight or pyvene behavior: pin and repair adapter rather than change protocol silently.
- GPU OOM: log peak, shrink microbatch or choose a documented smaller compatible model; do not silently omit failed cases.
- No approved large-run budget: stop at C1/C2 and deliver valid limited-scope evidence.
- Single GPU cannot validate multi-GPU. Graph rendering cannot validate attribution computation. Imported benchmark JSON cannot validate a benchmark runner.
