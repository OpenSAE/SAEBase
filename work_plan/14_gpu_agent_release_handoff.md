# Work Plan 14 — GPU Agent Release Handoff

> **Superseded by [WP15 GPU Agent runbook](15_gpu_agent_completion_runbook.md).** The old TRY_ONCE / HIDE / DEFER strategy is historical; install missing upstream dependencies, prepare real assets and complete G1–G6.


Date: 2026-10-22. Source branch: `docs/wp13-real-world-integration` (WP12 P0/P1/P2/P3–P8 stacked below it). Read [release plan](14_ship_first_release_plan.md). **Goal: working release features, not independent 50-feature certification.**

## Working constraints

Inspect current branch, dirty workspace, hardware, existing model and SAE caches first. Reuse exact working LIVE-01/02/03 private runs; do not rerun GPU-00, Stage D, the historical 48-site suite or a 500-prompt dataset. Keep real original source artifacts on GPU host. Send small plain JSON/Markdown receipts only. Do not send huge archives, weights, tokens, raw logits, private traces, SQLite, prompts or secrets. Don't request external subscriptions or rent new GPUs.

For each job: work on a clean worktree based on the WP13 branch; run the smallest genuine example using already available assets; repair one straightforward incompatibility if necessary; stop rather than looping. Report **SHIP / FIX_ONCE / HIDE / DEFER**, commands executed, source SHA, real output, resource figures, artifact path, and exact blocker. Never quietly return synthetic or replayed results as live. Ship a smaller set of working functionality when optional jobs cannot run.

## G1 P0 — New input genuinely works

First perform a real model/SAE inspect on **one new owner-created prompt** through the local `saebase.explore.capture_prompt_trace` or `scripts/wp12_p2_capture.py` route; check exact model/SAE/site and actual token-level nonzero features. Then try the existing private worker API/queue flow from [WP13 handoff](13_gpu_agent_real_environment_handoff.md) LIVE-08 for **two owner-only prompts** and authenticate retrieval.

- Verify baseline/no-op/ablation (and scale if supported) on an actually active token feature, and preserve one meaningful real numeric/logit result.
- If the API queue is broken and expensive to fix, leave it **DEFER** and ship a documented direct Python/CLI run. The direct local fresh-input workflow matters more than hosting.
- Never expose private route/tokens publicly. Report user-visible output and failures, measured runtime and peak VRAM.

## G2 P1 — One generation-time continuation

Reuse supported live KV/full-prefix backend and cached checkpoint. On one short original prompt with one observed active SAE feature, generate a few tokens for clean/no-op/edited paths. Show actual decoded continuations and clearly identify whether edited generation differs. Report real generation protocol and resource use. This is separate from LIVE-03 **teacher-forced logits**. If backend is unavailable, HIDE generation demo and retain the accurate recorded logit intervention.

## G3 P1 — Native benchmark in one real run

Use existing original pinned SAEBench integration `saebase.wp12.benchmarks.native_pilot` / `custom_saebench` with compatible dataset and checkpoint. Run just one valid Core or Sparse Probing test. Return source + metric IDs, measured score, saved native output path, runtime, and whether Research Lab actually imports/displays it. Only build a two-SAE comparison with truly compatible sources. If dependency/dataset access fails, DEFER live benchmark; existing recorded benchmark results remain available.

## G4 P1 — One circuit graph

Try original `run_native_circuit_tracer` on one **actually compatible** cached model/representation and short prompt. Import via `import_native_graph_file`. Report real graph nodes/edges and browser-friendly file. If possible demonstrate a supported node intervention, but do not block browser graph inspection on an incompatible intervention. If original graph computation is blocked, the CPU agent may only publish an already authentic, permitted recorded graph; HIDE any 'run live circuit' button.

## G5 P2 — Tiny training with reload

Only if small native trainer, model and dataset already work. Use `saebase.wp12.training.TrainPlan`, `run_training`, `register_checkpoint`; one framework (SAELens **or** Sparsify) is sufficient. Train a tiny valid SAE, reload in SAEBase and encode/decode. Test resume only if existing native state makes it effortless. If this becomes a new research project, DEFER training completely. No F50 distributed GPUs.

## G6 P2 — Local auto-interpretation

Only if genuine distinct positive/negative discovery and held-out activation contexts are already available and a small local instruct model fits. Use `scripts/wp12_autointerpret.py`, return candidate plus held-out measurement with `approved=false`. No model-authored interpretation automatically published. Otherwise HIDE.

## Small release receipt

Produce `/workspace/wp14/release_gpu_receipt.json` (or equivalent chosen existing private output root), plus a concise `release_gpu_receipt.md`. Suggested JSON:

```json
{
  "source_sha": "actual branch HEAD",
  "device": "actual GPU model",
  "jobs": [
    {
      "id": "G1",
      "decision": "SHIP",
      "model": "exact model and revision",
      "sae": "exact SAE and revision",
      "site": "actual layer and hook",
      "command": "actual command run, secrets removed",
      "result": "one concise genuine observed result",
      "seconds": 0,
      "peak_vram_bytes": 0,
      "artifact_path_private": "actual local file path",
      "ui_handoff": "small safe measured output or source pointer",
      "blocker": null
    }
  ]
}
```

Include G1–G6 with actual decision values; replace placeholders, do not manufacture numeric zeros or a SHIP decision for unrun jobs. Include at most a few kilobytes of source-linked redacted summaries for a reviewer. Keep screenshot paths and exact reproduction commands for supported features.

## CPU Integration Agent parallel work (not GPU Agent responsibility)

Stage approved existing LIVE-02 feature records; stage two real LIVE-03 logit comparison cards; ensure actual Prompt → Token → Feature → Intervention clicks; polish public Lab navigation; clean Python quick start; test populated Docker staging and restart; preserve existing HF Space revision. User chooses actual publication timing. Work is accepted by feature usefulness, not P8 raw-array audit.

**Stop condition:** once G1 works, send the first receipt immediately; P1/P2 should not hold up integration or release. If an optional job is blocked, state HIDE/DEFER and move on.
