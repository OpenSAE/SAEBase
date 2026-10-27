# WP15 — Final Release Candidate Handoff (after CPU/CI integration)

Date: 2026-10-22. **This is the current remaining-work handoff, not another GPU research plan.**

GPU follow-up A–E is recorded in [15_gpu_asset_review.md](15_gpu_asset_review.md). That note is for human review. It does not update production `main` or the Hugging Face Space. The raw trace database stays on the GPU host.  
Current release-candidate branch: `feat/wp15-six-workflows` ([PR #57](https://github.com/OpenSAE/SAEBase/pull/57)), with the pre-release integration ([PR #58](https://github.com/OpenSAE/SAEBase/pull/58)) already merged as commit `ae655959506f072b765ccb386372a7aacb2312a0`. PR #57 targets the WP13 development branch.  
Production GitHub `main` and Hugging Face `OpenSAE/SAEBase` Space **must remain unchanged until the actual owner-authorized release**.

## 1. Status and completed work

- G1 fresh real Gemma-3-270M feature capture, teacher-forced ablation/scale, and private inspect queue: GPU Agent-reported working. WP13 LIVE-02 source traces and LIVE-03 measured intervention cards exist in the private original run and small metadata reports. **One public authorized Prompt → Token → Feature trace is not yet staged.**
- G2 16-token real autoregressive steering: two recorded prompts, baseline/no-op/ablation/scale, measured probability changes and one first-generated-token branch change. Private decoded strings were not published. Source-linked numerical cards appear in the WP15 browser.
- G3 original pinned SAEBench Core/Sparse-Probing: native numerical record available for one Pythia-70M SAE. No broad cross-SAE quality claim.
- G4 original circuit-tracer record: Gemma-2-2B and matching transcoder, 155 nodes and 5,915 original edges. Bounded browser graph uses 748 edges. This is a recorded native graph, not a remote GPU service.
- G5 native Sparsify 50-step run: real checkpoint hash, `reloaded: true`, finite encode/reconstruction and metric record. Do not claim a strong newly trained SAE from 50 steps.
- G6 Qwen2.5 0.5B real local interpretation: unapproved candidate, held-out activation-prediction score **0.50** with only three positive and one negative held-out example. **Not a useful public explanation yet.**
- CI fixes: Ruff formatting from PR #57; normal Python/build checks; actual CPU Docker now loads all five WP15 summary cards and confirms they survive restart. Added staged-data override from `$OPENSAE_LAB_DATA/wp15/`, plus guarded optional display of reviewed G2 text and G6 explanation text. No unreviewed prompt, continuation or hypothesis can pass these display rules.
- UI clarity: probability columns and actual first-token generation branch are labeled, weak G6 diagnostic is explicitly identified, and the Circuit label does not imply review is complete.

**Do not re-run G3, G4 or G5 merely to satisfy paperwork; prioritize the missing human-visible product artifacts.**

## 2. Remaining actions that require GPU Agent / operator-owned source assets

### P0 A — Publish one authentic G1 interactive trace

The Integration Agent cannot reconstruct exact original token codes from tiny GitHub summaries. They are on the GPU host.

1. Locate the original WP13 LIVE-02 `private_raw` directory, currently identified in the LIVE matrix as `/workspace/wp13/private/runs/LIVE-02-20261022T1912Z-903d837/private_raw`. Check existence on the actual host, not from the README.
2. Using `saebase/tools/wp13_prepare_public_bundle.py`, produce a fresh small report plus a **review template**. Human/operator must select at least one **owner-authored, redistributable** prompt and mark the corresponding source/rights checks only after reading it. Do not set every review flag to true just for deployment.
3. Use `--publish-reviewed --review-file <actual reviewed JSON> --release-root <new immutable staging dir>` and stage the output into the actual `OPENSAE_LAB_DATA` used by the CPU Docker. Preserve source model/SAE/tokenizer hashes and all nonzero codes. Do not upload raw SQLite, private prompts or activations to GitHub.
4. Confirm one browser click path: listed original prompt → token → active feature → feature detail → back → appropriate *same-run* recorded record. Do not join G1 F1627 to LIVE-03 F506/F95 as if they are the same trace. Intervene should separately show LIVE-03 source-bound cards.

**Required handoff:** path of the small approved release bundle, number of reviewed traces, one screenshot or screencast, actual API trace ID and source SHA, plus operator review attribution. No massive raw ZIP.

### P0 B — Complete owner-private **edit** queue smoke

Earlier G1 private queue proved two inspect jobs and 401 for wrong token. It did not establish queue-based edits. Execute one source-identical private `ablate` or `scale` job using an actually active feature. Verify authenticated result retrieval, changed teacher-forced logits, and wrong-role 401. Report only redacted numeric receipt. If the direct Python API is the release-supported operation, label the queue operator-only and do not imply public hosted inference.

### P0 C — Prepare reviewed *actual decoded generation* demo

The current public `generation.json` contains safe measurements but intentionally omits source prompt and continuation strings. PR #58 now supports an optional reviewed record in `$OPENSAE_LAB_DATA/wp15/generation.json`.

Only after reviewing **both** original, owner-authorized prompts, add top-level
`rights_reviewed: true`, `public_prompt: true`, `public_continuations_reviewed: true` and, for **each** `samples[]`, a strictly bounded `public_example` with exactly:
```json
{
  "prompt_text": "exact original prompt text whose UTF-8 SHA256 equals sample.prompt_sha256",
  "baseline_continuation": "exact decoded original baseline text",
  "edited_continuation": "exact decoded original ablation text"
}
```
These values must come from the already executed G2 raw/private output, not hand-written approximations. The backend fails closed if text doesn't match the source hash, if review is incomplete, or if extra private fields are present. Review the rendered page to ensure the displayed edited continuation really belongs to that sample and condition.

**Required handoff:** staged rights-reviewed example metadata and a screenshot of the actual generation UI; keep unrelated private prompts private.

### P0 D — Improve G6 explanation before making semantic claims

The recorded 0.50 held-out result with only four held-out examples is **not** a public feature hypothesis. Do not publish it as an interpreted feature.

GPU Agent should select a genuinely better-supported feature or obtain more distinct owner-authored positive and negative prompts, construct disjoint discovery/held-out groups (aim for at least ~8 of each polarity in held-out when feasible), run the real local instruct model and inspect errors. Publish a meaningful short hypothesis **only if a human reviews both the text and diagnostic results**. A candidate may still be left unpublished if no meaningful one is demonstrated.

PR #58's `wp15_records` supports a reviewed override `$OPENSAE_LAB_DATA/wp15/explanation.json`: use real original generated description, `approved:true`, `human_reviewed:true`, `rights_reviewed:true`, `private_unpublished:false`, `description_published:true`. All other numerical/source fields remain from the real run. Do not artificially improve the numerical score.

**Required handoff:** source-linked reviewed candidate, prompt-group counts, score, real model and feature IDs, staging page screenshot. If still low quality, the feature page must say “not interpreted”; local auto-interpret CLI can remain usable.

### P0 E — Public data and UI scope confirmation

Existing WP15 raw metadata cards are staged for preview but have `rights_reviewed:false`; this is not a completed final public-data approval. Review which aggregate metrics/graph records can be redistributed from the original model/third-party sources, without copying private prompt text. Update **only the true approval flags** in the staged release bundle. Stage the generated circuit graph, benchmark and checkpoint **summaries**, not original weights or unreviewed raw datasets.

## 3. Remaining deploy operation after assets are approved

Integration Agent should:
1. Prepare one immutable small CPU staging bundle with `explore/`, `features/`, `published_wp12/`, and optional `wp15/` from the reviewed sources; verify exact expected backend directory layout. For WP15 cards, staged `$OPENSAE_LAB_DATA/wp15/` overrides the checked-in sanitized previews. Do not overwrite the original source-run directories.
2. Build and run `docker build -f saebase/research_lab/Dockerfile -t opensae-lab:wp15 .`, then run with `-v <approved_bundle>:/workspace/runs/research-lab:ro`. Confirm G1–G6 recorded journeys, at least one nonempty approved prompt trace, a circuit graph, metrics, training evidence, generation display and reviewed/accurate interpretation status. Verify reloading and browser navigation.
3. Run the full GitHub CI on the latest PR head. Package/format/CPU Docker pass is necessary; it is not proof of live HF staging. Capture the actual staging revision, public staging URL and a browser screenshot.
4. Only **after** a successful staging walkthrough, queue PR merges in dependency order (#50→#51→#52→#53→#54→#57, with PR #58 changes included). Capture the current production Space revision as a rollback baseline. Do not touch production until owner approves the final release step.

## 4. Decision criteria

**Release candidate ready** when the actual browser displays one approved trace, real measured intervention, one source-matched decoded steering example, a genuine native benchmark, a real graph, a real trained-checkpoint record, and either a genuinely reviewed useful explanation or a clearly labeled local explanation tool with its diagnostic limitations. Source-backed Python/CLI commands must be correct; public CPU pages never claim on-demand inference.

**Do not block for** 50-feature P8 certification, rerunning old GPU archives, remote NDIF, multi-GPU training, or more native runs just for completeness.

**Not yet established here:** permission to publish source texts and candidate explanation, actual owner-private GPU assets on this runtime, and a remotely visited staging Space with reviewed bundle. Those are real dependencies, not code-quality excuses.

**Next handoff to GPU Agent:** perform tasks A–E only, not a new broad experiment suite. Return small approved bundle paths, one improved G6 result if possible, one queue-edit receipt and screenshots; the release integrator can then perform the final staging/release operation.

Links: [WP15 master plan](15_complete_six_workflows_and_release.md), [PR #57](https://github.com/OpenSAE/SAEBase/pull/57), [pre-release fixes PR #58](https://github.com/OpenSAE/SAEBase/pull/58), [WP13 publication script](../../saebase/tools/wp13_prepare_public_bundle.py).
