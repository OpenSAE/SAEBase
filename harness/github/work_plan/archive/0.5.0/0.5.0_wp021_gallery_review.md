# WP13 LIVE-02/03 Review and Research Lab Integration

Updated: 2026-10-22. Goal: trust small real GPU Agent outputs for product development and turn them into genuinely browsable records without moving original arrays.

## What has been reviewed

- [PR #55](https://github.com/OpenSAE/SAEBase/pull/55) **merged into WP13**. The actual [LIVE-02 inventory](LIVE-02-small-inventory.json) is 8,765 bytes, with 20 source-bound traces, 240 tokens and 16,788 nonzero codes. There are ten prompts × two Gemma-3-270M SAE sites, each with prompt and original private-file SHA256. `reviewed_excerpt_count=0` and `public_release=false`. No original token activations or prompt strings were committed.
- [PR #56](https://github.com/OpenSAE/SAEBase/pull/56) **merged into WP13**. The actual [LIVE-03 summary](LIVE-03-small-summary.json) records 10.252 s and 2,904,854,016 B peak VRAM on the GPU Agent's environment, without full-vocabulary logits.
- Both PRs passed original GitHub CI. Their source runs are accepted as **real agent-reported execution**, not independent raw tensor verification. Historical GPU-00 and LIVE-02 are not rerun. Production Hugging Face Space remains unchanged.

### Exact measured editing claims

The same `WP13-DEMO-03` prompt, with SHA256 `d830ebdaf0340f9e6372843e1910c4b655b4d6f9c142c84ef27b2e4b30eff48f`, was edited at token position **1** (token ID `50429`, displayed as `Paris`), in two different layers of the same Gemma checkpoint. The model SHA256 and layer-specific SAE SHA256 are saved in the original JSON.

| Layer/feature | Operation | Readout at token 1: L2 / max absolute | Readout at token 1 argmax | Final token readout: L2 / max absolute | Final argmax |
| --- | --- | --- | --- | --- | --- |
| L5 F506 | No-op | 0 / 0 | Same | 0 / 0 | Same |
| L5 F506 | Ablation | 596.1752 / 6.1181 | 236764 → 14148 | 77.2575 / 0.8522 | Same (10508) |
| L5 F506 | 2× scaling | 319.0810 / 2.8990 | Same (236764) | 54.7857 / 0.6261 | Same (10508) |
| L9 F95 | No-op | 0 / 0 | Same | 0 / 0 | Same |
| L9 F95 | Ablation | 263.5511 / 2.5850 | 236764 → 14148 | 33.1951 / 0.3603 | Same (10508) |
| L9 F95 | 2× scaling | 240.9247 / 2.2172 | Same (236764) | 29.5774 / 0.3125 | Same (10508) |

**Important scientific wording:** these are **teacher-forced full-vocabulary logits**, not generated token sequences. Their reported argmax at position 1 changes under ablation; the final token's argmax remains `10508`, even though final-position logits change by nonzero L2. The raw L2 aggregates differences across a large vocabulary and cannot be treated as semantic quality improvement, robust causal generalization, or generated-text behavior. The observed feature F506 is *not* yet an independently supported “geography” or “Paris concept” explanation.

## Product integration now available

- `harness/github_work_plan/13_live_execution_matrix.csv` records LIVE-03 **WORKING** for actually measured, teacher-forced edits, while the *private new-prompt service* part stays unfinished.
- `saebase/tools/wp13_stage_live03.py` reads the small existing summary, checks real checkpoint/site hashes, no-op zero changes, expected token positions and signed source operations, and prepares **two source-bound, measured intervention cards**. Default run only validates and reports counts; it does not publish. Explicit `--publish-reviewed --reviewer ... --release-root ...` stages the approved records into the existing immutable Research Lab `published_wp12/interventions` mechanism.
- Research Lab's **Intervene** page now has a recorded-measurements gallery. If staging data are absent, it says so, rather than showing simulated effects. After approved records are locally staged, it shows one gallery item per real SAE site with separate token-1 and final-token no-op/ablation/scale metrics. It never claims text was generated.
- `saebase/research_lab/tests/test_wp13_live03_gallery.py` checks the real committed small summary, rejects invalid native hashes/mislabeled generation/nonzero no-op, stages in isolated test directories, and exercises API reads.

To validate only, without public release:

```bash
python saebase/tools/wp13_stage_live03.py \
  --source harness/github_work_plan/LIVE-03-small-summary.json
```

After a named owner/operator completes the source/rights review, stage **on the CPU deployment build host**, not on the production Space:

```bash
python saebase/tools/wp13_stage_live03.py \
  --source harness/github_work_plan/LIVE-03-small-summary.json \
  --publish-reviewed \
  --reviewer "Approved reviewer" \
  --release-root /private/wp13/staging-lab
```

The actual stage path must also include a valid read-only `OPENSAE_LAB_DATA` root with the historical Stage D evidence. This command **does not itself deploy** a Hugging Face Space. Large logits and private prompts remain on the local GPU host.

## Next useful tasks

1. **LIVE-02 public prompt/feature path**: obtain explicit review of one authored prompt and its source-bound trace first. On the GPU host run `saebase/tools/wp13_prepare_public_bundle.py` against the existing `private_raw` after review and build one genuine browsing bundle; then expand if suitable to 20. Avoid uploading SQLite in chat.
2. **LIVE-03 staging gallery**: verify local API returns the two recorded intervention cards and that the first-click UI shows the correct *token 1 vs final token* comparison. No new GPU compute required.
3. **LIVE-08 actual private new-prompt job**: the current LIVE-03 run has **not** exercised the authenticated private input → queue → worker → result path. Run two tiny owner jobs, verify authorization and result hashes, avoid public GPU hosting.
4. **Optional LIVE-04 interpretation**: choose a feature with genuine positive and negative held-out examples before calling the local instruct model. Do not auto-label F506 from one strong token or leak evaluation groups into discovery.

The next visible win is a user opening **Prompt → Token → Feature → Measured Intervention** in a CPU staging Lab with original source-linked values. A published production Space or complete P8 certification is a separate later milestone.
