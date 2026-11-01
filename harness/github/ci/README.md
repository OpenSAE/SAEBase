# SAEBase CI lanes: lightweight development, full release verification

## Daily commits and pull requests

`.github/workflows/ci.yml` runs on both PRs and frequent `main` pushes.
It selects the **union** of checks affected by all changed paths, including
both ends of renames. Unavailable Git history or unknown paths select every
*lightweight* scope, but never trigger publishing or full dependency installs.

- `changes`: compare PR base → tested merge commit, or push before → after.
- `quality`: always run Python syntax/new-file Ruff, workflow routing and
  release-contract regression.
- `harness-targeted`: Harness and universal Contract regression.
- `atlas-targeted`: Core/Knowledge source and Atlas browser JavaScript checks.
- `product-targeted`: core-only installation and bounded Registry/Runtime checks.
- `ci-required`: one stable branch-protection status. Fail if a selected scope
  fails, is cancelled or never completes; an intentionally skipped scope is
  allowed only when its classifier output is false.

The source of routing truth is `harness/github/ci/select_checks.py`.
Every routing rule has regression cases in `test_ci_select_checks.py`.
Do not classify unknown changes as documentation to make CI green.
PR and main use the same rules, and newer commits cancel superseded ordinary
CI runs. Main CI is intentionally not Release evidence.

## Release Validation

Full validation is a separate **manual** GitHub Action:
`.github/workflows/release-validation.yml`.

1. Select a current `main` commit and copy its exact 40-character SHA.
2. Manually dispatch `Release Validation` with `source_sha=<sha>`.
3. The first job requires the provided SHA to equal current `main`. Every
   full-testing job checks out that same SHA and waits for this source gate.
4. The original complete offline coverage, `.[analysis]` installation,
   packaging, real Research Lab CPU Docker, Hydra, Atlas/Harness, localization
   and pinned SAR/RSS regression run. All jobs must pass.
5. `release-validation-required` fails closed if any required job is absent,
   skipped, cancelled or unsuccessful. Distribution artifacts are produced
   by the exact full-validation run.
6. Only after full validation, explicitly dispatch
   `software-release.yml` with that same `source_sha`, selected version,
   and `mode=verify`. For publication, a human separately selects
   `mode=publish` and enters the exact `PUBLISH-vX.Y.Z` confirmation.
   The software release workflow checks the complete Release Validation run,
   its five historical required job names and its built distributions.
   A lightweight CI success cannot satisfy this gate.

A newer `main` commit invalidates the existing current-main publication
check. Validate the new exact SHA instead; never silently reuse a release
candidate from another commit. CI success never establishes independent
GPU/scientific validation or grants any publication permission.

## Other automated workflows

Historical v0.4.0/v0.5.0/v0.6.0/v0.6.5 release dispatchers are retained as
manual archive notices, not automatic release actions. Full pinned
research-regression is manual/reusable by Release Validation.
Atlas snapshot regeneration and browser-validated Pages deployment are
manual, separate choices. The weekly non-gating upstream compatibility
monitor remains scheduled.

## Measuring results

Report GitHub Actions `created_at`, start and completion for each workflow,
including queue delay; compare median and P90 PR/main wall time and total
runner-minutes against the previous unconditional CI runs. Distinguish
cancelled superseded runs and Release Validation from daily CI.
No duration estimate is a measured result until checked against real runs.
