# CI and Research Validation Lanes

SAEBase deliberately separates **frequent development checks** from **full
software release acceptance**, real model experiments, and historical evidence.
A lightweight result does not establish full scientific or release validity.

## PR and main: the same lightweight CI

[CI](../../../.github/workflows/ci.yml) runs after every PR update and
`main` push, using the actual Git diff (including old and new rename paths).
[Selector](../ci/select_checks.py) combines all relevant scopes; unknown
paths or missing Git history conservatively run all lightweight scopes.

- `changes` identifies the needed Harness, Atlas, Product and CI-policy checks.
- `quality` always verifies added Python quality, CI routing and release gates.
- `harness-targeted` exercises governance and Contract tests when relevant.
- `atlas-targeted` exercises source validation and JavaScript Atlas navigation
  when relevant.
- `product-targeted` runs bounded Registry, Runtime and configuration checks
  with the core/development dependencies when product code changes.
- `ci-required` is the **only recommended required branch-protection status**;
  skipped scopes are permitted only if they were not selected. Selected checks
  must finish successfully.

No routine PR/main run installs `.[analysis]`, runs full xdist coverage,
builds Wheel/sdist and Research Lab Docker, or runs pinned research acceptance.
The sole routine Python version is 3.11. PR and `main` use the same policy.
New commits cancel superseded ordinary runs, including rapid `main` pushes.
Exact per-commit full release evidence is provided separately.

## Full software release validation

[Release Validation](../../../.github/workflows/release-validation.yml)
is **manual** and accepts only an explicitly selected current `main` SHA.
Its source gate runs before any full job, and every job checks out that same
source revision.

It preserves the complete Python 3.11 offline suite + coverage, `.[analysis]`
installation, full package/sdist and real Research Lab CPU Docker checks,
localization reference tests, Hydra, Atlas/Harness checks and pinned original
SAR/RSS integration and ground-truth parity. Its final gate requires every
mandatory job to complete successfully. Distribution artifacts come from
the exact full-validation run.

[Software Release](../../../.github/workflows/software-release.yml) accepts
only a successful Release Validation run on that exact SHA, checks the five
historically required job names and downloads its distribution artifacts.
It requires explicit `PUBLISH-vX.Y.Z` confirmation for actual publication.
Lightweight `main` CI cannot supply release evidence or automatically publish.

The historical v0.4.0/v0.5.0/v0.6.0/v0.6.5 release dispatchers are now
manual-only archived notices. They no longer listen for normal CI successes.

## Other scheduled or manual integrations

[Research Integration Regression](../../../.github/workflows/research-regression.yml)
can run manually or be invoked by Release Validation. Both pinned upstream
sources remain fixed: SAR `c253dca303437318c1f4f4937557ac9d22143b19`,
RSS `0b321851e7b79564bf72656e1dbcaaca513048d8`.
[Upstream Compatibility Watch](../../../.github/workflows/upstream-compatibility.yml)
still checks floating upstream sources weekly; it is not a release proof.

Real M0/M1/M2 Pythia, native SAEBench, M3/M4, M5 matrix, Stage B/C and
Phase A CPU/GPU probes retain their explicit manual entrypoints and evidence
requirements. Software release CI never asserts new GPU scientific results
without real independent evidence.

Atlas source snapshot regeneration and browser-validated Pages deployment
are also manual. They no longer generate another `main` commit or expensive
Chromium validation following ordinary source changes.

## Measuring the improvement

For matched PR/main scopes record GitHub Actions creation time, start time,
finish time and per-job durations, plus cancelled superseded runs. Report
wall-clock median/P90 and total runner-minutes, separately from release tests.
Preserve previous run records; do not describe planned time savings as measured.
See [CI routing details](../ci/README.md) and
[Software release process](release-process.md).
