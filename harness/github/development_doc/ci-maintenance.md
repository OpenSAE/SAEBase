# CI and Research Validation Lanes

SAEBase separates ordinary software checks, original research regression,
real model and SAE acceptance, and historical release gates. Passing an
offline test does not establish that a real checkpoint was verified.

## Fast software and packaging CI

[CI](../../../.github/workflows/ci.yml) runs on every pull request and on pushes
to `main`. Five release-required job contexts remain, and normal PR checks run on a
single Python 3.11 version:

- `offline-tests (3.11)` runs the full offline regression suite in two
  isolated xdist workers (`--dist=loadfile`), retains branch coverage, and
  executes documentation/evidence, examples, CLI and installed-wheel checks.
  The `saebase-core-coverage-py311` artifact is unchanged.
- `quality` runs Ruff, incremental mypy, and pre-commit validation.
- `package` validates wheel and sdist contents and a clean sdist
  rebuild. Runtime configs remain available without bundling experiment caches.
- `analysis-install` verifies optional model analysis dependencies.
- `localization-native` tests the original, source-pinned Input
  Localization integration.

These five jobs are reported for every PR and are suitable candidates for
branch protection. The [administrative ruleset issue](https://github.com/OpenSAE/.github/issues/3)
tracks enabling that enforcement.

## CI runtime, performance and compatibility

The automatic CI and generic software release verification jobs now use Python
3.11. Setup-python's pip wheel/download cache is keyed by dependency
declarations. The same functional checks, source-pinned regression and exact-SHA
release gates remain required. Superseded pull-request runs may be cancelled;
all main-branch push CI runs remain independent for release provenance.

The two-worker `pytest-xdist` execution runs the **same suite** with per-file
worker affinity. It continues to generate the same branch coverage reports.
Always compare measured run time and coverage with the earlier serial baseline;
a faster CI cannot justify dropping a failing or flaky test.

`pyproject.toml` still declares support for Python 3.11 and 3.12, while
Python 3.12 is no longer tested in routine PR CI. If a future release relies on
3.12 compatibility, verify that interpreter explicitly before publication.
The Research Lab's existing Python 3.12 Docker **runtime image** remains
unchanged; the CI Docker smoke continues to exercise that image.

## Pinned research integration regression

[Research Integration Regression](../../../.github/workflows/research-regression.yml)
runs for changes to benchmarks, reliability, integrations and their research
tests. It can also be started manually. Both jobs use the original SAR and
RSS estimator implementations at immutable source revisions:

- SAR: `c253dca303437318c1f4f4937557ac9d22143b19`.
- RSS: `0b321851e7b79564bf72656e1dbcaaca513048d8`.

One job checks the original M3 estimator parity and protocol contracts.
The other checks M4 synthetic truth, factor interactions and negative
controls, preserving the native ground truth JSON as an artifact.
Their results are reproducible against the recorded source revisions.

[Upstream Compatibility Watch](../../../.github/workflows/upstream-compatibility.yml)
checks the *latest* SAR/RSS default branches once per week or on manual
request, recording the resolved source SHAs. Its findings are **not
required PR checks**, and a failure indicates a compatibility change
rather than invalidating the pinned historical regression.

## Manual real checkpoint and scientific acceptance

The following suites require model weights, checkpoint downloads,
permissions or substantial CPU/GPU work. None are automatically triggered
for a regular PR:

| Purpose | GitHub workflow |
| --- | --- |
| Real M0/M1/M2 Pythia | [Real Pythia Smoke](../../../.github/workflows/real-smoke.yml); choose `m0`, `m1`, `m2` or `all` |
| Real M3/M4 Pythia | [M3](../../../.github/workflows/m3-real.yml) and [M4](../../../.github/workflows/m4-real.yml) |
| Native SAEBench parity | [M3 Native SAEBench](../../../.github/workflows/m3-saebench-native.yml) |
| Frozen four-checkpoint release matrix | [M5 Matrix](../../../.github/workflows/m5-matrix.yml) |
| Original Phase A | [CPU feasibility](../../../.github/workflows/phase-a-cpu-probes.yml), [SAE-only](../../../.github/workflows/phase-a-sae-probes.yml), [model-SAE runner](../../../.github/workflows/phase-a-real.yml) |
| Stage B/C CPU acceptance | [Model pairs](../../../.github/workflows/stage-bc-cpu-pairs.yml), [SAE-only](../../../.github/workflows/stage-bc-cpu-sae.yml), [Gemma 9B](../../../.github/workflows/stage-bc-gemma9b-recheck.yml), [source metadata](../../../.github/workflows/stage-bc-source-head.yml) |

The old separate `m0-real.yml`, `m1-real.yml` and `m2-real.yml`
definitions were combined into a single manual matrix with the same
scripts, frozen fixtures, arguments and evidence artifact names.
The previous `m3-papers.yml` and `m4-ground-truth.yml`
jobs were consolidated into the pinned research workflow.
Previous GitHub Actions run records and the source history of those
retired workflow files remain intact. Historic scientific evidence was
not rewritten.

For changes affecting SAE normalization, hooks, checkpoint identity,
loading or native model behavior, maintainers should run the relevant
manual acceptance and link the actual passing result to the PR.

## Historical release safety

The [v0.2.0 M5 release gate](../../../.github/workflows/m5-release.yml)
still checks exact main-branch source SHA, real checkpoint evidence and
successful same-commit CI. The M5 acceptance itself is now manually
triggered, so routine code merges no longer start the old publish chain.

The [Stage D evidence publication](../../../.github/workflows/stage-d-evidence-release.yml)
continues to require successful main CI and refuses to overwrite an
existing evidence release. Missing original GPU run archives remain
tracked in [Issue #21](https://github.com/OpenSAE/SAEBase/issues/21).

Future versions now use the [General Software Release Process](release-process.md)
and its [manually dispatched release workflow](../../../.github/workflows/software-release.yml).
This requires exact-source successful main CI, a matching version and
verified distribution assets before creating a new tag. Existing tags,
including the historical `v0.2.0` release, are never moved or overwritten.

## Future maintenance

- Add offline regressions to the existing core CI instead of creating a
  separate workflow for every research feature.
- Pin original source revisions for reproducible PR decisions. Test
  upstream HEAD separately.
- Keep real checkpoint jobs opt-in with explicit timeouts and evidence.
- Do not configure manually or path-filter-triggered workflows as
  universally required CI checks.
- Before renaming a standard job or workflow used in `workflow_run`,
  update its dependent rulesets and release guards.
- Use `tests/test_ci_lanes.py` to catch accidental regressions in
  trigger policy or original-source pinning.
