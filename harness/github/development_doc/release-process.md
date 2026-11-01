# General Software Release Process

The [manual Software Release workflow](../../../.github/workflows/software-release.yml)
publishes new SAEBase versions after separate, source-bound verification.
The existing `v0.2.0` tag and the [historical M5 release
gate](../../../.github/workflows/m5-release.yml) remain untouched.

## Prepare the source

In a focused pull request, update the following files for the next
`major.minor.patch` version:

- `pyproject.toml` under `[project].version`.
- `saebase/__init__.py` with the identical `__version__`.
- `CHANGELOG.md` with a new `## X.Y.Z` section.
- `harness/github/development_doc/releases/vX.Y.Z.md` with changes, compatibility implications
  and links to the tests and scientific acceptance relevant to this release.

Model, SAE, hook, decoder, normalization, intervention or metric changes
require their own real acceptance evidence. CI package checks do not prove
scientific benchmark quality or checkpoint correctness. Include actual
evidence links rather than implying that old validation covered new code.
Review software source licensing and third-party provenance as usual.

## Exact-source Release Validation

Ordinary PR and `main` runs are deliberately lightweight. **They are not
release evidence**. After merging the candidate to `main`, copy the exact
current `main` SHA and manually start
[Release Validation](../../../.github/workflows/release-validation.yml) on
`main` with `source_sha=<40-character SHA>`.

The source gate verifies the requested SHA is the exact current main commit.
Every full job checks out that same source. It executes the old complete
offline coverage, `.[analysis]` install, full package/Research Lab Docker,
Localization, Hydra, quality, Atlas/Harness and pinned SAR/RSS regressions.
The final gate requires all mandatory jobs to succeed.

The software release workflow checks a successful **workflow_dispatch
Release Validation run for the identical SHA**, its five historical required
contexts (`offline-tests (3.11)`, `analysis-install`, `package`,
`localization-native`, `quality`), and uses only the distribution assets
from that very same full-validation run. Skipped/failed/cancelled jobs,
an older SHA or the lightweight main CI cannot substitute.

The standard automated lane runs Python 3.11. Python 3.12 is still declared
compatible, so relevant changes require their own explicit compatibility
evidence when releasing; the existing Research Lab Docker runtime remains
separate.

## Verify without publishing

In GitHub Actions choose **SAEBase Software Release (manual)**, run on
`main`, and enter the exact current main SHA and the version in the
project. Select `mode = verify` and leave the confirmation blank.

The job downloads the fully release-validated assets and independently checks package
metadata, all nine CLI names, software license declaration, absence of
checkpoint weights and archived Stage D evidence, installation outside
the checkout, clean rebuilding from sdist, and source/asset SHA256.

The candidate artifact contains the wheel, sdist,
`release-evidence.json` with version/commit/CI run and verification
scope, and `SHA256SUMS`. This mode **never creates a Git tag,
GitHub Release or PyPI upload**, including when run for the already
published version 0.2.0.

For a local preliminary check, without claiming that GitHub's CI gate was
verified:

~~~bash
python -m pip install -e ".[dev]"
python -m build
python saebase/tools/verify_distribution.py dist
python saebase/tools/check_installed_wheel.py dist
python saebase/tools/check_sdist_roundtrip.py dist
python saebase/tools/release_candidate.py prepare \
  --dist dist --output-dir release-candidate \
  --source-sha "$(git rev-parse HEAD)"
python saebase/tools/release_candidate.py verify-assets \
  --dist dist --output-dir release-candidate \
  --source-sha "$(git rev-parse HEAD)"
~~~

## Publish a new version

After reviewing the candidate, rerun the workflow on `main` with
`mode = publish` and the same exact source SHA and version. Enter
the explicit confirmation `PUBLISH-vX.Y.Z`.

The publishing job executes only if verification succeeds. It refuses:

- a source SHA that is not the current `main` HEAD;
- a missing, failed or wrong-SHA Release Validation job;
- a version inconsistent across project, package and release notes;
- a version that is older than or equal to an existing software tag;
- any attempt to move an existing tag or overwrite a Release;
- a missing wheel, sdist or incorrect checksum.

After rechecking checksums, the job creates a new annotated version tag
and publishes four release assets: wheel, sdist, `release-evidence.json`
and `SHA256SUMS`, using the release notes at the tagged source
commit. It verifies the four remote asset names and resolves the tag
back to that same SHA.

The earlier `v0.2.0` version cannot be republished using this
workflow. Previously published Release artifacts remain immutable.

The workflow intentionally does **not** publish to PyPI. Future PyPI
distribution should use a separate reviewed Trusted Publishing workflow,
not a long-lived personal API token.

## Atlas snapshots and historical dispatchers

Atlas snapshots and Pages deployment are manually triggered; ordinary
`main` pushes do not generate extra snapshot commits. If the source SHA
changes after full validation, restart Release Validation on the new
current-main SHA. No prior PR check or lightweight CI result can be reused.

Historical version-specific release dispatchers (v0.4.0, v0.5.0, v0.6.0,
v0.6.5) are retained as non-publishing manual archive notices.
There is no automatic publish transition from lightweight CI success.

## After publication

Review the new GitHub Release, its exact commit, release notes and all
four asset hashes. A recipient can download the four files and run:

~~~bash
sha256sum -c SHA256SUMS
~~~

The release evidence asserts software distribution integrity and
successful CI, **not** independent verification of new scientific claims.
Review [CI lanes](ci-maintenance.md), [provenance](PROVENANCE.md) and
the [historical v0.2.0 checklist](RELEASE_CHECKLIST.md) as necessary.

## One-time approved Research Lab v0.4.0 publication

The source of the version-specific release request lives at
[Research Lab v0.4.0 Approved Release Dispatch](../../../.github/workflows/research-lab-v04-release-dispatch.yml).
It exists solely for the explicit maintainer-approved Research Lab v0.4.0
publication after successful *push-to-main* CI of the final source.

That job does **not** publish assets itself or bypass the original manual
release gates. It dispatches the existing source-bound
`software-release.yml` using the exact main commit, required confirmation
`PUBLISH-v0.4.0`, and publish mode. The existing verifier still checks
all six CI contexts, builds distributions from the same run, verifies their
SHA-256 and installs them before making an immutable tag. Existing version
tags must not be overwritten. Future versions still use the normal manual
process; there is no general automatic software release enabled.

Release assets contain software, release-evidence and checksums only. The
GPU experiment archive and third-party source texts are excluded.

## One-time maintainer-authorized SAEBase v0.5.0 publication

The maintainer explicitly requested the v0.5.0 release. A version-specific
[one-time CI dispatch](../../../.github/workflows/saebase-v050-release-dispatch.yml)
reacts **only** to a completed, successful push-to-main CI for the exact
current head, confirms `pyproject.toml` is version 0.5.0 and the release
notes exist, and invokes the existing checked `software-release.yml`
workflow with `mode=publish` and `PUBLISH-v0.5.0`.
The software release gate independently rechecks all six CI contexts,
wheel and sdist hashes, installed import, source identity and tag
immutability before publication. It does not upload PyPI packages,
publish private GPU artifacts, or provision a public inference service.

This is deliberately scoped to **v0.5.0 only** and cannot publish
future versions. Prior v0.4.0 and all historical tags stay immutable.

## One-time maintainer-authorized SAEBase v0.6.0 publication

The maintainer explicitly authorized the v0.6.0 release following review of PR #60 and its source-bound CPU CI and documented bounded real-GPU tests. [The version-specific dispatch](../../../.github/workflows/saebase-v060-release-dispatch.yml) may act **only** after a successful push-to-main CI for the exact current main SHA, provided that source declares version 0.6.0 and contains its release notes.

The dispatch requests the existing `software-release.yml` `mode=publish` with `PUBLISH-v0.6.0`. The normal release gate still checks the original six CI contexts, wheel/sdist installation, tag immutability, SHA-256 and source identity. It cannot publish other versions or replace a version tag. No PyPI, GPU model assets or private research data are published.
