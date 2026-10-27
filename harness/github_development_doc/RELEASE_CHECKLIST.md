# SAEBase v0.2.0 Release Verification

For future software versions, see the [General Software Release Process](release-process.md). This checklist records the original v0.2.0 validation and must not be used to overwrite its tag or published assets.

The [official v0.2.0 GitHub Release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.2.0) has been published. Its annotated tag identifies the exact validated source commit. The release distributes SAEBase software, **not third-party model or SAE weights**.

## Completed publication checks

- [x] Python 3.11 and 3.12 tests, analysis-dependency installation, wheel build and source-archive verification.
- [x] Real checkpoint matrix for two independent model families with two different SAE weight hashes per model.
- [x] Identity-bound activation, SAE code and intervention caches; resume and integrity checks.
- [x] Per-metric benchmark results, original-estimator checks and prompt-group reliability audits.
- [x] Model and SAE source-card inventory and explicit licensing status.
- [x] GitHub Release with wheel, source archive, release-evidence JSON and SHA256SUMS.
- [x] Success of real multi-checkpoint acceptance and standard CI for the same source commit as the published tag.

The release verifier refuses missing, failed or mismatched results. The six-prompt real checkpoint matrix is engineering acceptance, not a population-level scientific ranking.

## Reproduce the code and distribution checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
python saebase/tools/verify_distribution.py dist
python saebase/tools/check_installed_wheel.py dist
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
```

For actual model/SAE execution, install `.[analysis]`, allow on-demand
checkpoint downloads, and provide sufficient RAM:

```bash
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
saebase-release-matrix --config conf/release/m5_matrix.json --report-only
```

`saebase-release-gate` checks the original evidence, source inventory,
release archives, SHA-256 digests and source commit.

## Boundaries that require separate confirmation

- The maintainers remain responsible for contributor licensing rights and upstream attributions.
- Model, SAE, tokenizer and dataset assets retain the original providers' access and redistribution conditions.
- Optional GPU backend performance requires its own hardware acceptance.
- Meaningful population-level model/SAE quality claims need representative data and protocols beyond the small engineering smoke.

This checklist is not a legal certification. Read [Release Notes](releases/v0.2.0.md), [Checkpoint Validation](checkpoint-validation.md), [Validation](validation.md) and [Provenance](PROVENANCE.md).
