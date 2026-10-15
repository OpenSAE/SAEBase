# SAEBase v0.2.0 — release checklist

This checklist covers a **code distribution** only. The release must not
contain third-party language-model or SAE weights.

## Automated gates

- [x] M0–M4 interfaces, provenance and original metric math remain unchanged.
- [x] Minimum real release matrix: Pythia-70M and GPT-2 Small, at least two
  **different actual SAE weights** per family with SHA-256 checkpoint identity.
- [x] Real M1 caching, M2 condition measurement, M3 reports, M4 grouped
  audits and repeated resume/report-only are tested in CI.
- [x] Python 3.11/3.12 offline tests, `.[analysis]` install, wheel and
  source distribution inspection run in GitHub Actions.
- [x] Third-party model/SAE source cards and license *status* are inventoried;
  distribution files contain no external checkpoint weights.
- [ ] Exact final `main` commit has successful CI and M5 real matrix runs.
- [ ] Final wheel/sdist and source SHA256SUMS reverified by the release workflow.
- [ ] `v0.2.0` GitHub tag and Release exist with artifacts and evidence.

Human/legal review remains separate: contributors must have rights to license
their code; upstream model/SAE licenses and datasets may impose conditions.
This checklist is **not** a legal certification.

## Verify code and distributions

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist
python scripts/check_installed_wheel.py dist
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
```

To reproduce the real matrix (opt-in downloads, additional CPU/RAM):

```bash
python -m pip install -e '.[analysis]'
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
saebase-release-matrix --config conf/release/m5_matrix.json --report-only
```

Given the verified M5 matrix root, distributions and exact commit,
`saebase-release-gate` produces `release-evidence.json` and `SHA256SUMS`.

The GitHub release workflow listens for a successful main-branch
`M5 real two-family checkpoint matrix acceptance`, verifies CI on the
*same commit*, downloads its test evidence and the CI-built distributions,
audits/checksums both, and publishes an annotated `v0.2.0` tag and release.
It refuses failed/missing/mismatched proof. No PyPI upload occurs.

See [docs/M5_RELEASE_MATRIX.md](M5_RELEASE_MATRIX.md),
[docs/M5_RELEASE_NOTES.md](M5_RELEASE_NOTES.md) and
[docs/PROVENANCE.md](PROVENANCE.md).
