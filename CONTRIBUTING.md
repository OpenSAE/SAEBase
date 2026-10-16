# Contributing to SAEBase

Thanks for contributing to reproducible SAE research.

## Start here

1. Open an issue describing a model adapter, SAE checkpoint, evaluation improvement, or bug.
2. Work in a focused branch and include a test for each behavioral change.
3. Run `python -m pip install -e ".[dev]"` and `python -m pytest -q`.
4. For checkpoint-dependent changes, include the model/SAE identifiers, hook and layer, hardware, and a minimal reproduction.
5. Submit a pull request that explains the motivation, evidence, and compatibility impact.

Please keep research metrics in their paper repositories. SAEBase supplies shared
model operations, SAE loading, interventions, and reproducible experiment I/O.

## Packaging and imports

Use `saebase.api` and `saebase.cli` for new public interfaces. The legacy top-level `main`, `models`, `scripts`, `tasks`, and `conf` names remain installed for compatibility. Do not remove or rename them without an explicit migration and independent installed-wheel tests. See [Packaging and Compatibility](docs/packaging.md).

## Standards

- Preserve the SAE identity, hook, token position, and residual semantics.
- Keep new features importable without requiring GPU-only optional backends.
- Include deterministic offline tests and identify steps requiring downloads.
- Keep public documentation aligned with shipped executable tasks.
- Record third-party source links, licenses, and attribution in the PR.

Review `docs/API.md`, `docs/RELEASE_CHECKLIST.md`, and `SECURITY.md` before a release.

## Code quality

Install `python -m pip install -e ".[dev,quality]"` and see [Code Quality and Coverage](docs/code-quality.md). Ruff and mypy initially cover selected modules; new Python files must pass Ruff lint and formatting. Existing code will not be reformatted across the repository.
