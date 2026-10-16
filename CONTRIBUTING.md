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

## Standards

- Preserve the SAE identity, hook, token position, and residual semantics.
- Keep new features importable without requiring GPU-only optional backends.
- Include deterministic offline tests and identify steps requiring downloads.
- Keep public documentation aligned with shipped executable tasks.
- Record third-party source links, licenses, and attribution in the PR.

Review `docs/API.md`, `docs/RELEASE_CHECKLIST.md`, and `SECURITY.md` before a release.
