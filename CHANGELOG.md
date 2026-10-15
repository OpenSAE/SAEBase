# Changelog

## Unreleased

- **Input Localization integration:** capture or reuse complete Boolean input grids of SAE codes, delegate optimal scope curves and optional paired estimates to the original independent Localize implementation, and provide an offline example and tests.
- **Documentation:** add Localize to the English/Chinese research integration overview and examples index. 

## 0.2.0 — Reproducible SAE Evaluation, Interventions, and Reliability

- **Activation storage:** identity-bound sharded activation and dense/CSR code stores, with independently verified inputs and resumable execution.
- **Feature interventions:** batched ablation, scaling, clamping, addition, coalition and reconstruction conditions, downstream behavior measurements and reference steering.
- **Benchmarking:** versioned configurations, execution preflight, activation/measurement cache reuse, separate Core, reconstruction and intervention metrics, plus JSONL/CSV/HTML reports.
- **Research integrations:** unmodified original SAR/RSS estimators and provenance-checked native SAEBench Core and Sparse Probing imports.
- **Reliability audits:** grouped prompt bootstrap, paired comparisons, confidence intervals, sample-size curves, rank stability, known synthetic factors/interactions and negative controls.
- **Real checkpoint validation:** Pythia-70M and GPT-2 Small, with two different publicly available SAE checkpoint hashes per model, six fixed prompts per model and repeatability evidence.
- **Release integrity:** Python 3.11/3.12 checks, wheel/sdist verification, external checkpoint-source inventory and a GitHub Release with machine-readable evidence and SHA256SUMS.

**Scope:** Bounded engineering compatibility, not a broad comparative ranking of SAE quality. Third-party weights are not redistributed, and publication to PyPI is not implied.

## 0.1.0 — Initial Standalone Release Candidate

- Packaged model and SAE registry, configuration resources and core research interfaces.
- Added activation capture, residual-preserving feature edits and activation-health evaluation.
- Added versioned prompt artifacts and original-estimator integration examples.
- Added installation, command-line utilities, initial CI and contributor documentation.

This earlier release candidate did not establish broad external checkpoint or GPU generation reproducibility.
