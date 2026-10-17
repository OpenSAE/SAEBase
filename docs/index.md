# SAEBase Documentation

SAEBase is an open-source toolkit for evaluating sparse autoencoders, intervening on learned features, reusing model activations and auditing experimental reliability.

Start with the task you want to perform. The documentation is organized by **capability**, not by the order in which the project was developed.

## Getting started

- [First Experiment](first-experiment.md) — a single CPU-only command that writes persistent metric, provenance and reliability reports.
- [Troubleshooting](troubleshooting.md) — installation, checkpoint and benchmark error checks.
- [Verified Models and SAE Checkpoints](verified-model-matrix.md) — 18 configurations, 48 evaluated layer sites, and linked results.
- [Getting Started](getting-started.md) — installation, offline example, benchmark preflight and a real checkpoint walkthrough.
- [Models and SAEs](model-support.md) — registry versus verified checkpoint coverage, public source links and access constraints.
- [Examples Index](../examples/README.md) — runnable demonstrations grouped by research task.
- [Python API](python-api.md) — public research functions, store/benchmark/audit entrypoints and CLI commands.
- [Packaging and Compatibility](packaging.md) — installable namespaces, original CLI compatibility and distribution boundaries.
- [API Reference](API.md) — quick lookup of major public interfaces.

## Research capabilities

- [Controlled Input Designs](controlled-inputs.md) — complete binary factor grids, state mapping, cache identity and Localize handoff.
- [Activation Capture and Storage](activation-store.md) — reusable, integrity-checked activation and SAE code caches.
- [Feature Interventions and Steering](feature-interventions.md) — residual-preserving edits, conditions and downstream measurements.
- [SAE Benchmarking](benchmarking.md) — metrics, versioned configurations, experiment planning, reports and external native results.
- [Reliability Analysis](reliability.md) — prompt-group bootstrap, paired comparisons and scientific ground-truth controls.
- [Input Localization Benchmarking](localization-benchmark.md) — exact curves, metric artifacts, native paired-estimator stability and matched SAE comparisons.
- [Research Integrations](research-integrations.md) — using original SAR/RSS/Localize estimators and native SAEBench results.
- [Third-Party Extensions](extensions.md) — external model, SAE and original benchmark providers.

## Evidence and maintenance

- [Contributor Guide](contributor-guide.md) — find a first task, run tests and submit a reproducible pull request.
- [Maintainer Workflow](maintainer-workflow.md) — ownership, code review and the CI/release process.
- [CI and Research Validation Lanes](ci-maintenance.md) — pinned regressions, optional real-checkpoint acceptance and release safeguards.
- [Benchmark Evidence and Reproducibility](stage-d-evidence-archive.md) — download and audit original experiment measurements and verification reports.
- [Earlier Checkpoint Studies](phase-a-model-coverage.md) and [CPU Validation Records](stage-bc-model-sae-coverage.md) — historical compatibility tests retained for reproducibility.
- [Validation and Reproducibility](validation.md) — actual public-checkpoint acceptance versus optional and unverified capabilities.
- [Checkpoint Acceptance Protocol](reference-checkpoint-validation.md) — detailed reference experiment and checksum requirements.
- [Public API Validation](public-api-validation.md) — detailed API behavior and compatibility tests.
- [Research Integration Validation](research-integration-validation.md) — original-source and protocol acceptance notes.
- [Release and Checkpoint Matrix](checkpoint-validation.md) — versioned multi-checkpoint acceptance, performance and integrity checks.
- [General Software Release Process](release-process.md) — exact-commit CI gate, candidate checksums and manual version publication.
- [Release Notes: v0.3.0](releases/v0.3.0.md) — current software capabilities, compatibility, and verified evidence.
- [Changelog](../CHANGELOG.md), [Provenance](PROVENANCE.md), [Release Checklist](RELEASE_CHECKLIST.md), [Contributing](../CONTRIBUTING.md) and [Security](../SECURITY.md).

## Terminology and compatibility

The `saebase` Python package has stable public modules for API access, storage, interventions, benchmarks, reliability and integrations. Persisted schemas such as `saebase.m2.measurement.v1` and existing CLI flags such as `--m3-root` are preserved for compatibility. These are **machine-facing identifiers**, not names that users must learn to understand the research workflow.

The project publishes source and executable tools; it does not redistribute third-party model or SAE weights. A registered model or SAE is not automatically an experimentally validated model/SAE pair.
