# SAEBase Documentation

Start with a working example, then choose the research task you want to perform. Guides are organized by capability rather than development stage.

## Getting started

- [**Quick Start**](../../README.md#-quick-start) — install SAEBase, run an offline example and try a real model/SAE in Python.
- [First Experiment](first-experiment.md) — create an HTML report, CSV metrics, source records and a reliability audit without model downloads.
- [Getting Started](getting-started.md) — installation, preflight and real-checkpoint commands.
- [Python API](python-api.md) · [API Reference](API.md) — model adapters, SAE features, stores and evaluation.
- [Runnable Examples](../../saebase/examples/README.md) · [Troubleshooting](troubleshooting.md).

## Research workflows

| Task | Guide |
| --- | --- |
| Capture and reuse model activations | [Activation Store](activation-store.md) |
| Edit SAE features and inspect downstream effects | [Feature Interventions](feature-interventions.md) |
| Measure reconstruction, sparsity and behavior | [SAE Benchmarking](benchmarking.md) |
| Bootstrap, paired comparisons and known-truth controls | [Reliability](reliability.md) |
| Study full input-factor grids | [Controlled Inputs](controlled-inputs.md) · [Input Localization](localization-benchmark.md) |
| Use original SAR, RSS and native SAEBench methods | [Research Integrations](research-integrations.md) |
| Load a custom model, SAE or estimator | [Third-Party Extensions](extensions.md) |

## Research Lab

- [Explore the Research Lab](../../saebase/research_lab/README.md) — recorded measurements, feature explorer and reports.
- [Measured results](research-lab/results-20261020.md) — model/SAE coverage and recorded SAR, RSS and feature-intervention evidence.
- [Release acceptance](research-lab/release-acceptance.md) — artifact checks, public data and deployment boundaries.

## Validated results

- [Verified Model and SAE Matrix](verified-model-matrix.md) · [Model Support](model-support.md)
- [GPU Performance Measurements](performance/workplan11_final_results.md) · [Performance Guide](performance.md)
- [Validation and Reproducibility](validation.md) · [Checkpoint Evidence](stage-d-evidence-archive.md)
- [Software Release v0.4.0](releases/v0.4.0.md) · [Previous Release v0.3.0](releases/v0.3.0.md)

## Repository and maintenance

- [**Repository Guide**](repository-guide.md) — where examples, configuration, research code, assets and evidence live.
- [Contributing](../../.github/CONTRIBUTING.md) · [Maintainer Workflow](maintainer-workflow.md) · [CI Checks](ci-maintenance.md)
- [Packaging](packaging.md) · [Software Release Process](release-process.md) · [Security](../../.github/SECURITY.md)
- [Historical Research and Engineering Notes](history.md) — older implementation, compatibility and checkpoint records.

Versioned source identities and older command aliases remain usable for reproducibility. They are not required to follow the current Quick Start.
