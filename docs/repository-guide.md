# Repository Guide

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`examples/quickstart/`](../examples/quickstart/README.md) | First API examples and a complete CPU experiment | [First experiment](../examples/quickstart/first_experiment.py) |
| [`examples/activations/`](../examples/activations/README.md) | Activation capture and code-cache reuse | [Activation store](../examples/activations/activation_store.py) |
| [`examples/interventions/`](../examples/interventions/README.md) | Feature edits, downstream logits and steering | [Pythia feature edit](../examples/interventions/pythia_feature_edit.py) |
| [`examples/benchmarking/`](../examples/benchmarking/README.md) | Benchmark preflight and reliability analysis | [Benchmark preflight](../examples/benchmarking/benchmark_preflight.py) |
| [`examples/research_integrations/`](../examples/research_integrations/README.md) | SAR, RSS, Input Localization and SAEBench | [Research examples](../examples/research_integrations/README.md) |
| [`saebase/`](../saebase/) | Python APIs for loading, measuring, editing and reporting | [API guide](python-api.md) |
| [`models/`](../models/) and [`conf/`](../conf/) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`apps/research-lab/`](../apps/research-lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../apps/research-lab/README.md) |
| [`assets/`](../assets/) | Branding and workflow figures used in documentation | [Workflow diagram](../assets/figures/saebase_workflow.svg) |
| [`docs/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`scripts/`](../scripts/README.md) | Maintenance, verification and reproducible GPU experiment entrypoints | [Script guide](../scripts/README.md) |
| [`work_plan/`](../work_plan/README.md) | Historical implementation plans and execution records | [Engineering history](../work_plan/README.md) |

## Reproducible evidence

The checked-in [model/SAE compatibility matrix](verified-model-matrix.md), [performance reports](performance/workplan11_final_results.md), and [Research Lab measurements](research-lab/results-20261020.md) describe distinct tested workloads. Reuse them with the original checkpoints, inputs and execution protocol recorded alongside each result.

The [coverage evidence directory](../conf/coverage/) contains versioned source records and checksums from earlier experiments. Internal schema IDs and archival filenames stay unchanged so existing measurement caches and prior releases remain interpretable. New users do not need them for the [Quick Start](../README.md#-quick-start).

The older checkpoint, implementation and release notes remain available through the [historical documentation index](history.md). They are not additional setup steps.

## Check your checkout

Run these commands from the repository root:

```bash
python -m pip install -e ".[dev]"
python scripts/check_documentation.py --check
python -m pytest -q
```

New users can begin with [Quick Start](../README.md#-quick-start), then choose a task from the [Examples Index](../examples/README.md).
