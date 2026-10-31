# Repository Guide

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`saebase/examples/quickstart/`](../../saebase/examples/quickstart/README.md) | First API examples and a complete CPU experiment | [First experiment](../../saebase/examples/quickstart/first_experiment.py) |
| [`saebase/examples/activations/`](../../saebase/examples/activations/README.md) | Activation capture and code-cache reuse | [Activation store](../../saebase/examples/activations/activation_store.py) |
| [`saebase/examples/interventions/`](../../saebase/examples/interventions/README.md) | Feature edits, downstream logits and steering | [Pythia feature edit](../../saebase/examples/interventions/pythia_feature_edit.py) |
| [`saebase/examples/benchmarking/`](../../saebase/examples/benchmarking/README.md) | Benchmark preflight and reliability analysis | [Benchmark preflight](../../saebase/examples/benchmarking/benchmark_preflight.py) |
| [`saebase/examples/research_integrations/`](../../saebase/examples/research_integrations/README.md) | SAR, RSS, Input Localization and SAEBench | [Research examples](../../saebase/examples/research_integrations/README.md) |
| [`saebase/`](../../saebase) | Reusable Python APIs, runtime capabilities, provenance and storage | [API guide](python-api.md) |
| [`experiments/`](../../experiments) | Concrete research experiment definitions and composition | [QA500 experiment](../../scripts/experiments/qa500/README.md) |
| [`scripts/`](../../scripts) | Shell launchers and installation entrypoints | [QA500 local runner](../../scripts/experiments/qa500.sh) |
| [`models/`](../../models) and [`conf/`](../../conf) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`saebase/research_lab/`](../../saebase/research_lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../../saebase/research_lab/README.md) |
| [`saebase/assets/`](../github_assets) | Branding and workflow figures used in documentation | [Workflow diagram](../github_assets/figures/saebase_workflow.svg) |
| [`harness/github_development_doc/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`saebase/tools/`](../../saebase/tools/README.md) | Maintenance, verification and reproducible GPU experiment entrypoints | [Script guide](../../saebase/tools/README.md) |
| [`harness/github_work_plan/`](../github_work_plan/README.md) | Historical implementation plans and execution records | [Engineering history](../github_work_plan/README.md) |

## Reproducible evidence

The checked-in [model/SAE compatibility matrix](verified-model-matrix.md), [performance reports](performance/workplan11_final_results.md), and [Research Lab measurements](research-lab/results-20261020.md) describe distinct tested workloads. Reuse them with the original checkpoints, inputs and execution protocol recorded alongside each result.

The [coverage evidence directory](../../conf/coverage) contains versioned source records and checksums from earlier experiments. Internal schema IDs and archival filenames stay unchanged so existing measurement caches and prior releases remain interpretable. New users do not need them for the [Quick Start](../../README.md#-quick-start).

The older checkpoint, implementation and release notes remain available through the [historical documentation index](history.md). They are not additional setup steps.

## Check your checkout

Run these commands from the repository root:

```bash
python -m pip install -e ".[dev]"
python saebase/tools/check_documentation.py --check
python -m pytest -q
```

New users can begin with [Quick Start](../../README.md#-quick-start), then choose a task from the [Examples Index](../../saebase/examples/README.md).
