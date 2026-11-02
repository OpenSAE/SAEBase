# Repository Guide

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`saebase/examples/quickstart/`](../../../saebase/examples/quickstart/README.md) | First API examples and a complete CPU experiment | [First experiment](../../../saebase/examples/quickstart/first_experiment.py) |
| [`saebase/examples/activations/`](../../../saebase/examples/activations/README.md) | Activation capture and code-cache reuse | [Activation store](../../../saebase/examples/activations/activation_store.py) |
| [`saebase/examples/interventions/`](../../../saebase/examples/interventions/README.md) | Feature edits, downstream logits and steering | [Pythia feature edit](../../../saebase/examples/interventions/pythia_feature_edit.py) |
| [`saebase/examples/benchmarking/`](../../../saebase/examples/benchmarking/README.md) | Benchmark preflight and reliability analysis | [Benchmark preflight](../../../saebase/examples/benchmarking/benchmark_preflight.py) |
| [`saebase/examples/research_integrations/`](../../../saebase/examples/research_integrations/README.md) | SAR, RSS, Input Localization and SAEBench | [Research examples](../../../saebase/examples/research_integrations/README.md) |
| [`saebase/`](../../../saebase) | Python APIs for loading, measuring, editing and reporting | [API guide](python-api.md) |
| [`models/`](../../../models) and [`conf/`](../../../conf) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`saebase/research_lab/`](../../../saebase/research_lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../../../saebase/research_lab/README.md) |
| [`saebase/assets/`](../assets) | Branding and workflow figures used in documentation | [Workflow diagram](../assets/figures/saebase_workflow.svg) |
| [`harness/github/development_doc/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`saebase/tools/`](../../../saebase/tools/README.md) | Maintenance, verification and reproducible GPU experiment entrypoints | [Script guide](../../../saebase/tools/README.md) |
| [`harness/github/work_plan/`](../work_plan/README.md) | Current task planning and review controls | [Work Plan guidance](../work_plan/README.md) |

## Reproducibility

Current experiments bind the model, dataset, task, source commit, configuration and recorded outputs to a single run identity. Use the declared configurations and task APIs rather than an older engineering plan as execution authority.

## Check your checkout

Run these commands from the repository root:

```bash
python -m pip install -e ".[dev]"
python saebase/tools/check_documentation.py --check
python -m pytest -q
```

New users can begin with [Quick Start](../../../README.md#-quick-start), then choose a task from the [Examples Index](../../../saebase/examples/README.md).
