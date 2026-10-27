# SAEBase Examples

Runnable examples are organized by **research task**, not implementation order. Start from the repository root with Python 3.11 or 3.12 and a source installation:

```bash
python -m pip install -e .
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
```

The first experiment writes an HTML report, metric summaries, and a reliability audit without downloading pretrained weights. Examples involving real checkpoints require the optional `.[analysis]` dependencies and the upstream model/SAE weights.

## Choose a task

| Task | Examples | Start here |
| --- | --- | --- |
| [Quickstart](quickstart/README.md) | Load models, SAEs and prompts; use the API; complete a first experiment | [First experiment](quickstart/first_experiment.py) |
| [Activations](activations/README.md) | Capture activations, reuse encoded features and inspect activation health | [Activation store](activations/activation_store.py) |
| [Interventions](interventions/README.md) | Remove or edit SAE features and inspect downstream effects | [Batch interventions](interventions/batch_interventions.py) |
| [Benchmarking & reliability](benchmarking/README.md) | Inspect workload, bootstrap results and run ground-truth controls | [Benchmark preflight](benchmarking/benchmark_preflight.py) |
| [Research integrations](research_integrations/README.md) | SAR, RSS, Input Localization and controlled input designs | [Integration index](research_integrations/README.md) |
| [Third-party extensions](extensions/README.md) | Register external model/SAE/metric providers and probe a real Qwen checkpoint | [Offline external providers](extensions/external_providers.py) |

### First, run an offline experiment

```bash
python saebase/examples/quickstart/first_experiment.py --dry-run
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
```

See the [first-experiment walkthrough](../harness/github_development_doc/first-experiment.md) for the resulting cache, benchmark report, and reliability audit. All numbers in this example come from **synthetic toy weights**, not published research findings.

### Then explore individual capabilities

```bash
python saebase/examples/quickstart/public_api_offline.py
python saebase/examples/activations/activation_store.py
python saebase/examples/interventions/batch_interventions.py
python saebase/examples/benchmarking/reliability_bootstrap.py
python saebase/examples/benchmarking/ground_truth_controls.py
```

Run a real checkpoint only after installing `.[analysis]` and reviewing the relevant project's access and hardware requirements.

The shared `_synthetic_common.py` helper is used by multiple offline examples and existing tests. It is not an independent command-line example.

## Source releases and compatibility

The original numbered paths remain accessible in the immutable [v0.3.0 GitHub release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.3.0) and its source archive. **The current `main` branch uses the task-based paths above.** Public package APIs, CLIs, verification records and existing release tags are unchanged.

[Getting Started](../harness/github_development_doc/getting-started.md) · [Documentation](../harness/github_development_doc/index.md) · [Python API](../harness/github_development_doc/python-api.md)
