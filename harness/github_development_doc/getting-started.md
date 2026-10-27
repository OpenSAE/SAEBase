# Getting Started

SAEBase supports Python 3.11 and 3.12. The core package and its first examples work without downloading any pretrained model.

## Install and explore SAE features

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .

python saebase/examples/quickstart/public_api_offline.py
```

This example creates a small local model and SAE, captures a selected token's activation, edits features and compares logits.

## Run a full experiment on CPU

```bash
python saebase/examples/quickstart/first_experiment.py --dry-run
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
```

This evaluates two synthetic SAEs, reuses cached activations, measures feature interventions and produces a reliability audit. The command prints the exact paths of the HTML report, CSV summary and JSON results. Run it again to inspect the persisted results without repeating valid completed measurements.

See [Your First SAEBase Experiment](first-experiment.md) for an explanation of the outputs.

## Load a real public model and SAE

Install the optional analysis dependencies and run a small Pythia example:

```bash
python -m pip install -e ".[analysis]"
python saebase/examples/interventions/pythia_feature_edit.py
```

For a reproducible checkpoint benchmark, inspect its bounded plan before downloading weights:

```bash
saebase-benchmark --config conf/benchmark/pythia_benchmark_smoke.json --dry-run
saebase-benchmark --config conf/benchmark/pythia_benchmark_smoke.json --real
```

The benchmark uses one frozen example token sequence to verify the software pipeline; it is not a language-model evaluation dataset. The HTML report and source/metric manifests are saved under `runs/<run_id>/`.

## Continue with your own research

Use [Model and SAE Support](model-support.md) to choose checkpoints, [Activation Store](activation-store.md) to reuse captured activations, [Feature Interventions](feature-interventions.md) to edit codes, [Benchmarking](benchmarking.md) to configure metrics and [Reliability](reliability.md) to assess grouped uncertainty.

For external SAR, RSS, Input Localization and SAEBench measurements, see [Research Integrations](research-integrations.md). To prepare your own prompts and source manifests, consult the [Data and Python API Guide](python-api.md).

[Examples](../../saebase/examples/README.md) · [Documentation](index.md) · [Troubleshooting](troubleshooting.md)
