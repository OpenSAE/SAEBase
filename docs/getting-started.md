# Getting Started

SAEBase provides a lightweight offline interface, and optional analysis dependencies for real pretrained language models and SAEs. Python 3.11–3.12 is supported.

## Install

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .
```

This installs the core APIs and CPU-only examples. Use `python -m pip install -e ".[dev]"` for tests and `python -m pip install -e ".[analysis]"` when you want to load pretrained checkpoints. GitHub [Releases](https://github.com/OpenSAE/SAEBase/releases) also contain archived versions; the current development APIs are available from this source checkout.

## Start without model downloads

Run the [first experiment](first-experiment.md) to create benchmark and reliability results in one directory:

```bash
python examples/quickstart/first_experiment.py --output runs/first-experiment
```

The command prints the location of a standalone HTML report, CSV summaries and verified JSON/manifest files. You can rerun it to verify and resume the same experiment. It uses two synthetic SAEs and no downloads.

To explore individual capabilities instead:

```bash
python examples/quickstart/public_api_offline.py
saebase-benchmark --config conf/benchmark/p0_small.json --dry-run
python examples/activations/activation_store.py
```

The first example uses an explicitly synthetic model/SAE to demonstrate capture, editing and logits. The benchmark preflight checks workload and identities without downloading checkpoints. The third example creates and reuses a sharded activation/code store offline.

## Run a real SAE benchmark

```bash
saebase-benchmark --config conf/benchmark/p0_small.json --real
saebase-benchmark --config conf/benchmark/p0_small.json --report-only
```

The configuration identifies a real Pythia-70M model/SAE pair and a small fixed-token validation sample. Real execution downloads checkpoints and needs an appropriate CPU or GPU environment. Results are written under `runs/<run_id>/` with separate metrics, source provenance, a CSV summary and standalone HTML. This example validates engineering behavior, not a representative language dataset.

For a real feature-edit example:

```bash
python examples/interventions/pythia_feature_edit.py
```

## Prepare your own prompt artifacts

SAEBase also supports versioned prompt datasets with explicit behavior, split and seed metadata.

```bash
saebase-prepare-prompts \
  --model-key pythia70m --behavior SimpleCopying \
  --device cpu --n-generate 300 --min-kept 20 --seed 0 \
  --data-dir ./data
```

This creates `data/SimpleCopying/pythia70m/v1/prompts.jsonl` and
`manifest.json`. The original activation-health CLI can evaluate these prompts:

```bash
saebase \
  --task sae --model pythia70m --sae pythia70m_resid_sm \
  --prompt-source SimpleCopying --layer 3 --n-prompts 5 \
  --data-dir ./data --device cpu \
  --run-dir ./experiment/pythia70m-smoke
```

The output includes `sae_activation_summary.json` and a resolved Hydra configuration. Model/SAE checkpoint downloads are required for these two commands.

## Choose what to do next

If you want to evaluate two SAE dictionaries over shared model activations, see [Activation Store](activation-store.md) and [SAE Benchmarking](benchmarking.md). If you want to edit features and measure output changes, see [Feature Interventions](feature-interventions.md). For paired statistical comparisons and known-truth controls, see [Reliability Analysis](reliability.md).

[Examples](../examples/README.md) · [Python API](python-api.md) · [Model Support](model-support.md) · [Troubleshooting](troubleshooting.md)
