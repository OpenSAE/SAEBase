# SAEBase Examples

This index groups executable examples by **what a researcher wants to do** rather than their file creation order. Numbered filenames are retained for reproducibility and backward compatibility.

## First experiment with saved results

Start with one command that writes a versioned benchmark report, a CSV summary, a reliability audit and reusable activation/intervention caches:

```bash
python examples/19_first_experiment_offline.py --output runs/first-experiment
```

The [first experiment walkthrough](../docs/first-experiment.md) explains the generated files and identity-checked resume. The example uses a deterministic synthetic model and two toy SAE weights, so no checkpoint download or GPU is needed.

## Learn the public API

| Example | What it demonstrates | Requires real weights? |
| --- | --- | --- |
| [`01_load_model.py`](01_load_model.py) | Registered model loading | Depending on execution |
| [`02_load_sae.py`](02_load_sae.py) | SAE loading and checkpoint profiles | Depending on execution |
| [`03_load_prompts.py`](03_load_prompts.py) | Prompt preparation and access | Depends on prepared inputs |
| [`05_public_api_offline.py`](05_public_api_offline.py) | ModelAdapter, activation capture, feature edits and logit comparison | No |
| [`06_public_api_pythia.py`](06_public_api_pythia.py) | The same public API with a genuine Pythia checkpoint | Yes |

Start here without downloading weights:

```bash
python examples/05_public_api_offline.py
```

## Activation inspection and caching

| Example | What it demonstrates |
| --- | --- |
| [`04_run_activation_health.sh`](04_run_activation_health.sh) | Activation-health task |
| [`10_activation_store_offline.py`](10_activation_store_offline.py) | Verified activation/code shards and resuming previously captured data |
| [`16_controlled_input_design.py`](16_controlled_input_design.py) | Enumerated binary factor grid, cached activations, bijective state reindexing, and Localize handoff |

```bash
python examples/10_activation_store_offline.py
python examples/16_controlled_input_design.py
```

## Interventions and downstream behavior

| Example | What it demonstrates |
| --- | --- |
| [`11_batch_interventions_offline.py`](11_batch_interventions_offline.py) | Batch feature conditions, raw measurements and steering reference |
| [`06_public_api_pythia.py`](06_public_api_pythia.py) | Feature removal and downstream logit differences on actual weights |

```bash
python examples/11_batch_interventions_offline.py
```

## Benchmarks, reliability and synthetic controls

| Example | What it demonstrates |
| --- | --- |
| [`12_benchmark_offline.py`](12_benchmark_offline.py) | Read-only benchmark preflight and compute budget |
| [`13_reliability_offline.py`](13_reliability_offline.py) | Prompt-group bootstrap and paired SAE comparisons |
| [`14_ground_truth_offline.py`](14_ground_truth_offline.py) | Known synthetic factors, interaction truth and invariance controls |
| [`09_second_model_smoke.py`](09_second_model_smoke.py) | Explicit opt-in secondary model/checkpoint loading |

```bash
python examples/12_benchmark_offline.py
python examples/13_reliability_offline.py
python examples/14_ground_truth_offline.py
```

## Integrate published research

| Example | What it demonstrates |
| --- | --- |
| [`07_sar_via_saebase.py`](07_sar_via_saebase.py) | Coalition interventions and the unchanged external SAR estimator |
| [`08_rss_via_saebase.py`](08_rss_via_saebase.py) | Clean, selection and held-out features with the unchanged RSS estimator |
| [`15_input_localization_via_saebase.py`](15_input_localization_via_saebase.py) | Complete controlled input grid and original optimal Localize scope curve |
| [`17_localization_benchmark.py`](17_localization_benchmark.py) | End-to-end Localize Benchmark with full curve artifacts and M4 stability audit |

The SAR/RSS examples need their original research estimator repositories. The Localize example needs the independent `sae-input-localization` package installed from [OpenSAE/SAE-Input-Localization](https://github.com/OpenSAE/SAE-Input-Localization). Their defaults use synthetic demonstration data; real checkpoint runs are opt-in. `17_localization_benchmark.py` additionally needs the installed original Localize package and writes complete M3/M4 reports.

## Use third-party models and benchmarks

| Example | What it demonstrates | Requires real weights? |
| --- | --- | --- |
| [`18_external_extensions_offline.py`](18_external_extensions_offline.py) | Opt-in model/SAE providers, source-hashed benchmark and identity patch checks | No |
| [`19_first_experiment_offline.py`](19_first_experiment_offline.py) | Persistent benchmark, shared caches, reliability and HTML report | No |

See [Third-Party Extensions](../docs/extensions.md) for registering an installed research adapter.

## A real multi-checkpoint benchmark

For an end-to-end, versioned validation of Pythia-70M and GPT-2 Small with two public SAEs each, use the stable command-line interface rather than an extra ad hoc example:

```bash
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
```

Real weights are downloaded on demand and are not included in the SAEBase package. Small smoke-test results should not be interpreted as a general ranking of SAE quality.

[Documentation Index](../docs/index.md) · [Getting Started](../docs/getting-started.md) · [Python API](../docs/python-api.md)
