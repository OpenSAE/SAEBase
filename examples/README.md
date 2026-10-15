# SAEBase Examples

This index groups executable examples by **what a researcher wants to do** rather than their file creation order. Numbered filenames are retained for reproducibility and backward compatibility.

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

```bash
python examples/10_activation_store_offline.py
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

The original research estimator repository must be supplied for these examples. Their defaults use synthetic demonstration data; real checkpoint runs are opt-in.

## A real multi-checkpoint benchmark

For an end-to-end, versioned validation of Pythia-70M and GPT-2 Small with two public SAEs each, use the stable command-line interface rather than an extra ad hoc example:

```bash
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
```

Real weights are downloaded on demand and are not included in the SAEBase package. Small smoke-test results should not be interpreted as a general ranking of SAE quality.

[Documentation Index](../docs/index.md) · [Getting Started](../docs/getting-started.md) · [Python API](../docs/python-api.md)
