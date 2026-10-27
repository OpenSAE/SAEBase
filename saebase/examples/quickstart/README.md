# Quickstart

Start here for model/SAE loading, the public Python API, and a complete local experiment.

| File | What it does | Pretrained weights? |
| --- | --- | --- |
| [`first_experiment.py`](first_experiment.py) | Writes benchmark and reliability reports, with resumable caches | No |
| [`public_api_offline.py`](public_api_offline.py) | Captures and edits toy SAE features, reads downstream logits | No |
| [`load_model.py`](load_model.py) | Loads a registered model | Yes |
| [`load_sae.py`](load_sae.py) | Loads a registered SAE at a selected layer | Yes |
| [`load_prompts.py`](load_prompts.py) | Reads prepared prompt artifacts | Needs prepared data |
| [`model_checkpoint_probe.py`](model_checkpoint_probe.py) | Probes an actual registered model/SAE pair | Yes |

From the repository root:

```bash
python saebase/examples/quickstart/first_experiment.py --dry-run
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
python saebase/examples/quickstart/public_api_offline.py
```

The first two examples use small synthetic model/SAE weights. Real checkpoint examples require optional analysis dependencies and may require checkpoint access approval.

[Examples index](../README.md) · [First-experiment walkthrough](../../harness/github_development_doc/first-experiment.md)
