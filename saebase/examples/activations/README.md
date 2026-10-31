# Activation capture and storage

Demonstrate activation extraction, shared SAE encoding, cache integrity and resume behavior.

| File | What it does | Requirements |
| --- | --- | --- |
| [`activation_store.py`](activation_store.py) | Captures synthetic activations once and reuses them across two SAE checkpoints | Offline, core installation |
| [`activation_health.sh`](activation_health.sh) | Runs the registered activation-health task | Prepared prompts, real model and SAE weights |

```bash
python saebase/examples/activations/activation_store.py
# Run only after preparing prompts and model checkpoints:
bash saebase/examples/activations/activation_health.sh
```

[Examples index](../README.md) · [Activation Store](../../../harness/github/development_doc/activation-store.md)
