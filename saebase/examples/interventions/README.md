# Feature interventions

Edit SAE features and measure effects on downstream model behavior.

| File | What it does | Requirements |
| --- | --- | --- |
| [`batch_interventions.py`](batch_interventions.py) | Ablation, scaling, batched conditions and steering on a deterministic toy model | Offline, core installation |
| [`pythia_feature_edit.py`](pythia_feature_edit.py) | Removes features in a real Pythia model and reports logit changes | Public pretrained weights, analysis dependencies |

```bash
python saebase/examples/interventions/batch_interventions.py
# Optional, with real checkpoint dependencies:
python saebase/examples/interventions/pythia_feature_edit.py
```

[Examples index](../README.md) · [Feature Interventions](../../../harness/github/development_doc/feature-interventions.md)
