# Research integrations

SAEBase provides the model/SAE operations while original research projects retain their estimators and evaluation protocols.

| File | What it does | Extra requirements |
| --- | --- | --- |
| [`sar.py`](sar.py) | Feature-coalition interventions with the original SAR estimator | [SAR repository](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) |
| [`rss.py`](rss.py) | Clean/selection/held-out feature changes with the original RSS estimator | [RSS repository](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) |
| [`input_localization.py`](input_localization.py) | Controlled input grid and optimal scope curve | [Input Localization](https://github.com/OpenSAE/SAE-Input-Localization) |
| [`controlled_inputs.py`](controlled_inputs.py) | Complete binary input designs and cache reuse | Offline; optional Localize |
| [`localization_benchmark.py`](localization_benchmark.py) | Full Localize benchmark and reliability example | Independent Localize package |

From the repository root:

```bash
python examples/research_integrations/controlled_inputs.py
python examples/research_integrations/sar.py --paper-dir /path/to/SAR-repo
python examples/research_integrations/rss.py --paper-dir /path/to/RSS-repo
python examples/research_integrations/input_localization.py
```

The default SAR/RSS examples use synthetic inputs; the paper repositories are needed for their original estimators. Install Input Localization separately before its examples.

[Examples index](../README.md) · [Research Integrations](../../docs/research-integrations.md)
