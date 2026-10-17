# Third-party extensions

Register external models, SAE checkpoints and benchmark providers through the explicit SAEBase extension API.

| File | What it does | Requirements |
| --- | --- | --- |
| [`external_providers.py`](external_providers.py) | Registers synthetic providers and runs an offline integration demo | Offline, core installation |
| [`qwen3_external_probe.py`](qwen3_external_probe.py) | Loads pinned Qwen3-1.7B Base/SAE weights at layer 7, checks an identity patch and feature ablation | GPU/appropriate RAM, upstream checkpoints |

```bash
python examples/extensions/external_providers.py
# Optional, after installing the analysis dependencies and obtaining weights:
python examples/extensions/qwen3_external_probe.py
```

A successful external probe establishes one checkpoint/site integration, not general interpretability or full population-level reliability.

[Examples index](../README.md) · [Extension API](../../docs/extensions.md)
