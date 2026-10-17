# Benchmarking and reliability

Inspect benchmark plans and evaluate statistical reliability before running costly experiments.

| File | What it does | Requirements |
| --- | --- | --- |
| [`benchmark_preflight.py`](benchmark_preflight.py) | Prints a deterministic benchmark plan without loading weights | Offline |
| [`reliability_bootstrap.py`](reliability_bootstrap.py) | Group bootstrap and paired comparisons | Offline |
| [`ground_truth_controls.py`](ground_truth_controls.py) | Known-factor, interaction and invariance controls | Offline |

```bash
python examples/benchmarking/benchmark_preflight.py
python examples/benchmarking/reliability_bootstrap.py
python examples/benchmarking/ground_truth_controls.py
```

These examples demonstrate method execution rather than scientific claims about pretrained SAE quality.

[Examples index](../README.md) · [Benchmarking](../../docs/benchmarking.md) · [Reliability](../../docs/reliability.md)
