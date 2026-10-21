# Example benchmark configurations

- `pythia_benchmark_smoke.json` runs a minimal real Pythia-70M model/SAE benchmark with frozen reference tokens.
- `pythia_grouped_benchmark.json` supplies three independent prompt groups for a reliability audit.

These are engineering examples, not research datasets. Run `saebase-benchmark --config conf/benchmark/pythia_benchmark_smoke.json --dry-run` before downloading checkpoints.
