# Example benchmark configurations

## 目的和主要功能

说明固定样例配置的输入、用途与最小评测安排。

## 适用范围

用于选择工程示例和检查评测配置，真实执行采用所选资源与任务授权。

- `pythia_benchmark_smoke.json` runs a minimal real Pythia-70M model/SAE benchmark with frozen reference tokens.
- `pythia_grouped_benchmark.json` supplies three independent prompt groups for a reliability audit.

These are engineering examples, not research datasets. Inspect the selected plan with `saebase-benchmark --config conf/benchmark/pythia_benchmark_smoke.json --dry-run` when choosing its resources and execution scope.
