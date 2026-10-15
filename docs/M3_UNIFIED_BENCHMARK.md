# M3 — Unified SAE Benchmark

M3 builds a versioned benchmark planner, a bounded M1/M2 orchestration layer,
separate metric implementations, adapters to the *original* SAR/RSS estimators
and provenance-checked imports for native SAEBench Core / Sparse Probing output.

## Status and scientific validation

Engineering implementation is distinct from numerical validation. The offline
example proves only the synthetic execution contract. SAEBench JSON import tests
prove that an already-generated native result is faithfully transcribed, **not**
that a SAEBench evaluation ran or that its dataset/protocol matches SAEBase.
M3 acceptance requires separately executing two upstream native evaluations
with pinned model, SAE checkpoint, dataset, preprocessing and protocol, then
comparing numerical outputs. Unsupported or not-yet-executed paths are marked
`blocked` in reports rather than assigned zero or called validated.

## Install / preflight

```bash
python -m pip install -e '.[dev]'
saebase-benchmark --config conf/benchmark/p0_small.json --dry-run
python -m pytest -q tests/test_m3_plan.py tests/test_m3_execution.py
```

Dry-run validates the input without model downloads. It emits deterministic
run IDs, a task DAG, total SAE/metric jobs and pessimistic upper bounds for
intervention measurements and requested token positions. Unknown execution
time and GPU memory appear as null.

For real weights (network access and adequate RAM required):

```bash
saebase-benchmark --config conf/benchmark/p0_small.json --real
saebase-benchmark --config conf/benchmark/p0_small.json --real --resume
saebase-benchmark --config conf/benchmark/p0_small.json --report-only
```

The smoke fixture token IDs originate from a frozen reference and should not be
treated as a study dataset. The code evaluates the exact target token supplied
by each sample; it does not claim whole-corpus language modeling perplexity.

## Dataset, metric and storage contract

`BenchmarkConfig` has schema `saebase.benchmark.config.v1`. It defines
`targets` with a model ID, multiple SAE IDs and an explicit layer/hook site,
`dataset.id` and tokenized `dataset.samples`, one or more exact metric IDs,
optional M2 `conditions`, a seed, a bounded compute budget, estimator
`sources` and optional metric parameters in `options`.

Metric results include version/protocol, model/SAE/site, dataset/split,
unit/valid observations, result status, original measurement references,
upstream commit/hash, warnings and costs. A metric can be `completed`,
`undefined`, `failed`, `unsupported` or `blocked`. Missing data and
invalid denominators do not silently become zeros.

Health metric semantics are explicit:

- `core.l0`: arithmetic mean of per-selected-token nonzero code counts.
- `core.reconstruction_mse`: mean of per-token elementwise SAE reconstruction MSE.
- `core.fvu`: mean per-token `1 - fve` as defined by
  `saebase.sae.contracts.reconstruction_stats`, **not** the global
  variance-normalized SAEBench FVU.
- `core.explained_variance`: legacy per-token `fve` with original 1e-12
  denominator stabilizer; **not** interchangeable with SAEBench's official
  explained variance.
- `reconstruction.target_nll`: mean negative log probability at an explicitly
  requested next-token position after full SAE reconstruction replacement.
- `reconstruction.loss_recovered`: uses an independently measured full-site zero
  ablation loss on the same model, samples and token readouts. It is undefined
  if ablated loss equals clean loss.
- `intervention.*`: aggregates unchanged M2 `ResultWriter` per sample and
  per named condition. Conditional values are never mixed across interventions.

The result directory is `runs/<run_id>/` and contains:
`resolved_config.json`, `plan.json`, `provenance.json`, `metrics.jsonl`,
`metrics_manifest.json`, `summary.csv`, `report.html`. Shared M1/M2 measurement caches live below
`runs/_cache/` rather than inside any metric-specific run. M1 identities are independent of SAE; M2 measurement
identities bind both checkpoint hashes. Repeated complete run IDs are reused.

## Original-paper integrations

Set `sources.sar` to a trusted local checkout or `sar.py`, and
`sources.rss` to a trusted local checkout or `rss.py`. SAEBase invokes the
**unmodified** estimator via `saebase.integrations`, preserving its
coalition/held-out definitions. SAR requires target and reference IDs; RSS
requires disjoint selection/held-out variants and their original prompt texts
supplied through `options.rss.prompts`. The local source file SHA-256 is saved.

Native SAEBench integration accepts `sources.saebench.core.l0`,
`sources.saebench.core.mse` and
`sources.saebench.sparse_probing.sae_top_1_test_accuracy` pointing to
**upstream-generated** JSON files. It verifies the declared upstream evaluation
type, commit, exact SAE lens id and native model name; it does not rerun,
reinterpret or copy upstream estimators. A benchmark with SAEBench metrics must
set `options.saebench_sae_lens_ids` mapping every SAEBase SAE key to the
official upstream checkpoint ID, and `options.saebench_model_names` mapping
each model key to the native SAEBench model name. Optional
`options.saebench_sae_releases` maps each SAE key to the upstream release ID.
A provenance mismatch produces a blocked metric, never a mislabeled score.
For formal comparisons, independently verify input matching and conduct two
native runs under an explicitly pinned protocol.

## Boundaries and next steps

M3 has no composite SAE ranking. M4 will add grouped bootstrap and synthetic
ground truth audit to these frozen measurements. M5 will execute a full
multi-model multi-checkpoint matrix and publish verified v0.2 artifacts.

### Reuse across benchmark configurations

Cache paths are shared under `runs/_cache/m1/` and `runs/_cache/m2/`, keyed
by exact model/SAE/site and their M1/M2 identity manifests rather than M3
metric selections. Changing the benchmark's metric set changes `run_id` but
not these physical caches; previously committed activation rows and M2
interventions are reused after content and identity checks. `--no-resume`
rejects only an existing completed M3 report, not valid shared caches.
