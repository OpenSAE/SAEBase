# M4 — Reliability Audit & Ground Truth

M4 is the reliability layer on top of M3, not another SAE benchmark.
We distinguish an M3 metric's point estimate from uncertainty about that
metric and from validity against known ground truth. All audit outputs are
versioned and are explicitly linked to the unchanged M3 metric manifest.

## M4-1: evaluation stability and paired confidence intervals

A stable `saebase-audit` entrypoint accepts a previously completed M3 run.
For example:

```bash
python -m pip install -e '.[dev]'
python examples/13_reliability_offline.py
saebase-benchmark --config conf/benchmark/m4_pythia_three_groups.json --dry-run
saebase-benchmark --config conf/benchmark/m4_pythia_three_groups.json --real --output runs
RUNID=$(python -c 'from saebase.benchmarks import load_config;print(load_config("conf/benchmark/m4_pythia_three_groups.json").run_id)')
saebase-audit --m3-root "runs/$RUNID" --config conf/reliability/m4_pythia_smoke.json --dry-run
saebase-audit --m3-root "runs/$RUNID" --config conf/reliability/m4_pythia_smoke.json --run
saebase-audit --m3-root "runs/$RUNID" --config conf/reliability/m4_pythia_smoke.json --report-only
```

`--real` belongs to the M3 model execution stage. The M4 audit stage does
not load model/SAE checkpoints or call a model forward.

Each `Observation` must have an exact `prompt_group_id`, dataset,
split, SAE, metric, condition, evaluation seed and sample ID. The default
independent resampling unit is the **original prompt group**, not its
variants, coalitions or tokens. Whole groups are drawn with replacement;
all variants and token positions belonging to that group stay together.
The mean remains observation-unit-weighted (not a mean of group means).
Train, calibration and held-out populations are never pooled.

Bootstrap defaults: five explicit evaluation seeds (0 to 4), 512 bootstrap
draws per seed, two-sided 95% percentile intervals and fixed random
generators. Confidence intervals and point estimates retain their
`metric_id`, `metric_version`, `protocol_id`, counts and direction
without constructing a composite SAE score.

For SAE pairs, exact `prompt_group_id` alignment and sample/condition
identity are required. Identical groups are drawn for both SAEs; we
bootstrap their difference directly. Ranking stability (Kendall tau,
top-k overlap, rank reversal frequency) is produced only for metrics whose
registered direction is `higher` or `lower`. Contextual metrics do not
acquire artificial rankings. Tied ranks remain invalid and are counted.
Subsample size curves use the configured sizes, clamped to actual groups.

## M3 measurement reuse and source verification

Supported raw observation families in the initial M4-1 stage:

- `core.l0` uses the original M1 SAE code cache and validates the exact
  code sample, token positions, stored checkpoint identity and M3 numerical mean.
- `intervention.*` loads M2's checksum-verified measurement journal,
  checks source identities, sample/condition and each M3 measurement reference.
- Other scalar-only M3 result families have no subgroup data at this stage
  and are labeled `unsupported`, not assigned synthetic uncertainty.

The M4 report is saved as `reliability.json`,
`reliability_manifest.json`, `reliability_summary.csv` and
`reliability.html` under the original `runs/<m3_run_id>/`.
M3's `report.html` automatically adds the M4 audit once both source
manifests validate. Changes to M3 measurements invalidate the audit; changing
bootstrap seeds never reexecutes model forwards.

The real Pythia smoke exercises three published M0 reference prompts.
Three prompt groups are sufficient to validate engineering behavior but
not to claim narrow, generalizable population confidence intervals.

## M4-2

The next part adds known factor dictionaries, known-order SAR behavior
interactions, RSS selection/held-out directional truth, controllable
degeneration, invariance/negative controls and formal validation reports.
M5 remains responsible for the large multi-checkpoint release matrix.
