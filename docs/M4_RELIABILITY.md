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
- For `core.reconstruction_mse`, `core.fvu` and
  `core.explained_variance`, `--sae-only` explicitly loads the frozen SAE
  checkpoint and decodes M1 activations/codes **without a model forward**.
  Full decoder weight SHA-256 is checked against M3 provenance.
- Other scalar-only M3 result families without subgroup data are explicitly
  labeled `unsupported`, not assigned synthetic uncertainty.

The M4 report is saved as `reliability.json`,
`reliability_manifest.json`, `reliability_summary.csv` and
`reliability.html` under the original `runs/<m3_run_id>/`.
M3's `report.html` automatically adds the M4 audit once both source
manifests validate. Changes to M3 measurements invalidate the audit; changing
bootstrap seeds never reexecutes model forwards.

The real Pythia smoke exercises three published M0 reference prompts.
Three prompt groups are sufficient to validate engineering behavior but
not to claim narrow, generalizable population confidence intervals.

## M4-2: known truth, controllable degradation and negative controls

M4-2 adds deterministic independent synthetic controls. The tests never
represent the training of actual published SAE checkpoints. The fixture
families are independent factors, correlated factors, parent-child hierarchical
factors and overcomplete superposed dictionaries. Each fixture records its
dictionary, generative mechanism and seed. Feature matching is fit only on
`calibration` groups; the mapping is frozen before scoring `test` groups.
Permutation/sign-aware recovery and reconstruction error remain separate
properties. A faithful reconstruction alone does not establish unique features.

Behavioral ground-truth masks use a fully known centered Boolean interaction
expansion with one-, two- and three-way terms. The **unchanged** original SAR
implementation fits only order 1 and 2. Thus, an injected third-order term
should leave a positive SAR2 residual. We do not claim an upstream SAR3 metric.
RSS fixtures inject precisely known responsive support and directions, select
on one variant collection and measure persistence on a disjoint held-out
collection, including deliberately flipped held-out direction and empty support
cases. All SAR/RSS source file SHA-256 digests are retained in the report.

Controlled degradations independently vary evaluation noise at a **fixed
synthetic dictionary**, preserving separate training and evaluation seeds.
The report measures feature recovery, exact paired differences in
reconstruction error, percentile CI, Kendall tau, top-k overlap and ranking
reversal frequencies. Other negative controls cover null intervention,
feature permutation, legal inverse code/decoder positive scale changes,
constant SAR behavior, a known affine readout, empty RSS support, nonfinite
RSS input rejection and the precise scaling rule for MeanRSS.

For a complete M3-attached audit with the original paper estimators:

```bash
saebase-audit --m3-root "runs/$RUNID" \
  --config conf/reliability/m4_ground_truth.json --run --ground-truth \
  --sar-source /path/to/SAR/sar.py --rss-source /path/to/RSS/rss.py
```

Or run only the independent synthetic scientific checks:

```bash
saebase-ground-truth --config conf/reliability/m4_ground_truth.json \
  --sar-source /path/to/SAR/sar.py --rss-source /path/to/RSS/rss.py \
  --output synthetic-ground-truth.json
```

Changing either original source file invalidates the attached M4 report.
Failures and undefined metrics are explicit, not forced to favorable values.
The formal release/multi-model matrix remains the responsibility of M5.

### SAE-only Core reconstruction audits

The CLI defaults to fully model- and SAE-free auditing for raw M2 metrics
and core L0. Add `--sae-only` for Core reconstruction MSE/FVU/EV. This
explicitly permits SAE checkpoint downloads/loading and re-decodes the
existing M1 data; it does **not** reload the language model or perform any
additional model forward. The SAE state SHA-256 is verified before evaluation.
Changing bootstrap seeds simply reuses the saved activations/codes.

### Predeclared primary comparisons and multiplicity

`AuditConfig.primary_comparisons` optionally names explicit pairs of SAE
identifiers. These are the primary pairwise comparisons; when multiple
primary pairs are supplied, percentile interval coverage is adjusted
using Bonferroni (`1 - (1-confidence)/n_primary`). When not supplied,
all compatible comparisons are marked **exploratory** and the reported
descriptive intervals are unadjusted. Neither family may be presented
as a single universal SAE total score.

### Evaluation-seed sensitivity of native estimators

In addition to a frozen baseline SAR/RSS fixture, `coalition_reseed_trials`
repeat the original SAR estimator with each evaluation seed, re-drawing
coalition masks and fit/held-out partitions while keeping the known
interactions fixed. `rss_reseed_trials` independently re-draw the RSS
selection/held-out variants while keeping the truth fixed. Their raw
seed-specific SAR1/SAR2, support, direction and persistence are retained.
These synthetic tests are not equivalent to retraining real SAEs or fitting
new probes on empirical datasets; that distinction is explicit.
