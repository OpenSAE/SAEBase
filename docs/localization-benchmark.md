# Input Localization Benchmarking

SAEBase uses the original [SAE Input Localization](https://github.com/OpenSAE/SAE-Input-Localization) estimator in benchmarking and reliability analysis. The input grid contains the binary factors defined for the experiment, not SAE feature ablations. The estimator remains in the original repository.

## Run the executable example

```bash
python -m pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git
python examples/research_integrations/localization_benchmark.py --dry-run
python examples/research_integrations/localization_benchmark.py --output runs
```

The example uses a synthetic model and two SAE checkpoint identities. All eight configurations of three controlled factors are captured and cached, and both scope-curve metrics are generated. This is an engineering demonstration, not the paper's learned-SAE experiment.

## Configure a run

Use the existing saebase.benchmark.config.v1 format. The dataset must supply one exact SampleRecord per state in a complete uniform binary grid, each with distinct token IDs, one selected token position, variant_role="controlled", the same split and the **same prompt_group_id**. All states form one evaluation population, not independent prompts.

The options.localization section has this shape:

```json
{
  "localization": {
    "name": "controlled-three-factor-design",
    "factors": [
      {"name": "first", "levels": ["0", "1"]},
      {"name": "second", "levels": ["0", "1"]},
      {"name": "third", "levels": ["0", "1"]}
    ],
    "sample_ids_by_state": ["s0","s1","s2","s3","s4","s5","s6","s7"],
    "pairs_per_scope": 64
  }
}
```

Those sample IDs must exactly match the eight samples in dataset.samples; they are ordered by the binary factor bitmask. pairs_per_scope is optional and specifies pairs **per candidate scope** for the original paper's paired estimator. Preflight rejects invalid grids without downloading model weights.

Add both metrics (or either one) to the benchmark metric list:

```json
["localization.budget_at_90pct", "localization.budget_fraction_at_90pct"]
```

The first measures the minimum number of input-feature incidences needed to explain 90% of SAE activation variance, and the second divides that count by n_factors * n_features. They are **contextual descriptors**, not globally comparable SAE-quality scores. Zero activation variance is reported as undefined.

## Full curve artifacts

The benchmark writes a metric ledger plus run-local files in localization/<artifact-id>.json. Each artifact preserves exact and normalized optimal scope curves, optimal feature assignments as bitmasks, original estimator source SHA-256, the complete versioned controlled-input manifest, source cache identities, optional paired plug-in estimate, and a whole-file checksum. Each scalar metric row contains the artifact reference, and the HTML report links to it. Report-only and resume workflows verify native source and artifact integrity before reusing the result.

The activation and SAE code caches can be reused across compatible SAE checkpoints. Existing SampleRecord, activation store, benchmark schemas and v0.2.0 release artifacts are unchanged.

## Reliability and comparisons

Use a reliability audit configuration with "metric_ids": ["localization.budget_at_90pct"], plus evaluation_seeds for repeated paired estimation, for example [0,1,2,3,4]. Execute:

```bash
saebase-audit --results-dir runs/<run_id> --config path/to/audit.json --run
```

The audit loads the saved SAE codes without model forwards. It uses the original Localize paired estimator to report, per seed, the estimated budget and deviation from the exact full-grid curve. It reports observed seed-to-seed ranges, **not population confidence intervals**: a complete factorial grid is one group, and its individual states must not be bootstrapped as independent observations. These results are finite-pair Monte Carlo sensitivity checks, not post-selection certificates.

Two SAE checkpoints may be compared only when the model, hook, dataset, state mapping and original estimator match. Raw budget differences are only shown when SAE feature counts agree; fractional budget differences retain feature-count metadata. All comparisons are descriptive and explicitly avoid SAE rankings or claims about overall SAE quality.

The audit outputs localization_audits and localization_comparisons alongside the other reliability results, with corresponding HTML and CSV entries. Bootstrap analysis for other supported measurements remains unchanged.

[Controlled Input Designs](controlled-inputs.md) · [Benchmarking](benchmarking.md) · [Reliability](reliability.md)
