# SAEBase API Reference

SAEBase organizes its APIs around loading models, inspecting representations, reusing activations, running feature interventions, benchmarking and reliability analysis. For runnable code, see [Python API](python-api.md) and [Examples](../examples/README.md).

## Models and features

| Symbol | Purpose |
| --- | --- |
| `saebase.api.ModelAdapter` | Compatible model activation capture and patching |
| `saebase.api.ActivationSite` | Exact layer, hook and token position |
| `saebase.api.SAEIdentity` | Model/SAE checkpoint metadata and provenance |
| `saebase.api.load_registered` | Load a compatible registered model/SAE pair |
| `saebase.api.capture_features` | Inspect selected activation, codes and residual |
| `saebase.api.edit_features` | Reconstruct a code edit into a residual-preserving activation patch |
| `saebase.api.coalition_replacements` | Construct retain/ablate feature edits |
| `saebase.api.evaluate_edit` | Compare baseline and edited logits, checking identity replay |

The residual-preserving patch is `h' = h + decode(z') - decode(z)`. Full
reconstruction replacement is a distinct experimental control.
[Feature Interventions](feature-interventions.md) documents both.

## Activation/code stores

`saebase.store` provides `ActivationIdentity`, `CodeIdentity`,
`SampleRecord`, `TokenSelection`, `ActivationWriter`, `CodeWriter`,
`ShardedReader`, `collect_activations` and `encode_from_store`.

A cache's reuse is conditional on model, SAE, tokenizer, activation site,
token positions and sample identities, not just a matching file name.
[Activation Store](activation-store.md)

## Batch interventions

`saebase.interventions` provides `Condition`, `MeasurementSample`,
`ExperimentIdentity`, `RunBudget`, `BatchEngine`, `ResultWriter`,
`read_results` and steering utilities. Operator types include identity,
ablate, scale, clamp, add, coalition and reconstruction.

[Feature Interventions](feature-interventions.md)

## SAE benchmarking

`saebase.benchmarks` provides `BenchmarkConfig`, `MetricSpec`,
`MetricResult`, `load_config`, `get_metric`, `metric_specs` and
`build_plan`. The benchmark runner and result reader are available via
`saebase.benchmarks.runner.execute` and
`saebase.benchmarks.results.read_metrics`.

Per-metric units, status, protocol, sample counts, condition and upstream
source identity remain explicit. No universal composite quality score is
generated. [Benchmarking](benchmarking.md)

## Reliability and research controls

`saebase.reliability` provides `AuditConfig`, `Observation`,
`group_samples`, `bootstrap_metric`, `paired_bootstrap`,
`sample_size_curve`, `evaluate_m3`, `evaluate_ground_truth`,
`write_audit` and `read_audit`.

The function name `evaluate_m3` is a retained compatibility symbol for
a benchmark reliability audit. Prompt groups are the resampling units,
rather than separately counting dependent token/prompt variants.
[Reliability Analysis](reliability.md)

## Original-paper integrations

`saebase.integrations.sar` and `saebase.integrations.rss` use original
external research estimators. The native SAEBench importer reads separately
generated upstream JSON and validates model, SAE and source/protocol fields.
[Research Integrations](research-integrations.md)

## Console commands

| Command | Function |
| --- | --- |
| `saebase` | Registered tasks and SAE activation-health evaluation |
| `saebase-prepare-prompts` | Versioned prompt preparation |
| `saebase-check-env` | Runtime dependency validation |
| `saebase-intervene` | Batch interventions and steering controls |
| `saebase-benchmark` | SAE benchmark preflight, execution, resume and reports |
| `saebase-audit` | Reliability of existing benchmark measurements |
| `saebase-ground-truth` | Independent synthetic scientific controls |
| `saebase-release-matrix` | Public real multi-checkpoint acceptance |
| `saebase-release-gate` | Distribution/evidence integrity checking |

The `saebase-audit --m3-root` flag and machine-readable schema IDs remain
unchanged for backward compatibility.

[Getting Started](getting-started.md) · [Python API](python-api.md) · [Documentation Index](index.md)
