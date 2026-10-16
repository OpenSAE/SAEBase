# Python API and Command-Line Interfaces

SAEBase exposes distinct APIs for activation inspection, storage, intervention experiments, benchmarking, and reliability analysis. The original Python names and CLI commands are retained for compatibility.

## Public research API

The primary interface is `saebase.api`.

| Symbol | Purpose |
| --- | --- |
| `ActivationSite` | Selects a layer, exact model hook and token position |
| `ModelAdapter` | Normalizes supported model activation capture and patching |
| `SAEIdentity` | Records loaded checkpoint, dimensions, model, hook, layer and provenance |
| `load_registered` | Loads a registered, compatible model and SAE with actual weights |
| `capture_features` | Returns selected raw activations, SAE codes and residuals |
| `edit_features` | Produces a feature edit and the corresponding patched activation |
| `coalition_replacements` | Builds feature edits that retain a specified coalition |
| `evaluate_edit` | Compares baseline and edited output logits and verifies identity replay |

A real checkpoint example:

```python
from saebase.api import load_registered, capture_features, evaluate_edit, coalition_replacements

pair = load_registered(model_key="pythia70m",
                       sae_key="pythia70m_resid_sm", layer=3, device="cpu")
tokens = pair.model.tokens(prompt="Alice Emma Alice")
snapshot = capture_features(pair.model, pair.sae,
                            site=pair.site, tokens=tokens)
feature_ids = snapshot.codes.abs().topk(4).indices.tolist()
replacements = coalition_replacements(
    snapshot.codes, selected_indices=feature_ids,
    retained_indices=feature_ids[:2])
result = evaluate_edit(pair.model, pair.sae, site=pair.site,
                       tokens=tokens, replacements=replacements)
print(result.logit_delta.float().norm())
```

This example downloads real checkpoints. Use `python examples/05_public_api_offline.py` first if you only need to inspect the API behavior.

Feature editing uses `h' = h + decode(z') - decode(z)` and therefore preserves the original reconstruction residual. Full SAE reconstruction replacement is a separate intervention condition.

## Activation and code stores

The `saebase.store` package provides `ActivationIdentity`, `CodeIdentity`,
`SampleRecord`, `TokenSelection`, `ActivationWriter`, `CodeWriter`,
`ShardedReader`, `collect_activations` and `encode_from_store`.

Storage manifests associate exact model/SAE weights, tokenizer, site, selected token positions and sample identities. Previously committed rows can be reused only when the identity and integrity checks succeed. See [Activation Store](activation-store.md).

## Interventions and measurements

`saebase.interventions` exposes named `Condition` operators,
`MeasurementSample`, `ExperimentIdentity`, `BatchEngine`,
`ResultWriter`, `RunBudget` and `read_results`, plus reference steering
utilities. See [Feature Interventions](feature-interventions.md) for operator semantics and the important distinction between feature edits and full reconstruction.

## Benchmarks and results

`saebase.benchmarks` includes `BenchmarkConfig`, `MetricSpec`,
`MetricResult`, the metric registry and `build_plan`.
`saebase.benchmarks.runner.execute` executes a versioned evaluation;
`saebase.benchmarks.results.read_metrics` verifies a completed ledger.

`MetricResult` retains the sample unit, requested/valid counts, protocol,
status and source identity. It is not a universal metric score. See [Benchmarking](benchmarking.md).

## Reliability and scientific controls

`saebase.reliability` exposes `AuditConfig`, `Observation`,
`bootstrap_metric`, `paired_bootstrap`, `sample_size_curve`,
`evaluate_m3`, `evaluate_ground_truth` and report utilities.
`evaluate_m3` is a stable internal-facing function name; user-facing
documentation calls this a benchmark audit. See [Reliability Analysis](reliability.md).

CLI entry points are provided by the canonical `saebase.cli` module. The
original command names and legacy Python modules (`main`, `scripts`, `models`,
`tasks`, and `conf`) remain available. See [Packaging and Compatibility](packaging.md).

## Command-line overview

| Command | Main use |
| --- | --- |
| `saebase` | Registered task and activation-health evaluation |
| `saebase-prepare-prompts` | Prepare versioned prompt artifacts |
| `saebase-intervene` | Execute and resume batch feature interventions |
| `saebase-benchmark` | Preflight and execute versioned SAE benchmark runs |
| `saebase-audit` | Analyze completed benchmark measurements |
| `saebase-ground-truth` | Independent synthetic scientific controls |
| `saebase-release-matrix` | Reproduce real multi-checkpoint acceptance |
| `saebase-release-gate` | Check validated release evidence and distribution assets |

Public flags `--dry-run`, `--real`, `--report-only` and `--resume`
have different meanings depending on the command. Check `<command> --help`.
The `saebase-audit --m3-root` flag name is unchanged for backward compatibility.

[API quick reference](API.md) · [Examples Index](../examples/README.md) · [Documentation Home](index.md)

## Controlled input designs

`saebase.data` supplies `BinaryFactor`, `ControlledInputDesign`, `collect_controlled_input_activations` and checked sidecar manifests for complete binary input experiments. Use `saebase.integrations.collect_localization_grid(..., design=...)` or `localization_grid_from_code_store(..., manifest=...)` to hand these inputs to the original Localize estimator.

[Controlled Input Designs](controlled-inputs.md) · [Offline example](../examples/16_controlled_input_design.py)

