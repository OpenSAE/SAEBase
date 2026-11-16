# Python API and Command-Line Interfaces

SAEBase exposes distinct APIs for activation inspection, storage, intervention experiments, benchmarking, and reliability analysis. The original Python names and CLI commands are retained for compatibility.

## 维护数据与科研调用接口

从具体调用者怎样准备输入、选择资源和使用结果确定接口变化。已有模型、数据与任务事实沿调用传递，保存与读取使用对应组件的正式能力。错误与部分完成的信息支持定位实际对象并继续处理；通过调用者完成预定工作来检查接口。

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

This example downloads real checkpoints. Use `python scripts/examples/first_experiment.py --dry-run` first if you only need to inspect the API behavior.

Feature editing uses `h' = h + decode(z') - decode(z)` and therefore preserves the original reconstruction residual. Full SAE reconstruction replacement is a separate intervention condition.

## Activation and code stores

The `saebase.store` package provides `ActivationIdentity`, `CodeIdentity`,
`SampleRecord`, `TokenSelection`, `ActivationWriter`, `CodeWriter`,
`ShardedReader`, `collect_activations` and `encode_from_store`.

Storage manifests associate exact model/SAE weights, tokenizer, site, selected token positions and sample identities. Previously committed rows can be reused only when the identity and integrity checks succeed. See [Activation Store](activation-store.md).

## Interventions and measurements

`tasks.intervene.backend` exposes named `Condition` operators,
`MeasurementSample`, `ExperimentIdentity`, `BatchEngine`,
`ResultWriter`, `RunBudget` and `read_results`, plus reference steering
utilities. See [Feature Interventions](feature-interventions.md) for operator semantics and the important distinction between feature edits and full reconstruction.

## Benchmarks and results

`tasks.benchmark.execution` includes `BenchmarkConfig`, `MetricSpec`,
`MetricResult`, the metric registry and `build_plan`.
`tasks.benchmark.execution.runner.execute` executes a versioned evaluation;
`tasks.benchmark.results.metrics.read_metrics` verifies a completed ledger.

`MetricResult` retains the sample unit, requested/valid counts, protocol,
status and source identity. It is not a universal metric score. See [Benchmarking](benchmarking.md).

## Reliability and scientific controls

`tasks.benchmark.reliability` exposes `AuditConfig`, `Observation`,
`bootstrap_metric`, `paired_bootstrap`, `sample_size_curve`,
`evaluate_benchmark`, `evaluate_ground_truth` and report utilities.
`evaluate_benchmark` is a stable internal-facing function name; user-facing
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
Use `saebase-audit --results-dir` to select a saved benchmark directory.

[API quick reference](API.md) · [Examples Index](../../../doc/code/development/first-experiment.md) · [Documentation Home](index.md)

## Controlled input designs

`saebase.datasets` supplies `BinaryFactor`, `ControlledInputDesign`, `collect_controlled_input_activations` and checked sidecar manifests for complete binary input experiments. Use `tasks.benchmark.methods.collect_localization_grid(..., design=...)` or `localization_grid_from_code_store(..., manifest=...)` to hand these inputs to the original Localize estimator.

[Controlled Input Designs](controlled-inputs.md) · [Offline example](../../../tests/product/test_controlled_inputs.py)


## 按位点捕获并编辑特征

`load_registered` 返回模型、SAE和位点身份。`capture_features` 接收实际输入与明确位置，返回激活、编码、重构及残差；`evaluate_edit` 在该位置编辑特征并比较下游输出。

```python
from saebase.api import load_registered, capture_features, evaluate_edit
pair = load_registered(model_key="pythia70m", sae_key="pythia70m_resid_sm", layer=3, device="cpu")
snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt="Alice Emma Alice")
feature = int(snapshot.codes.abs().argmax())
result = evaluate_edit(pair.model, pair.sae, site=pair.site, prompt="Alice Emma Alice", replacements={feature: 0.0})
print(pair.identity.to_dict(), result.logit_delta)
```

该片段使用实际模型权重；软件的离线数值示例使用[完整实验演示](../../../scripts/examples/first_experiment.py)。非末尾位置、残差保留和编辑输出的检查归对应产品测试。
