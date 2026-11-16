# Research Integrations

SAEBase provides common capture, feature-edit and benchmark infrastructure while preserving scientific estimators in their **original research repositories**.

## 维护科研方法接入

制定接入时，选定目标研究动作和原版方法需要的输入，再安排测量、转换与结果读取。原估计器、输入划分和返回统计含义由各自负责位置确定。缺少原版源码、输入或能力时，定位实际对象和影响并给出处理方法；保存的观测能由所属任务重新读取，并交给原方法或后续分析使用。

## SAR — Behavioral Compositionality

[SAR](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) measures non-additive effects of selected SAE features under coalition interventions.

SAEBase captures the relevant model activations and downstream readouts, constructs coalition measurements, and delegates the estimator to the original `sar.py`. The adapter does not redefine SAR or manufacture third-order estimator outputs; the original SAR reference fits first- and second-order approximations.

```bash
SAEBASE_SAR_SOURCE=/path/to/SAR-repo python -m pytest -q tests/product/test_reference_sae_contracts.py -k original_sar
```

The demo uses deterministic synthetic inputs unless a real checkpoint run is explicitly requested.

## RSS — Representation Redistribution

[RSS](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) measures structured feature redistribution under input transformations, including reproducible support and directional persistence.

SAEBase keeps clean, selection and held-out variants separate. Selection and evaluation have distinct identities; the adapter verifies that supplied text matches the declared token IDs and calls the original `rss.py` implementation.

```bash
SAEBASE_RSS_SOURCE=/path/to/RSS-repo python -m pytest -q tests/product/test_reference_sae_contracts.py -k original_rss
```

Do not interpret a held-out statistic as a selection result or treat repeated variants as independent prompt groups.

## Input Localization — controlled input scopes

[Input Localization](https://github.com/OpenSAE/SAE-Input-Localization) measures which controlled binary inputs are needed to explain frozen SAE feature activations. SAEBase provides two input routes without copying the original mathematical estimator.

- `ControlledInputDesign.from_renderer(...)` enumerates the binary factors defined for the experiment, and `collect_controlled_input_activations(...)` caches their complete, token-aligned grids with a checked state-to-sample manifest.
- `collect_localization_grid(..., design=...)` uses the public capture API to encode each input configuration in explicit Boolean bitmask order.
- `localization_grid_from_code_store(..., manifest=...)` reconstructs the same grid from identity-checked SAEBase code caches and verifies their associated activation checksums.
- `evaluate_localization(...)` calls the original `sae_localize.exact_scope_curve`; optionally it also calls the original paired plug-in estimator. It returns the optimal curve, total variance, budget and feature scope assignment.

Install the original project separately:

```bash
python -m pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git
python -m pytest -q tests/product/test_input_localization.py
python tests/product/test_controlled_inputs.py
```

See [Controlled Input Designs](controlled-inputs.md) for enumeration, bijective reindexing, and sidecar provenance. The original estimators also run through [Localize Benchmarking](localization-benchmark.md), with versioned full curves and finite-pair reliability analysis.

Input state `i` represents the Boolean factors encoded in its bits. The caller supplies a complete, correctly aligned, uniformly weighted product design; prompt labels alone cannot establish causal validity. Exact grid optimization is exponential in factor count. Paired finite-sample results are *estimates*, not post-selection certificates.

## SAEBench — native output import

Selected native SAEBench Core and Sparse Probing metrics can be imported from **already generated upstream JSON reports**. The importer verifies the native evaluation type, original source commit, model name and SAE ID/release where configured.

It does not duplicate or silently execute upstream SAEBench estimators. Numerical acceptance independently runs native SAEBench evaluations using its original repository, then checks that SAEBase imports the recorded values faithfully. Different FVU or reconstruction protocols must retain their original numerical definitions.

[Benchmarking](benchmarking.md) · [Original SAR](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) · [Original RSS](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) · [Original Localize](https://github.com/OpenSAE/SAE-Input-Localization)

## Reproducibility rules

- Record the exact external estimator source SHA-256 and source commit when available.
- Keep calibration/selection separate from held-out or test observations.
- Keep source measurement rows independent of metric summaries and reliability resampling.
- Mark missing, unsupported and undefined outputs explicitly; do not replace them with favorable numbers.
- Do not redistribute third-party estimator sources, checkpoint weights or datasets without the original rights.

[Validation and Evidence](validation.md) · [Reliability Analysis](reliability.md) · [Examples](../../../doc/code/development/first-experiment.md)


## 方法接入与完整流程

SAR的科研计算由Capture维护，Benchmark接入选择冻结的输入与读出方式。RSS将选择样本与留出样本分别收集后调用原方法。Localization读取完整受控输入网格，保存曲线与设计身份，并可从编码缓存重新读取。SAEBench执行与结果导入保留上游协议和来源。

对应实现位于 `tasks/benchmark/methods/`。受控网格的重排与缓存复用由 `tests/product/test_controlled_inputs.py` 检查，完整Localization评测、统计与结果读取由 `tests/product/test_input_localization.py` 检查。
