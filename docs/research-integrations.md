# Research Integrations

SAEBase provides common capture, feature-edit and benchmark infrastructure while preserving scientific estimators in their **original research repositories**.

## SAR — Behavioral Compositionality

[SAR](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) measures non-additive effects of selected SAE features under coalition interventions.

SAEBase captures the relevant model activations and downstream readouts, constructs coalition measurements, and delegates the estimator to the original `sar.py`. The adapter does not redefine SAR or manufacture third-order estimator outputs; the original SAR reference fits first- and second-order approximations.

```bash
python examples/07_sar_via_saebase.py --paper-dir /path/to/SAR-repo
```

The demo uses deterministic synthetic inputs unless a real checkpoint run is explicitly requested.

## RSS — Representation Redistribution

[RSS](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) measures structured feature redistribution under input transformations, including reproducible support and directional persistence.

SAEBase keeps clean, selection and held-out variants separate. Selection and evaluation have distinct identities; the adapter verifies that supplied text matches the declared token IDs and calls the original `rss.py` implementation.

```bash
python examples/08_rss_via_saebase.py --paper-dir /path/to/RSS-repo
```

Do not interpret a held-out statistic as a selection result or treat repeated variants as independent prompt groups.

## SAEBench — native output import

Selected native SAEBench Core and Sparse Probing metrics can be imported from **already generated upstream JSON reports**. The importer verifies the native evaluation type, original source commit, model name and SAE ID/release where configured.

It does not duplicate or silently execute upstream SAEBench estimators. Numerical acceptance independently runs native SAEBench evaluations using its original repository, then checks that SAEBase imports the recorded values faithfully. Different FVU or reconstruction protocols must retain their original numerical definitions.

[Benchmarking](benchmarking.md) · [Original SAR](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) · [Original RSS](https://github.com/OpenSAE/Redistribution-Sensitivity-Score)

## Reproducibility rules

- Record the exact external estimator source SHA-256 and source commit when available.
- Keep calibration/selection separate from held-out or test observations.
- Keep source measurement rows independent of metric summaries and reliability resampling.
- Mark missing, unsupported and undefined outputs explicitly; do not replace them with favorable numbers.
- Do not redistribute third-party estimator sources, checkpoint weights or datasets without the original rights.

[Validation and Evidence](validation.md) · [Reliability Analysis](reliability.md) · [Examples](../examples/README.md)
