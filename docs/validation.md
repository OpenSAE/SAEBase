# Validation and Reproducibility

SAEBase distinguishes **what can be configured**, **what has run with real public weights**, and **what its scientific results establish**. A successful registry lookup or a synthetic smoke test is not proof of a real checkpoint evaluation.

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
## Verified model and SAE evaluations

**18** selected model-SAE configurations, **48/48** real-checkpoint compatibility tests, **48/48** benchmark evaluations on 64 held-out prompt groups, and **48/48** reliability audits (2026-10-16). See the [per-layer reports](verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

## Earlier software validation (v0.2.0)

| Validation area | Publicly exercised evidence |
| --- | --- |
| Installation and core API | Python 3.11/3.12 offline tests, wheel/sdist validation, installed CLI from outside the source tree |
| Real model/SAE matrix | Pythia-70M deduped and GPT-2 Small, two different SAE weight checkpoints per model |
| Activation and code stores | Exact tokenizer/model/site/SAE provenance, resume, SHA-256 verification, sample and token alignment |
| Feature interventions | Identity/reconstruction and feature conditions, downstream effects, replay and condition-specific measurement IDs |
| Benchmarks | Deterministic planning, separated metrics, external estimator provenance and JSONL/CSV/HTML reports |
| Reliability | Grouped prompt bootstrap, paired comparisons, independently generated synthetic factor and interaction truths, negative controls |
| Original research estimators | SAR/RSS original-source checks; native SAEBench Core and Sparse Probing evaluations and import parity |
| Distribution | Verified GitHub `v0.2.0` wheel, source archive, release evidence, checksums and annotated tag |

The historical `v0.2.0` multi-checkpoint matrix used **six fixed prompts per model family** and at least two independent SAE checkpoint SHA-256 identities per model. This is sufficient to exercise the software pipeline, not to estimate population-level SAE quality with narrow confidence intervals.

## Verification examples

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python scripts/verify_distribution.py dist
saebase-release-matrix --config conf/release/m5_matrix.json --dry-run
```

The real matrix additionally requires `.[analysis]`, external model/SAE downloads and sufficient RAM:

```bash
saebase-release-matrix --config conf/release/m5_matrix.json --real --device cpu
saebase-release-matrix --config conf/release/m5_matrix.json --report-only
```

Benchmark runs and audits use versioned manifests and measured checkpoint SHA-256 identities. Restarts do not assume cached data are valid; they validate exact model/site/input identities. External source changes invalidate reports rather than silently altering scientific protocols.

## Reproducibility boundaries

- An SAE's lower reconstruction MSE does not, by itself, establish greater behavioral faithfulness or interpretability.
- Different FVU denominators, split definitions, readout units or SAEBench protocols are not interchangeable.
- Prompt variants are clustered under a common `prompt_group_id` for reliability estimates.
- Synthetic ground-truth experiments verify the methods' behavior against constructed known effects, not against a real model's ground truth.
- GPU throughput for optional vLLM/NNsight backends has not been established by checkpoint compatibility testing; it requires separate performance measurements.
- Some registered models or upstream checkpoints may require authentication or additional hardware.
- Third-party model/SAE weights are not bundled in SAEBase's distribution.

## Evidence and releases

The [v0.3.0 software release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.3.0) provides the verified Python wheel, source archive, release evidence and checksums. [Checkpoint and layer results](verified-model-matrix.md) are backed by independently downloadable [compact reports and original run archives](stage-d-evidence-archive.md). The [GitHub Actions history](https://github.com/OpenSAE/SAEBase/actions) records software tests and release verification. Earlier [v0.2.0 artifacts](https://github.com/OpenSAE/SAEBase/releases/tag/v0.2.0) remain available for historical reproduction.

[Checkpoint Acceptance Protocol](reference-checkpoint-validation.md) · [Detailed Release Matrix](checkpoint-validation.md) · [v0.3.0 Release Notes](releases/v0.3.0.md) · [Provenance](PROVENANCE.md)
