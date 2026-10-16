# Validation and Reproducibility

SAEBase distinguishes **what can be configured**, **what has run with real public weights**, and **what its scientific results establish**. A successful registry lookup or a synthetic smoke test is not proof of a real checkpoint evaluation.

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
## Current main Stage D evidence

**48/48** real pair acceptances, **48/48** 64-group engineering benchmark runs and **48/48** reliability audits across **18** profiles (2026-10-16). See the [per-site reports](verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

## `v0.2.0` release acceptance (historical)

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
- GPU vLLM/NNsight performance is an optional, separate claim, and has not been established by the CPU real-checkpoint matrix.
- Some registered models or upstream checkpoints may require authentication or additional hardware.
- Third-party model/SAE weights are not bundled in SAEBase's distribution.

## Evidence and releases

The [GitHub Actions runs](https://github.com/OpenSAE/SAEBase/actions) document standard package CI, real checkpoint tests and source-pinned integration acceptance. [v0.2.0 GitHub Release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.2.0) supplies the wheel/sdist, `SHA256SUMS` and `release-evidence.json` for independent artifact checks.

[Checkpoint Acceptance Protocol](reference-checkpoint-validation.md) · [Detailed Release Matrix](checkpoint-validation.md) · [Release Notes](releases/v0.2.0.md) · [Provenance](PROVENANCE.md)
