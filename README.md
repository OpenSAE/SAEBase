# 🧱 SAEBase: Research Infrastructure for Sparse Autoencoders

🌍 Languages: [English](README.md) | [中文](README_cn.md) &nbsp; [![CI](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml)

👉 **TL;DR.** SAEBase is an open-source toolkit for reproducible sparse autoencoder research. Load models and SAEs, capture and reuse activations, intervene on features, benchmark reconstruction and downstream effects, and audit results with verifiable provenance. The Research Lab provides a browser workflow for exploring these measurements.

<p align="center">
  <a href="#-quickstart"><b>🚀 Quickstart</b></a> &nbsp; | &nbsp;
  <a href="#-public-python-api"><b>🔬 Python API</b></a> &nbsp; | &nbsp;
  <a href="#-research-integrations"><b>🧩 Research Integrations</b></a> &nbsp; | &nbsp;
  <a href="#-documentation"><b>📚 Documentation</b></a>
</p>

🤗 **Hugging Face:** [SAEBase Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [SAEBase Evaluation Dataset](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation)

---

## 🧭 Overview

SAEBase provides a common research workflow for evaluating sparse autoencoders (SAEs) beyond reconstruction alone. It connects pretrained models, SAE dictionaries, versioned input samples, activation-level measurements, controlled feature edits, and statistical reliability checks without conflating their results.

<p align="center">
  <img src="assets/figures/saebase_workflow.svg" alt="SAEBase workflow: models and versioned inputs, activation capture and reuse, SAE encoding, reconstruction and feature interventions, benchmarking, reliability and reports" width="1080"/>
</p>

The workflow is **capture once, evaluate multiple SAEs**: a verified activation cache can be reused by different SAE checkpoints at the same model site. Feature edits can then be measured against original model outputs, and independently interpretable metrics are reported without producing a single composite SAE score.

## ✨ Highlights

- 🧠 **Model and SAE integration.** Unified checkpoint profiles, model adapters, layers, named hooks, compatibility checks, and identities for registered or custom SAEs.
- 🧭 **Controlled input designs.** Enumerate complete binary factor grids, track state-to-sample mappings, and reuse cached activations under bijective input reindexing.
- 🗃️ **Activation capture and reuse.** Sharded, resumable activation stores; lossless dense/CSR SAE codes; exact token selection, sample groups, splits, and checkpoint SHA-256 verification.
- 🎛️ **Feature interventions and steering.** Identity, ablation, scaling, clamping, addition, coalition, and full-reconstruction controls, with batch evaluation and downstream logit readouts. Feature edits can preserve the original reconstruction residual.
- 📊 **SAE benchmarking.** Separate L0, reconstruction MSE, FVU/explained variance, token-level behavioral loss, logit/KL changes, localization budgets and condition-specific metrics under reproducible configurations.
- 📈 **Reliability analysis.** Prompt-group bootstrap, paired SAE comparisons, confidence intervals, sample-size curves, ranking stability, and analytical ground-truth/negative-control tests.
- 🧩 **Research integrations.** Original [SAR](https://github.com/OpenSAE/SAE-Behavioral-Compositionality) and [RSS](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) and [Input Localization](https://github.com/OpenSAE/SAE-Input-Localization) estimators plus provenance-checked imports of native SAEBench outputs.
- ⚙️ **Reproducible experiments.** Compute preflight, exact cache identities, interrupted-run recovery, source/checkpoint hashes, and standalone JSONL/JSON, CSV, and HTML reports.

---

## 🔬 Research Lab

**Explore real SAE measurements in one place.** The Research Lab brings together a browsable results table, a feature explorer, intervention controls, SAR and RSS analyses, and reproducible reports.

| Research Lab evidence | Current recorded scope |
| :--- | :--- |
| Model / SAE coverage | 48 selected sites × 500 frozen base prompts captured in the Research Lab GPU run |
| Feature exploration | 300 curated statistical cards across six deep-dive sites; labels and source text require review |
| Feature interventions | Six real model/SAE sites with measured clean-versus-edited logits |
| SAR | 1,485 K16, 14 K8 and 1 K4 measured groups, under a separately named decoder-mediated intervention protocol |
| RSS | Six sites × 500 parent prompts; 138,000 reported unique clean/variant site-input observations under *proposed* transformations |
| Input Localization | One real 64-state factor grid, analyzed on 55 features observed on that grid |

[**Open the Research Lab on Hugging Face**](https://huggingface.co/spaces/OpenSAE/SAEBase) · [Evaluation Dataset](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation) · [Local setup](apps/research-lab/README.md) · [Results and methods](docs/research-lab/results-20261020.md) · [Reproduction and release notes](docs/releases/v0.4.0.md)

```bash
docker build -f apps/research-lab/Dockerfile -t opensae-lab:cpu .
docker run --rm -p 7860:7860 opensae-lab:cpu
```

Open http://127.0.0.1:7860 for the public **historical** evidence browser. New Research Lab GPU measurements are stored in a separately maintained private experiment archive, so they are **not automatically available** through this standalone Docker image. Public online inference remains disabled.

---

## ⚙️ Install

SAEBase supports Python 3.11–3.12. Install the lightweight core from a source checkout to run the offline tutorials:

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .
saebase --list
```

<details>
<summary>📦 Other installation options</summary>

To run experiments with real pretrained models and SAE checkpoints, install the optional analysis dependencies:

```bash
python -m pip install -e ".[analysis]"
```

For tests and packaging, install `.[dev]`. Optional experimental GPU generation integrations use `.[generation]`; GPU vLLM/NNsight acceleration is not part of the default verified CPU workflow. Versioned wheels and source archives are available under [GitHub Releases](https://github.com/OpenSAE/SAEBase/releases); this project does not claim a PyPI upload.

</details>

## 🚀 Quickstart

**1. Explore the public API without downloading checkpoints:**

```bash
python examples/quickstart/public_api_offline.py
```

This self-contained example uses a small synthetic model and SAE to demonstrate activation capture, feature editing, identity checks, and downstream logit comparisons. Its numerical outputs are not research findings.

To generate a complete offline benchmark and reliability report that you can open in a browser:

```bash
python examples/quickstart/first_experiment.py --output runs/first-experiment
```

Follow the [first experiment walkthrough](docs/first-experiment.md) to inspect the generated metrics, cache and paired audit.

**2. Inspect a benchmark before committing compute:**

```bash
saebase-benchmark --config conf/benchmark/p0_small.json --dry-run
```

This validates the experiment, its measurement count, and the execution dependencies without loading model weights.

**3. Run the same configuration using public checkpoints:**

```bash
saebase-benchmark --config conf/benchmark/p0_small.json --real
saebase-benchmark --config conf/benchmark/p0_small.json --report-only
```

Real runs download the requested weights and produce `metrics.jsonl`, `summary.csv`, `report.html`, configuration, and provenance under `runs/<run_id>/`. This tiny fixed-token example is an engineering smoke test, not a study dataset. For prompt preparation and the original activation-health command, see [Getting Started](docs/getting-started.md).

---

## 🧠 Model and SAE Support

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
**Verified checkpoints (2026-10-16).** Across **18** model-SAE configurations (270M-9B), **48/48** selected model/layer sites passed real-checkpoint compatibility tests, **48/48** completed benchmark evaluations on 64 held-out prompt groups, and **48/48** completed reliability audits. See the [verified model and SAE matrix](docs/verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->


The earlier [v0.2.0 release](https://github.com/OpenSAE/SAEBase/releases/tag/v0.2.0) verified two SAE checkpoints each on Pythia-70M and GPT-2 Small. These smaller compatibility tests are preserved as historical evidence. For current checkpoint identities, evaluated layers, and individual reports, see [Model and SAE Support](docs/model-support.md).

---

## 🗃️ Activation Capture and Storage

Capture model activations once and reuse the stored rows when evaluating multiple SAEs at the same hook. The store records the model and tokenizer identity, selected token positions, `sample_id`, `prompt_group_id`, split, and source activation checksums. SAE codes are stored separately in lossless dense or CSR representations.

```bash
python examples/activations/activation_store.py
```

Readers verify manifests and shards, and interrupted runs can resume instead of repeating committed captures. Reuse is conditional on exact model/site/input identities; it is never inferred solely from a file name. See [Activation Store](docs/activation-store.md).

---

## ⚡ Feature Interventions and Steering

SAEBase supports batched identity, ablation, scaling, clamping, addition, coalition, and reconstruction conditions. Feature-level edits can use residual-preserving patching:

```text
edited_activation = original_activation + decode(edited_codes) - decode(original_codes)
```

A **full SAE reconstruction replacement** is a separate experimental control and must not be confused with residual-preserving feature edits. Results distinguish interventions by condition and include downstream logits, KL divergence, target-token effects (when defined), and activation changes.

```bash
saebase-intervene --config conf/interventions/m2_pythia_example.json --dry-run
python examples/interventions/batch_interventions.py
```

Real checkpoint execution requires `--real`. Optional GPU acceleration delivered **1.28×** faster Qwen3-1.7B feature-intervention evaluation (unchanged metrics), plus KV-cached generation with live SAE edits (**89 tokens/s**) or vLLM SAE decoder-direction steering (**158 tokens/s**) on an RTX 5070 at 128 tokens. The generation modes use distinct intervention protocols; the original full-prefix reference remains available. [GPU results](docs/performance/workplan11_final_results.md) · [Feature Interventions](docs/feature-interventions.md).

---

## 📊 SAE Benchmarking

The benchmark runner expands model/SAE/metric configurations into a deterministic plan, reuses identity-verified activation/code and intervention caches, and emits each metric with its own units, protocol, sample counts, status, and source provenance.

| Metric family | Examples | Interpretation |
| :--- | :--- | :--- |
| **Core representations** | L0, reconstruction MSE, FVU, explained variance | SAE sparsity and reconstruction properties; denominator/protocol definitions matter |
| **Behavioral reconstruction** | Target-token negative log-likelihood, loss recovered | Full reconstruction relative to the unmodified model and a measured zero-site ablation |
| **Feature interventions** | Logit differences, KL, activation delta, target-token changes | Downstream effects for individual, named edit conditions |
| **External research metrics** | SAR, RSS, native SAEBench Core and Sparse Probing | Original estimators or original externally generated results, with source and protocol checks |

```bash
saebase-benchmark --config conf/benchmark/p0_small.json --dry-run
python examples/benchmarking/benchmark_preflight.py
```

SAEBase's per-token FVU is not automatically numerically interchangeable with SAEBench's global reconstruction protocol. Native SAEBench metrics are **imported from an upstream-generated JSON result**; the benchmark importer does not silently rerun upstream evaluations. See [Benchmarking](docs/benchmarking.md).

---

## 📈 Reliability and Reproducibility

Reliability audits reuse already computed measurements and resample complete **prompt groups**, keeping tokens or prompt variants within the same independent group. They report paired confidence intervals, subsample-size sensitivity, ranking stability for metrics with a meaningful direction, and explicitly undefined or unsupported comparisons.

```bash
python examples/benchmarking/reliability_bootstrap.py
python examples/benchmarking/ground_truth_controls.py
```

To audit a completed benchmark, run `saebase-audit --m3-root runs/<run_id> --config conf/reliability/m4_pythia_smoke.json --run --sae-only`. Use a matching multi-prompt benchmark run for this audit. The legacy `--m3-root` flag is retained for CLI compatibility; it refers to the benchmark's output directory, not a required workflow stage.

Analytical controls include known factor dictionaries, coalition interaction orders, RSS support/direction persistence, controllable degradation, and negative controls. These synthetic tests **do not certify the interpretability of a real SAE**. For reconstruction uncertainty, `--sae-only` can decode verified cached codes without re-running the language model. See [Reliability Analysis](docs/reliability.md).

---

## 🧩 Research Integrations

SAEBase supplies shared model operations; original research projects retain their own metric definitions and scientific protocols.

| Project | SAEBase integration | Example |
| :--- | :--- | :--- |
| 🧩 **[SAR — Behavioral Compositionality](https://github.com/OpenSAE/SAE-Behavioral-Compositionality)** | Capture feature contributions, apply coalition edits, call the unchanged SAR estimator. | [SAR example](examples/research_integrations/sar.py) |
| 🔀 **[RSS — Representation Redistribution](https://github.com/OpenSAE/Redistribution-Sensitivity-Score)** | Capture disjoint clean/selection/held-out observations, call the unchanged RSS estimator. | [RSS example](examples/research_integrations/rss.py) |
| 🧭 **[Input Localization](https://github.com/OpenSAE/SAE-Input-Localization)** | Capture or reuse complete, ordered Boolean input grids of SAE codes; delegate optimal scope curves to the original Localize package. | [Localize example](examples/research_integrations/input_localization.py) |
| 📊 **[SAEBench](https://github.com/adamkarvonen/SAEBench)** | Import and verify selected native Core and Sparse Probing result fields against the original model/SAE identity and upstream commit. | [Benchmark documentation](docs/benchmarking.md) |

```bash
python examples/research_integrations/sar.py --paper-dir /path/to/SAR-repo
python examples/research_integrations/rss.py --paper-dir /path/to/RSS-repo
```

The examples use deterministic synthetic inputs by default; real checkpoint use is opt-in. For Localize, install the independent `sae-input-localization` package first and run `python examples/research_integrations/input_localization.py`. Try the cached, reindexed grid via `python examples/research_integrations/controlled_inputs.py` (no model downloads). See [Controlled Input Designs](docs/controlled-inputs.md), [Localize Benchmarking](docs/localization-benchmark.md), and [Research Integrations](docs/research-integrations.md).

---

## 🔬 Public Python API

The stable `saebase.api` includes `ModelAdapter`, `ActivationSite`, `load_registered`, `capture_features`, `edit_features`, and `evaluate_edit`. Research workflows can also use `saebase.store`, `saebase.interventions`, `saebase.benchmarks`, and `saebase.reliability` directly.

```python
from saebase.api import load_registered, capture_features, evaluate_edit

pair = load_registered(model_key="pythia70m",
                       sae_key="pythia70m_resid_sm", layer=3, device="cpu")
snapshot = capture_features(pair.model, pair.sae,
                            site=pair.site, prompt="Alice Emma Alice")
# See examples/interventions/pythia_feature_edit.py for a complete feature-edit example.
```

The code above downloads real weights. Prefer the offline example for a no-network start. See the complete [Python API Guide](docs/python-api.md), [API Reference](docs/API.md), and [Examples Index](examples/README.md).

---

## 🏗️ Architecture

| Component | Responsibility |
| :--- | :--- |
| [`saebase/api/`](saebase/api/) | Model adapters, capture/edit functions, activation sites and SAE identity |
| [`saebase/store/`](saebase/store/) | Identity-bound activation/code shards, storage readers and resumable capture |
| [`saebase/interventions/`](saebase/interventions/) | Batch conditions, measurement journals, patch execution and steering |
| [`saebase/benchmarks/`](saebase/benchmarks/) | Metric registry, experiment planning, cached execution and reports |
| [`saebase/reliability/`](saebase/reliability/) | Grouped bootstrap, paired comparisons, ground truth and audits |
| [`saebase/integrations/`](saebase/integrations/) | Original SAR/RSS/Localize interface adapters and controlled-grid benchmark analysis |
| [`saebase/sae/`](saebase/sae/) | SAE profiles, loading and reconstruction diagnostics |
| [`saebase/data/`](saebase/data/) | Versioned prompt data, complete controlled input designs, manifests and compatibility contracts |
| [`models/`](models/) · [`conf/`](conf/) | Model plugins and versioned experiment configurations |
| [`scripts/`](scripts/) · [`examples/`](examples/) | CLIs, executable demonstrations and validation utilities |

## ✅ Validation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

| Area | Publicly validated scope |
| :--- | :--- |
| Core APIs and packaging | Python 3.11/3.12 CI, wheel/sdist construction and installation checks |
| Real model/SAE interoperability | 18 model-SAE configurations, 48 evaluated model/layer sites (270M-9B), with [per-site reports](docs/verified-model-matrix.md) |
| Storage, interventions and reports | Identity-aware cache reuse, condition-specific evaluation, resume and integrity checks |
| Research estimators | Original SAR/RSS estimator tests; Localize delegation contracts and pinned native SAEBench Core/Sparse Probing checks |
| Reliability | Prompt-group bootstrap, paired comparisons and independently generated analytical controls |
| Experimental backends | vLLM/NNsight remain optional; this release does not claim GPU throughput validation |

The 48-site evaluation uses 64 held-out prompt groups per site, drawn from controlled templates. It verifies benchmark execution and reliability under this protocol, not general SAE quality or interpretability. The model and SAE weights remain with their original publishers. See [Validation and Reproducibility](docs/validation.md), the [Research Lab evidence summary](docs/research-lab/results-20261020.md), and the [v0.4.0 release notes](docs/releases/v0.4.0.md).

---

## 📚 Documentation

[Documentation Index](docs/index.md) · [Getting Started](docs/getting-started.md) · [Python API](docs/python-api.md) · [Examples](examples/README.md) · [Model Support](docs/model-support.md)

[Activation Store](docs/activation-store.md) · [Feature Interventions](docs/feature-interventions.md) · [Benchmarking](docs/benchmarking.md) · [Reliability](docs/reliability.md) · [Research Integrations](docs/research-integrations.md) · [Extensions](docs/extensions.md)

[Validation](docs/validation.md) · [Changelog](CHANGELOG.md) · [v0.4.0 Release Notes](docs/releases/v0.4.0.md) · [Provenance](docs/PROVENANCE.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0 License](LICENSE)

<p align="center">
  <b>Open research infrastructure for faithful representation analysis.</b><br>
  <a href="https://github.com/OpenSAE">OpenSAE</a> · <a href="https://github.com/OpenSAE/SAEBase/issues">Issues & contributions</a>
</p>
