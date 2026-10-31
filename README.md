# 🧱 SAEBase: SAE Research Tools and Agent Workflows

🌍 Languages: [English](README.md) | [中文](README_cn.md) &nbsp; [![CI](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml)

**Software version:** [v0.6.5 — Auditable Agent Research Workflows](harness/github_development_doc/releases/v0.6.5.md) · [Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [Python API](harness/github_development_doc/python-api.md)

**SAEBase** helps researchers study sparse autoencoders (SAEs) in language models. Explore features, test how feature edits affect model behavior, and build reproducible experiments. Its Research Harness also helps coding agents carry out scoped work with traceable evidence and independent review.

<p align="center">
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> &nbsp; | &nbsp;
  <a href="#-research-harness"><b>🧭 Harness</b></a> &nbsp; | &nbsp;
  <a href="#-research-workflows"><b>🔬 Workflows</b></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/spaces/OpenSAE/SAEBase"><b>🤗 Research Lab</b></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation"><b>🤗 Evaluation Dataset</b></a>
</p>

---

## 🧭 Research Harness

**Build SAE experiments with AI agents while keeping decisions and evidence under your control.** The SAEBase Harness gives coding agents a clear task scope, connects their work to real research code, and supports independent checks of what actually ran.

| Research step | What the Harness helps you do |
| --- | --- |
| **Plan** | Define the objective, what an agent may change, and how the result will be reviewed. |
| **Run** | Work in an isolated copy of the project, using SAEBase's actual research interfaces and saved experiment outputs. |
| **Verify** | Check behavior independently, including expected results, failures, and compatibility with existing workflows. |
| **Trace** | Connect a result to its inputs, code, models, commands, and output files. |
| **Improve** | Distinguish issues in the implementation, environment, and verification process before the next attempt. |

**Atlas** provides five connected views of the project. [Core](https://atheavenma.github.io/opensae/saebase/atlas/core.html) explains the architecture; [Knowledge](https://atheavenma.github.io/opensae/saebase/atlas/knowledge.html) organizes research questions; [Code](https://atheavenma.github.io/opensae/saebase/atlas/code.html) links to source implementations; [Environment](https://atheavenma.github.io/opensae/saebase/atlas/environment.html) describes computing resources; and [Context](https://atheavenma.github.io/opensae/saebase/atlas/context.html) explains terminology.

[Explore the Harness](harness/README.md) · [Agent collaboration guide](harness/core/AGENT_FLOW.md) · [Research workflow documentation](harness/github_development_doc/index.md)

## 🧭 Overview

SAEBase connects pretrained models, SAE dictionaries, activation measurements, controlled feature interventions, benchmarks, and reliability analysis in one reusable workflow.

<p align="center">
  <img src="harness/github_assets/figures/saebase_workflow.svg" alt="SAEBase: models and inputs, reusable activation capture, SAE feature edits, benchmarking and reports" width="1080"/>
</p>

**Capture once, evaluate multiple SAEs.** Verified activations can be reused across compatible SAE checkpoints at the same model site, without repeating the language-model forward pass.

## ✨ Highlights

- **Explore SAE features.** Capture model activations, inspect sparse feature codes, and compare reconstructions.
- **Test interventions.** Ablate, scale, add, or replace selected features and measure the effect on model outputs.
- **Reuse measurements.** Cache activations, resume interrupted work, and compare compatible model and SAE checkpoints.
- **Evaluate results.** Run reconstruction and behavior measurements with controls and reproducible reports.
- **Work with agents.** Give coding agents scoped research tasks and independently review their implementations and outputs.
- **Extend your research.** Connect datasets, models, tasks, and established research methods through the existing interfaces.

## 🧪 Research Lab

The [Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) lets you browse recorded SAE research results, feature measurements, intervention examples, and reports. Recorded results are distinguished from illustrative examples and new computations. Fresh model inference requires a separately configured runtime and appropriate weights.

[Open the Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [Evaluation Dataset](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation) · [Measured results and methods](harness/github_development_doc/research-lab/results-20261020.md)

## 🚀 Quick Start

### 1. Install and run a complete example

Use Python 3.11 or 3.12. The first examples run on CPU without downloading model checkpoints.

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .

python saebase/examples/quickstart/public_api_offline.py
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
```

The first Python example demonstrates activation capture and a feature edit. The second example evaluates two small synthetic SAEs and writes `report.html`, `summary.csv` and a reliability report. It prints the exact output paths so you can open the HTML report in a browser.

### 2. Try a real model and SAE

Install the optional model dependencies, then run the provided Pythia-70M example. This downloads public model and SAE weights.

```bash
python -m pip install -e ".[analysis]"
python saebase/examples/interventions/pythia_feature_edit.py
```

Use the same operations directly in Python:

```python
from saebase.api import (
    load_registered, capture_features, coalition_replacements, evaluate_edit,
)

bundle = load_registered(
    model_key="pythia70m", sae_key="pythia70m_resid_sm", layer=3, device="cpu"
)
tokens = bundle.model.tokens(prompt="Alice Emma Alice")
snapshot = capture_features(bundle.model, bundle.sae, site=bundle.site, tokens=tokens)
features = snapshot.codes.abs().topk(4).indices.tolist()
replacement = coalition_replacements(
    snapshot.codes, selected_indices=features, retained_indices=features[:2]
)
effect = evaluate_edit(
    bundle.model, bundle.sae, site=bundle.site, tokens=tokens, replacements=replacement
)
print(float(effect.logit_delta.float().norm()))
```

This removes two selected feature contributions and reports the change in next-token logits. For other checkpoints, token selection and editing operations, see the [Python API](harness/github_development_doc/python-api.md) and [model registry](harness/github_development_doc/model-support.md).

### 3. Work with a coding agent

Open this source checkout in your coding-agent environment and start with the [Harness guide](harness/README.md). Describe the research change you want, review its allowed scope, and have a separate reviewer check the real run and saved evidence before accepting the result. The complete Harness is available in the **source repository**; installing the Python wheel alone provides the SAEBase package rather than the full agent workspace.

## 🔬 Research Workflows

SAEBase separates what gets measured from how the results are analyzed.

| Goal | Runnable example | Guide |
| --- | --- | --- |
| Capture and reuse activations | [Activation store](saebase/examples/activations/activation_store.py) | [Capture and storage](harness/github_development_doc/activation-store.md) |
| Test feature interventions | [Batch interventions](saebase/examples/interventions/batch_interventions.py) | [Feature editing and steering](harness/github_development_doc/feature-interventions.md) |
| Benchmark and audit results | [First experiment](saebase/examples/quickstart/first_experiment.py) | [Benchmarking](harness/github_development_doc/benchmarking.md) · [Reliability](harness/github_development_doc/reliability.md) |
| Integrate research methods | [Research examples](saebase/examples/research_integrations/README.md) | [SAR, RSS, Localization and SAEBench](harness/github_development_doc/research-integrations.md) |

Feature edits can preserve the part of an activation not reconstructed by the SAE:

```text
edited_activation = original_activation + decode(edited_codes) - decode(original_codes)
```

Full SAE reconstruction is a separate experimental control. Reliability estimates resample independent prompt groups rather than treating related token variants as independent observations.

### ⚡ Performance

SAEBase includes optional optimizations for repeated model evaluations and feature interventions. Results and hardware conditions are documented separately in the [performance report](harness/github_development_doc/performance/workplan11_final_results.md).

## 🧠 Model and SAE Support

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
**Verified checkpoints (2026-10-16).** Across **18** model-SAE configurations (270M-9B), **48/48** selected model/layer sites passed real-checkpoint compatibility tests, **48/48** completed benchmark evaluations on 64 held-out prompt groups, and **48/48** completed reliability audits. See the [verified model and SAE matrix](harness/github_development_doc/verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->

[Model/SAE matrix](harness/github_development_doc/verified-model-matrix.md) · [Validation methods](harness/github_development_doc/validation.md)

## 📚 Documentation

[Getting Started](harness/github_development_doc/getting-started.md) · [Python API](harness/github_development_doc/python-api.md) · [Examples](saebase/examples/README.md) · [Research Lab](saebase/research_lab/README.md) · [Full documentation](harness/github_development_doc/index.md)

Explore the code: [SAEBase API](saebase/) · [Models](models/) · [Datasets](datasets/) · [Tasks](tasks/) · [Harness](harness/README.md) · [Configuration](conf/) · [Repository Guide](harness/github_development_doc/repository-guide.md).

[Contributing](.github/CONTRIBUTING.md) · [Changelog](harness/github_development_doc/CHANGELOG.md) · [Citation](CITATION.cff) · [Apache-2.0 License](LICENSE) · [Security](.github/SECURITY.md)

---

<p align="center">
  <em>Building open foundations for understanding and evaluating LLMs.</em><br>
  <a href="https://github.com/OpenSAE">OpenSAE</a> · <a href="https://github.com/OpenSAE/SAEBase/issues">Issues</a>
</p>
