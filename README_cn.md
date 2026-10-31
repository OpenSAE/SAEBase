# 🧱 SAEBase：SAE 科研工具与 Agent 工作流

🌍 语言：[English](README.md) | [中文](README_cn.md) &nbsp; [![CI](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml)

**软件版本：** [v0.6.5 — Auditable Agent Research Workflows](harness/github_development_doc/releases/v0.6.5.md) · [Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [Python API](harness/github_development_doc/python-api.md)

**SAEBase** 帮助研究者理解和评估语言模型中的稀疏自编码器（SAE），探索特征、测量特征干预对模型行为的影响，并组织可复现的实验。Research Harness 进一步支持编程 Agent 在明确范围内执行任务，让结果可追溯、可独立验证。

<p align="center">
  <a href="#-快速开始"><b>🚀 快速开始</b></a> &nbsp; | &nbsp;
  <a href="#-research-harness"><b>🧭 Harness</b></a> &nbsp; | &nbsp;
  <a href="#-研究工作流"><b>🔬 研究工作流</b></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/spaces/OpenSAE/SAEBase"><b>🤗 Research Lab</b></a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation"><b>🤗 Evaluation Dataset</b></a>
</p>

---

## 🧭 Research Harness

**让 AI Agent 参与 SAE 科研工程，同时让研究者掌握目标、决策和验证过程。** SAEBase Harness 为 Agent 提供明确的任务范围，让代码修改连接到真实科研接口，并支持独立检查实际执行结果。

| 科研环节 | Harness 提供的帮助 |
| --- | --- |
| **制定任务** | 明确研究目标、允许修改的范围和验收条件。 |
| **执行实验** | 在隔离的项目副本中调用真实 SAEBase 接口，保存实验结果。 |
| **独立验证** | 检查功能行为、异常输入和已有工作流的兼容性。 |
| **追踪证据** | 将结果与输入、代码、模型、执行命令和产物关联。 |
| **持续改进** | 区分实现、环境与验证过程的问题，支持下一轮修复。 |

**Atlas** 提供五个相互关联的研究视图。[Core](https://opensae.github.io/SAEBase/atlas/core.html) 介绍项目架构，[Knowledge](https://opensae.github.io/SAEBase/atlas/knowledge.html) 组织研究问题，[Code](https://opensae.github.io/SAEBase/atlas/code.html) 连接真实源码，[Environment](https://opensae.github.io/SAEBase/atlas/environment.html) 介绍计算资源，[Context](https://opensae.github.io/SAEBase/atlas/context.html) 解释研究术语。

[了解 Harness](harness/README.md) · [Agent 协作指南](harness/core/AGENT_FLOW.md) · [科研流程文档](harness/github_development_doc/index.md)

## 🧭 概览

SAEBase 将预训练模型、SAE 字典、激活测量、受控特征干预、基准评测和可靠性分析连接为可复用的研究流程。

<p align="center">
  <img src="harness/github_assets/figures/saebase_workflow.svg" alt="SAEBase 模型、激活提取、SAE 特征干预、评测与报告流程" width="1080"/>
</p>

**一次提取，多 SAE 评估。** 在相同的模型层位上，通过身份校验的激活缓存可以由兼容的 SAE checkpoint 复用，避免重复执行模型前向。

## ✨ 核心能力

- **探索 SAE 特征。** 提取模型激活，观察稀疏特征编码，分析重构结果。
- **测试特征干预。** 消融、缩放、添加或替换特征，测量模型输出的变化。
- **复用实验结果。** 缓存激活、中断后恢复，并比较兼容的模型和 SAE。
- **评估科研结果。** 使用重构和行为指标、实验对照及可复现报告。
- **与 Agent 协作。** 在明确的任务范围内开发和实验，并独立验证实现与产物。
- **扩展研究流程。** 通过现有接口接入数据集、模型、任务及研究方法。

## 🧪 Research Lab

[Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) 提供可浏览的 SAE 研究结果、特征测量、干预案例和实验报告，并区分已记录的真实结果、演示案例与新计算。执行新输入的真实模型推理，需要单独配置运行环境及相应权重。

[打开 Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [评估数据集](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation) · [实验结果与方法](harness/github_development_doc/research-lab/results-20261020.md)

## 🚀 快速开始

### 1. 安装并运行完整示例

需要 Python 3.11 或 3.12。下面两个示例均可在 CPU 上运行，无须下载预训练模型。

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .

python saebase/examples/quickstart/public_api_offline.py
python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment
```

第一个示例展示激活提取和 SAE 特征编辑。第二个示例使用两个小型合成 SAE，生成 `report.html`、`summary.csv` 和可靠性分析报告。命令会打印具体输出路径，可以直接用浏览器打开 HTML 报告。

### 2. 使用真实模型和 SAE

安装可选的模型分析依赖，然后运行 Pythia-70M 示例。该示例需要下载公开的模型与 SAE 权重。

```bash
python -m pip install -e ".[analysis]"
python saebase/examples/interventions/pythia_feature_edit.py
```

也可以直接使用 Python API：

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

这段代码保留两个特征贡献，移除另两个，并输出下一个 token 的 logits 变化幅度。其他模型、token 位置和编辑操作参见 [Python API](harness/github_development_doc/python-api.md) 和 [模型注册表](harness/github_development_doc/model-support.md)。

### 3. 与编程 Agent 协作

在编程 Agent 环境中打开 SAEBase 源码仓库，从 [Harness 指南](harness/README.md)开始。提出需要完成的科研任务，确认允许修改的范围，再由独立审阅流程检查真实执行行为及产物证据。**完整 Harness 随源码仓库提供**；单独安装 Python wheel 主要提供 SAEBase 软件接口。

## 🔬 研究工作流

SAEBase 将模型测量与后续统计分析分开，方便按需要组合。

| 目标 | 可运行的示例 | 详细文档 |
| --- | --- | --- |
| 提取并复用激活 | [Activation Store](saebase/examples/activations/activation_store.py) | [激活存储](harness/github_development_doc/activation-store.md) |
| 测试特征干预 | [Batch Interventions](saebase/examples/interventions/batch_interventions.py) | [特征编辑与 Steering](harness/github_development_doc/feature-interventions.md) |
| 基准与可靠性分析 | [First Experiment](saebase/examples/quickstart/first_experiment.py) | [Benchmark](harness/github_development_doc/benchmarking.md) · [Reliability](harness/github_development_doc/reliability.md) |
| 接入研究方法 | [Research Examples](saebase/examples/research_integrations/README.md) | [SAR、RSS、Localization 和 SAEBench](harness/github_development_doc/research-integrations.md) |

特征编辑可以保留 SAE 尚未重构出来的激活部分：

```text
edited_activation = original_activation + decode(edited_codes) - decode(original_codes)
```

完整 SAE 重构是另一种实验对照。可靠性分析以独立的 prompt group 为采样单位，不把同一个提示的多个 token 或变体当作相互独立的数据。

### ⚡ 运行性能

SAEBase 为重复模型评测和特征干预提供可选加速功能。具体实测结果、硬件条件和适用范围参见[性能报告](harness/github_development_doc/performance/workplan11_final_results.md)。

## 🧠 模型与 SAE 支持

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
**已验证的模型与 SAE（2026-10-16）。** 覆盖 **18** 组模型与 SAE 配置（270M-9B），**48/48** 个选定层位通过真实权重兼容性检查，**48/48** 个完成基于 64 组独立留出提示的基准评估，**48/48** 个完成统计可靠性审计。详见[模型支持与逐层验证结果](harness/github_development_doc/verified-model-matrix.md)。
<!-- END VERIFIED_MODEL_SUPPORT -->

[模型与 SAE 验证矩阵](harness/github_development_doc/verified-model-matrix.md) · [验证方法](harness/github_development_doc/validation.md)

## 📚 文档

[入门指南](harness/github_development_doc/getting-started.md) · [Python API](harness/github_development_doc/python-api.md) · [示例](saebase/examples/README.md) · [Research Lab](saebase/research_lab/README.md) · [文档目录](harness/github_development_doc/index.md)

查看源码：[SAEBase API](saebase/) · [Models](models/) · [Datasets](datasets/) · [Tasks](tasks/) · [Harness](harness/README.md) · [配置](conf/) · [仓库目录指南](harness/github_development_doc/repository-guide.md)。

[参与贡献](.github/CONTRIBUTING.md) · [更新记录](harness/github_development_doc/CHANGELOG.md) · [引用信息](CITATION.cff) · [Apache-2.0 许可证](LICENSE) · [安全说明](.github/SECURITY.md)

---

<p align="center">
  <em>Building open foundations for understanding and evaluating LLMs.</em><br>
  <a href="https://github.com/OpenSAE">OpenSAE</a> · <a href="https://github.com/OpenSAE/SAEBase/issues">Issues</a>
</p>
