# 🧱 SAEBase：稀疏自编码器研究基础设施

## OpenResearch Harness 与统一科研工作区

当前开发分支在保留 SAEBase v0.5 的 API、CLI、模型注册、数据处理、实验后端和发布流程的基础上，增量引入 [OpenResearch](https://github.com/OpenSAE/OpenResearch) 的执行层与治理框架。该分支属于源代码集成阶段，尚未构成新的正式发行版本。

| 模块 | 整合方式 |
| --- | --- |
| `saebase/` | 保留原有 SAEBase 基础算法、API、模型/SAE 注册、CLI 与科研实现。 |
| `dlbase/` | 新增独立 Hydra 执行入口与 Dataset/Model/Task Registry。 |
| `datasets/` | 并入 Vision CUB-200 与 NLP 数据集；保留原有 `saebase/data/`。 |
| `models/` | 保留全部旧模型，新增 Vision 和 OpenResearch 的 Gemma-2 适配。 |
| `tasks/` | 保留旧任务入口，新增七类 LLM Task。 |
| `harness/` 与 `.cursor/` | 引入 Agent Flow、五棵 Tree、Atlas、Hooks、规则和 Skills。 |
| `doc/` | 引入知识图与代码关系说明；继承的节点状态需要按 SAEBase 实际情况重新核验。 |

原有 `conf/config.yaml` 和 `saebase` 命令保持不变；DLBase 使用 `conf/openresearch.yaml` 和 `python -m dlbase.train`。为避免破坏旧插件入口，新注册模块分别使用 `models/dlbase_plugins.py` 和 `tasks/dlbase_plugins.py`。

```bash
python .cursor/hooks/validate_repo.py --full
python -m dlbase.train --cfg job
```

新增顶层 `datasets` 与 Hugging Face 的同名第三方库存在命名冲突，已引入隔离数据加载适配器；正式使用前仍需完成其余调用路径、GPU 环境、权重兼容性和全部回归测试的验收。

[迁移说明](docs/openresearch-integration.md) · [Harness](harness/README.md) · [Atlas](harness/visualization/README.md)


🌍 语言：[English](README.md) | [中文](README_cn.md) &nbsp; [![CI](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenSAE/SAEBase/actions/workflows/ci.yml)

**最新软件版本：** [v0.5.0 Public Research Beta](docs/releases/v0.5.0.md) · [OpenSAE Playground](https://atheavenma.github.io/opensae/workspace.html) · [Python API](docs/python-api.md)

👉 **简介。** SAEBase 用于理解、测试和修改语言模型中的稀疏自编码器（SAE）特征。它提供激活提取与复用、重构分析、特征干预、下游行为评测以及可追溯的实验结果。

<p align="center">
  <a href="#-快速开始"><b>🚀 快速开始</b></a> &nbsp; | &nbsp;
  <a href="#-研究工作流"><b>🔬 研究工作流</b></a> &nbsp; | &nbsp;
  <a href="#-research-lab"><b>🧪 Research Lab</b></a> &nbsp; | &nbsp;
  <a href="#-文档"><b>📚 文档</b></a>
</p>

🤗 **Hugging Face：** [SAEBase Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) · [Evaluation Dataset](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation)

---

## 🧭 概览

SAEBase 将预训练模型、SAE 字典、激活测量、受控特征干预、基准评测和可靠性分析连接为可复用的研究流程。

<p align="center">
  <img src="assets/figures/saebase_workflow.svg" alt="SAEBase 模型、激活提取、SAE 特征干预、评测与报告流程" width="1080"/>
</p>

**一次提取，多 SAE 评估。** 在相同的模型层位上，通过身份校验的激活缓存可以由兼容的 SAE checkpoint 复用，避免重复执行模型前向。

## ✨ 核心能力

- 🧠 **模型与 SAE。** 已注册 checkpoint、明确的激活位置、模型适配器及来源身份校验。
- 🗃️ **可复用的激活。** 支持恢复的分片缓存、无损 SAE 编码、指定 token 测量和完整性校验。
- 🎛️ **特征干预。** Identity、ablation、scaling、clamping、addition、coalition 和 reconstruction，支持测量下游 logits 与行为变化。
- ⚡ **高速执行。** 可选的 suffix replay、逐 token SAE KV 缓存干预，以及 vLLM decoder-direction steering。
- 📊 **基准与可靠性。** 重构、稀疏性、行为指标、配对比较、按 prompt group 的 Bootstrap 与负对照。
- 🧩 **研究集成。** 原始 SAR、RSS 估计器、Input Localization，以及已验证的 SAEBench 原生结果导入。

## 🧪 Research Lab

[Research Lab](https://huggingface.co/spaces/OpenSAE/SAEBase) 展示已公开的 SAE 测量与报告。下表还包括单独记录的 GPU 实验结果。

| 研究内容 | 已记录的实验结果 |
| --- | --- |
| 模型与 SAE 覆盖 | 48 个选定层位、500 个冻结的父提示 |
| Feature Explorer | 六个主要层位的 300 张特征统计卡片 |
| 特征干预 | 六个真实模型/SAE 层位的 clean 与 edited logits |
| SAR | 1,485 组 K16、14 组 K8 与一组 K4 测量 |
| RSS | 六个层位，共 138,000 条原始输入和候选变体记录 |
| Input Localization | 一个 64 状态的输入网格、55 个有观测支持的特征 |

[**打开 Research Lab**](https://huggingface.co/spaces/OpenSAE/SAEBase) · [数据集](https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation) · [实验结果](docs/research-lab/results-20261020.md) · [软件版本说明](docs/releases/v0.4.0.md)

## 🚀 快速开始

### 1. 安装并运行完整示例

需要 Python 3.11 或 3.12。下面两个示例均可在 CPU 上运行，无须下载预训练模型。

```bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .

python examples/quickstart/public_api_offline.py
python examples/quickstart/first_experiment.py --output runs/first-experiment
```

第一个示例展示激活提取和 SAE 特征编辑。第二个示例使用两个小型合成 SAE，生成 `report.html`、`summary.csv` 和可靠性分析报告。命令会打印具体输出路径，可以直接用浏览器打开 HTML 报告。

### 2. 使用真实模型和 SAE

安装可选的模型分析依赖，然后运行 Pythia-70M 示例。该示例需要下载公开的模型与 SAE 权重。

```bash
python -m pip install -e ".[analysis]"
python examples/interventions/pythia_feature_edit.py
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

这段代码保留两个特征贡献，移除另两个，并输出下一个 token 的 logits 变化幅度。其他模型、token 位置和编辑操作参见 [Python API](docs/python-api.md) 和 [模型注册表](docs/model-support.md)。

## 🔬 研究工作流

SAEBase 将模型测量与后续统计分析分开，方便按需要组合。

| 目标 | 可运行的示例 | 详细文档 |
| --- | --- | --- |
| 提取并复用激活 | [Activation Store](examples/activations/activation_store.py) | [激活存储](docs/activation-store.md) |
| 测试特征干预 | [Batch Interventions](examples/interventions/batch_interventions.py) | [特征编辑与 Steering](docs/feature-interventions.md) |
| 基准与可靠性分析 | [First Experiment](examples/quickstart/first_experiment.py) | [Benchmark](docs/benchmarking.md) · [Reliability](docs/reliability.md) |
| 接入研究方法 | [Research Examples](examples/research_integrations/README.md) | [SAR、RSS、Localization 和 SAEBench](docs/research-integrations.md) |

特征编辑可以保留 SAE 尚未重构出来的激活部分：

```text
edited_activation = original_activation + decode(edited_codes) - decode(original_codes)
```

完整 SAE 重构是另一种实验对照。可靠性分析以独立的 prompt group 为采样单位，不把同一个提示的多个 token 或变体当作相互独立的数据。

### ⚡ 实测加速

在单张 RTX 5070 上，Qwen3-1.7B Base 的 25 个样本、7 种条件的完整干预评测通过可选 suffix replay 加速 **1.28×**，报告的指标保持一致。128-token 生成时，逐 token SAE 编辑与 KV 缓存达到 **89 tokens/s**，vLLM SAE decoder-direction steering 达到 **158 tokens/s**。两种生成方式采用不同的干预协议。

[完整 GPU 结果](docs/performance/workplan11_final_results.md) · [性能实现说明](docs/performance.md)

## 🧠 模型与 SAE 支持

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
**已验证的模型与 SAE（2026-10-16）。** 覆盖 **18** 组模型与 SAE 配置（270M-9B），**48/48** 个选定层位通过真实权重兼容性检查，**48/48** 个完成基于 64 组独立留出提示的基准评估，**48/48** 个完成统计可靠性审计。详见[模型支持与逐层验证结果](docs/verified-model-matrix.md)。
<!-- END VERIFIED_MODEL_SUPPORT -->

这里的兼容性、基准评测与可靠性结果基于已记录的参考协议；模型注册成功不代表所有后端的科学结果均可互换。[模型与 SAE 验证矩阵](docs/verified-model-matrix.md) · [验证说明](docs/validation.md)

## 📚 文档

[入门指南](docs/getting-started.md) · [Python API](docs/python-api.md) · [示例](examples/README.md) · [Research Lab](apps/research-lab/README.md) · [文档目录](docs/index.md)

仓库按用途组织：`saebase/` 存放核心 API 和执行引擎，`models/` 管理模型适配器，`conf/` 保存实验配置，`examples/` 提供可运行的例子，`assets/` 保存图片和品牌素材，`docs/` 包含教程与可复现实验证据。早期验证脚本和工作计划与 Quickstart 分开保留。[仓库目录指南](docs/repository-guide.md)

[参与贡献](CONTRIBUTING.md) · [更新记录](CHANGELOG.md) · [引用信息](CITATION.cff) · [Apache-2.0 许可证](LICENSE) · [安全说明](SECURITY.md)

---

<p align="center">
  <b>From learned features to testable explanations.</b><br>
  <a href="https://github.com/OpenSAE">OpenSAE</a> · <a href="https://github.com/OpenSAE/SAEBase/issues">Issues</a>
</p>
