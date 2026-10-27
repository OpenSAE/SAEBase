# OpenResearch · 科学研究知识与方法

Knowledge Tree 以可定位的源码证据组织现有科研方向。目前覆盖：

- **CUB-200 × ResNet-18**：鸟类分类数据、分类模型、训练与可复现评测。
- **LLM SAE 特征分析**：模型与 Prompt 资产、特征捕获、教师强制干预、生成干预、电路追踪、自动解释、基准与可靠性评测、SAE 训练。

八个 LLM SAE 方向分别拥有可追踪的研究问题、已有方法接口与独立的执行目标。方法节点的绿色表示**源码方法已经存在**；科研有效性、因果证据、质量指标及跨数据集结论另行验证。仍处于研究过程中的问题保留 `partial` 状态，公开 Atlas 的已有能力视图主要展示有源码证据的方法部分。

| 文件 | 权威内容 |
| --- | --- |
| [context_graph.json](context_graph.json) | CUB 和 SAE 研究问题、方法、具体操作与独立状态 |
| [glossary.json](glossary.json) | 数据、模型、SAE 方法、可靠性和实验资源的术语 |
| [context_attributes.json](context_attributes.json) | 源码、方法参数、示意图和公式的补充说明 |
| [foundations/sae_methods.md](foundations/sae_methods.md) | SAE 的实验定义、操作范围与科学验证边界 |
| [foundations/](foundations/) | CUB 数据、模型、训练、评测与复现 |

[Knowledge Tree](https://atheavenma.github.io/opensae/openresearch/atlas/knowledge.html) 展示两个研究领域；[Context Tree](https://atheavenma.github.io/opensae/openresearch/atlas/context.html) 提供统一词汇；[Code Tree](https://atheavenma.github.io/opensae/openresearch/atlas/code.html) 保持源代码本身的模块与实体结构。

`harness/core/project_graph.json` 管理项目架构；Knowledge 负责科学问题和方法。术语前向引用与回链在 CI 中共同校验。科研结果的真实验收须提供独立实验数据、基准与可复现运行证据。
