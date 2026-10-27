# LLM SAE 研究方法与可核验边界

本页为 OpenResearch Knowledge Tree 中的 LLM SAE 方法提供术语与证据背景。实验所用数据、模型和任务应通过运行配置明确选择；当前代码包含 Gemma-2 2B 的模型适配、SAE 加载、特征捕获、干预、生成、电路、解释、评测与训练接口。

## 研究问题与实现方法

- 特征捕获：从指定模型、层和激活位置提取表示，用 SAE 编码得到特征响应。代码依据：`saebase/api/features.py`、`tasks/LLM/capture/operation.py`。
- 教师强制干预：在相同输入与读取位置替换激活或 SAE 特征，比较下游输出。代码依据：`tasks/LLM/intervene/operation.py`。
- 生成干预：在生成过程中执行干预，对照基线续写及干预后的输出。代码依据：`tasks/LLM/generate/operation.py`。
- 电路分析：使用有来源的模型组件和干预追踪接口分析候选因果路径。代码依据：`tasks/LLM/circuits/backend/circuits.py`。
- 自动解释：根据示例生成特征解释假设，通过独立样本检验其行为。代码依据：`tasks/LLM/explain/interpretation.py`。
- SAE 基准：执行指标、重建、干预、可靠性及 SAR/RSS 的实验适配接口。代码依据：`tasks/LLM/benchmark/backend/runner.py`、`tasks/LLM/benchmark/integrations/sar.py`、`tasks/LLM/benchmark/integrations/rss.py`。
- SAE 训练：通过原生训练接口执行 SAE 训练与参数配置。代码依据：`tasks/LLM/train/backend/training.py`。
- 模型与 Prompt：以明确模型身份及可追溯 Prompt 产物维护输入。代码依据：`models/LLM/gemma2_2b_base/gemma2_2b_base.py`、`datasets/NLP/artifacts.py`。

## 状态含义

Knowledge 中标为 implemented 的 SAE 方法节点表示其明确命名的**代码操作或数据协议已有对应实现**。它仅证明源码接口存在，不能用来推断机制解释的因果正确性、生成控制质量、基准排名或训练结果。

尚需独立实验验证的研究问题保持 partial。任何效果结论都应另附模型身份、SAE 权重、数据划分、对照、测量结果和可复现的实验产物。Knowledge、Code 和 Core 分别维护科研问题、源码事实和项目设计，不以类别名称互相替代。
