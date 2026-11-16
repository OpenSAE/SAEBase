# conf/models_log.md — HookedTransformer models

## 目的和主要功能

记录模型键、权重来源与当前加载窗口，供选择模型和维护配置时查阅。

## 适用范围

用于模型配置与资源选择；实际登记由 models/plugins.py，层位与 SAE 由所属配置维护。

模型执行窗口统一为 `min(模型原生上下文上限, 4096)`，由 `models/loading_policy.py` 计算。TransformerLens 在构建时设置 `n_ctx`；Hugging Face 后端通过包装配置提供同一字段。SAR 的完整答案评分沿用该执行窗口，题目与答案共同占用它。GPT-2 Small 保持 1024；本批其他九个模型使用 4096。原生窗口来自模型文本配置，不使用 tokenizer 的无穷大占位值或后端较小的保守默认值。

| key | HF / TL name | notes |
|-----|--------------|-------|
| pythia70m | EleutherAI/pythia-70m-deduped | smoke default |
| pythia410m | EleutherAI/pythia-410m-deduped | MLP TopK SAE |
| gemma270m | google/gemma-3-270m | Gemma Scope 2 |
| gemma1b | google/gemma-3-1b-pt | Gemma Scope residual |
| gemma1b_it | google/gemma-3-1b-it | instruct; gated HF |
| r1_distill_qwen15b | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | R1 distill; reasoning TopK SAE L20 |
| r1_distill_llama8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | R1 distill Llama; Llama-Scope R1 L23 |
| gemma2_2b_base | google/gemma-2-2b | Gemma-2 residual SAE |
| gemma2_2b_it | google/gemma-2-2b-it | instruct; gated HF |
| qwen25_3b | Qwen/Qwen2.5-3B | Hooked LM plugin |
| qwen25_3b_it | Qwen/Qwen2.5-3B-Instruct | instruct variant |
| gemma3_4b_pt | google/gemma-3-4b-pt | Scope 2 resid 65k; L 9/17/22/29 |
| gemma3_4b_it | google/gemma-3-4b-it | native IT Scope 2 |
| llama31_8b | meta-llama/Llama-3.1-8B | Llama-Scope LXR-8x；默认第21层、32k特征；gated |
| llama31_8b_instruct | meta-llama/Llama-3.1-8B-Instruct | 协议 10；andyrdt BatchTopK L19 k=64（threshold inference） |
| qwen3_8b | Qwen/Qwen3-8B-Base | official residual TopK 64k |
| qwen3_1_7b | Qwen/Qwen3-1.7B | 协议 11b；adamkarvonen BatchTopK L14 65k L0≈80 |
| qwen35_4b | Qwen/Qwen3.5-4B | 协议 11d；Matryoshka BatchTopK L15 65k k=100 |
| gemma2_9b_pt | google/gemma-2-9b | Scope 16k; L 9/20/31 |
| gemma2_9b_it | google/gemma-2-9b-it | native IT Scope 16k |
| gemma3_12b_pt | google/gemma-3-12b-pt | Scope 2; L 12/24/31/41 |
| gemma3_12b_it | google/gemma-3-12b-it | native IT Scope 2 |
| qwen35_9b | Qwen/Qwen3.5-9B-Base | official residual TopK 64k |
| gemma3_27b_pt | google/gemma-3-27b-pt | Scope 2; L 16/31/40/53; quant OK |
| gemma3_27b_it | google/gemma-3-27b-it | native IT Scope 2; quant OK |
| qwen35_27b | Qwen/Qwen3.5-27B | official residual TopK 80k; quant OK |

最终层由模型实际层数确定，索引为 `model.cfg.n_layers - 1`。量化加载按所选模型与任务的真实支持判断。完整 SAE 与层位选择读取 `conf/sae/`、所属模型配置和 `saebase/api/sae_profiles.py`；已有真实观测见[模型支持资料](../doc/code/development/model-support.md)。

Qwen3.5 4B 使用 `conf/runtime/environments.yaml` 中的 Qwen3.5 专用解释器；L15 Matryoshka SAE 显式绑定 Neuronpedia `qwen3.5-4b/15-res-matryoshka-65k`。来源仍核对仓库和物理目录；本地声明与上游目录冲突时拒绝加载解释。
