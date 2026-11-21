# doc/code/maintenance/model_log.md — HookedTransformer models

## Purpose

Records the model key, the weight source and the current load window for reference in the selection of the model and maintenance configuration.

## Scope

For model configuration and resource selection; the actual registration is models/plugins.py and the layer and SAE are maintained by the configuration.

The model execution window is uniform as `min(native_model_context_limit, 4096)` and is calculated as `models/loading_policy.py`. TransformerLens sets `n_ctx` at the time of construction; Hugging Face backends provide the same field by packaging configuration. The full answer rating for SAR follows the executive window, where the title is occupied with the answer. GPT-2 Small maintains 1024; the other nine models in this batch use 4096. Original windows are from the model text configuration and do not use tokenizer 's inexhaustible large-position or smaller backend conservative defaults.

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
| llama31_8b | meta-llama/Llama-3.1-8B | Llama-Scope LXR-8x; default 21st floor, 32k feature;gad |
| llama31_8b_instruct | meta-llama/Llama-3.1-8B-Instruct | protocol 10;andyrdt CatchTopK L19 k=64 (threshold inference) |
| qwen3_8b | Qwen/Qwen3-8B-Base | official residual TopK 64k |
| qwen3_1_7b | Qwen/Qwen3-1.7B | protocol 11b;adamkarvon BattTopK L14 65k L0 ⁇ 80 |
| qwen35_4b | Qwen/Qwen3.5-4B | protocol 11d;Matryoshka BattTopK L15 65k k=100 |
| gemma2_9b_pt | google/gemma-2-9b | Scope 16k; L 9/20/31 |
| gemma2_9b_it | google/gemma-2-9b-it | native IT Scope 16k |
| gemma3_12b_pt | google/gemma-3-12b-pt | Scope 2; L 12/24/31/41 |
| gemma3_12b_it | google/gemma-3-12b-it | native IT Scope 2 |
| qwen35_9b | Qwen/Qwen3.5-9B-Base | official residual TopK 64k |
| gemma3_27b_pt | google/gemma-3-27b-pt | Scope 2; L 16/31/40/53; quant OK |
| gemma3_27b_it | google/gemma-3-27b-it | native IT Scope 2; quant OK |
| qwen35_27b | Qwen/Qwen3.5-27B | official residual TopK 80k; quant OK |

The final layer is determined by the actual number of models and the index is `model.cfg.n_layers - 1`. Quantified loads are judged by the true support of the selected model and mission. Full SAE and layer selection read `conf/sae/`, its own configuration and `base/interfaces/sae_profiles.py`; real observations are available [model support information](../documentation/source/development/model-support.md).

Qwen3.5 4B use `conf/base/runtime/runtime.yaml` Medium Qwen3.5 Dedicated interpreter; L15 Matryoshka SAE Obvious binding Neuronpedia `qwen3.5-4b/15-res-matryoshka-65k`.The source still checks the warehouse and physical directory; local statements refuse to load explanations in case of conflict with upstream directories.

HF resets new, independently edited batch tail layer; cache captures stopped at the selected decoder layer. Optimization is followed by pre-testing values for mission performance, and the structure that is not supported retains the original execution and records the cause; does not change the model/SAE identity, weight or numerical accuracy.

GPU scientifics implement a policy to increase the uniform value: Standard Runtime uses `runtime.numerics=vllm`, model pre-loading settings, calculation strategies in fixed work processes; re-use CUDA/BLAS setting, matrix multiplication and log-softmax in vLLMM0.27.1 and a newer version of softmax; preserves the cumulative syntax of the FP32 with low precision input, where HF Attention uses Torch to the math end. `native` Visible recurrence of the historical value path. Restorative identity is maintained, the numerical strategy is used as an independent source to write the results and to integrate them into the rest of the work process; Runtume binds the calculation version before the running comparison to avoid replicating the old results. Design, upstream comit, and acceptance records are found in [Asymmetry and Accelerating](../documentation/source/development/batch-invariant-execution.md).

Qwen3.5 GPU numeric policy upgrades to stand-alone v5: repeat vLLM/FLA fixed line cumulative kernels, replace CUDA cumsum in Transformers 5.223.0 GDAN, and eliminate the difference between the true input of the bat. Qwen ' s old results are retained and recalculated separately, and other models continue V4. HF tail layer reuse is prepared for Hidden, avoiding recalculating the front level of the answer. Source, license and authentication are found in the above-mentioned design document and `third_party/accelerator_kernels/gdn_source.json`.
