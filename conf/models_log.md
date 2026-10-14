# conf/models_log.md — HookedTransformer models

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
| llama31_8b | meta-llama/Llama-3.1-8B | Llama-Scope LXR-8x; gated |
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

Final layer = `model.cfg.n_layers - 1` (26 blocks → L25).
Quantized checkpoints are allowed. Full SAE layer/id table: [`doc/00_model_sae_info/00_model_sae_info.md`](../doc/00_model_sae_info/00_model_sae_info.md).
