# HF model cache cleanup execution record

Background and objectives: The user has clearly requested that seven models (formerly 85.85 GiB) other than the current four model Capture be temporarily offloaded and that the GPS-2 cache be repeated. Only local HFC caches are cleared, scientific profiles, experimental results, corresponding SAEs and current four models are maintained.

GPU has been confirmed as uncalculated and previous large tasks have been stopped. Seven model weights and duplicate copies of Models-gpt2 and their lock directory are deleted from the accurate repository directory, and models-openai-community-gpt2 are maintained.

| Cache repository | Use GiB before deleting | Result |
| --- | ---: | --- |
| models--google--gemma-3-12b-it | 22.737 | Deleted |
| models--Qwen--Qwen3.5-9B-Base | 18.002 | Deleted |
| models--Qwen--Qwen3-8B-Base | 15.267 | Deleted |
| models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B | 14.966 | Deleted |
| models--google--gemma-2-2b | 9.760 | Deleted |
| models--Qwen--Qwen3-1.7B-Base | 3.216 | Deleted |
| models--google--gemma-3-1b-pt | 1.899 | Deleted |
| models--gpt2 | 0.513 | Deleted |

Total deletion of occupation: 86.360 GiB; space available for file systems: 31.369.117.216 GiB.

Complete check: None of the eight specified directories exist and all of the current four model caches exist; no changes have been made to SAE, test results and model configurations. AI opinion: When future batches of the old ten models are restored, the missing model needs to be redownloaded. See cleanup.json for a more detailed list.

The subsequent batch consistency experiment found that TransformerLens was still read under an alias gpt2 and had re-created 0.51 GiB duplicate weights. Two complete SHA256 were checked and replaced with a hard link to content blob, with only one physical weight; the necessary aliases data were retained, and subsequent reading was not repeated. The deletion of any warehouse would not destroy another link. See ../20261116_batch_invariance/gpt2_cache_deduplication.json for the record.
