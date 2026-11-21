# Model

## Purpose

Description of the use, selection, input output and limitations of the model item by item.

## Scope

For the selection of actual operations, the interpretation of results and the maintenance of project descriptions.

## Purposes and access

Model to receive text, SAEBASE can extract or modify the activation of the specified layer, and then observe the text renewal fractions. The SLEE encoder is the weight of the separate training, which is used to express the layer activation as a thin feature. Experimentally saves the basic weights, SAE and layers.

## Shared Parameters

Select the model using `model=<name>`. The default SAE and layers are covered by the task parameter and the condition is checked by [set-up and precheck](tasks.md) (for the selection of operations and access to their products, see how the capture results are available for the intervention selection). The project model card is used to describe local selection. The upstream background is shown in [Gemma 3 technical report](https://arxiv.org/abs/2503.19786), [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).

## 1. Gemma 3

### 1.1 `gemma1b`

Gemma 3 is a billion-quantitative version that is suitable for a smaller-scale modulation of pre-training and command.

Select with `model=gemma1b`. Preset weighting is `google/gemma-3-1b-pt`. Entering is the sequence of text and after-words, which is mainly active as the middle of the extension fraction and the selected position. Default SAE is `gemma1b_scope_resid`. Preset layer is `gemma1b_scope_resid` Layer 13. Option `gemma1b_pt_l25_transfer_big`, `gemma1b_pt_l25_transfer_small`. The meaning of the same layer or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-1b-pt)

### 1.2 `gemma1b_it`

Gemma 3 ' s one billion-volume version, suitable for a smaller-scale modulation of pre-training against commands, is the directive ' s gratuity and weights, with a distinction between the presentation format and suitability of the instructions when comparing the base version.

Select with `model=gemma1b_it`. Pre-set weights are marked `google/gemma-3-1b-it`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma1b_it_l25_native_big`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-1b-it)

### 1.3 `gemma270m`

Gemma 3 ' s 270 million scale option, suitable for low-cost mechanism experiments, with dimensions and characterization numbers determined by this weight.

Select with `model=gemma270m`. Pre-set weights are marked `gemma-3-270m`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma270m_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml)

### 1.4 `gemma3_12b_it`

Gemma 3's 12 billion-quantitative version, which is used for larger-scale feature comparisons, should be tested by a physical model. This is a command shift of weight and weight, with a distinction between a hint format and suitability when comparing the base version.

Select with `model=gemma3_12b_it`. Pre-weighted weights are `google/gemma-3-12b-it`. Entering is a sequence of text and after-words, which is primarily read to activate as the middle of the extension fraction and the selected location. Default SAE is `gemma3_12b_it_scope_resid`. Pre-set layer is `gemma3_12b_it_scope_resid` Layer 24. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-12b-it)

### 1.5 `gemma3_12b_pt`

Gemma 3's 12 billion-volume version, which is used for larger-scale character comparisons, should be tested on the basis of a physical model, with basic pre-training weights as the subject of the experiment.

Select with `model=gemma3_12b_pt`. Pre-set weights are marked `google/gemma-3-12b-pt`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma3_12b_pt_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-12b-pt)

### 1.6 `gemma3_27b_it`

Gemma 3's 27-billion-volume version, both weight and intermediate activation costs are subject to environmental precheck. This is the command that adjusts the weight of weight, distinguishing between the hint format and suitability when comparing the base version.

Select with `model=gemma3_27b_it`. Pre-set weights are marked `google/gemma-3-27b-it`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma3_27b_it_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-27b-it)

### 1.7 `gemma3_27b_pt`

Gemma 3 ' s 27 billion-quantitative version, with weight and intermediate activation costs subject to environmental precheck, is subject to a base pre-training weight.

Select with `model=gemma3_27b_pt`. Pre-set weights are marked `google/gemma-3-27b-pt`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma3_27b_pt_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-27b-pt)

### 1.8 `gemma3_4b_it`

Gemma 3's 4 billion-quantitative version, used for cross-structure and layer comparisons in the project, is a command shift of weights, with a distinction between a hint format and suitability when comparing the base version.

Select with `model=gemma3_4b_it`. Pre-set weights are marked `google/gemma-3-4b-it`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma3_4b_it_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-4b-it)

### 1.9 `gemma3_4b_pt`

Gemma 3's 4 billion-quantitative version for cross-structure and layer comparison in the project, with base pre-training weights as the subject of the experiment.

Select with `model=gemma3_4b_pt`. Preset weights are `google/gemma-3-4b-pt`. Entering is the sequence after the text and the fraction, which is primarily read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma3_4b_pt_scope_resid`. Preset layer is `gemma3_4b_pt_scope_resid` Layer 17. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-3-4b-pt)

## 2. Gemma 2

### 2.1 `gemma2_2b_base`

Gemma 2 ' s 2 billion-quantitative Transformer is one of the language models commonly used in the project, with a base pre-training weight as the subject of the experiment.

Select with `model=gemma2_2b_base`. Preset weights are marked with `google/gemma-2-2b`. Entering is the sequence of text and after-words, which is primarily active as the middle of the extension fraction and the selected position. Default SAE is `gemma2_2b_pt_l25_canonical`. Preset layer is `gemma2_2b_pt_l25_canonical` 25th Layer. You can also choose `gemma2_2b_it_matryoshka_l12_32k`, `gemma2_2b_it_scope_resid_l12`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-2-2b)

### 2.2 `gemma2_2b_it`

Gemma 2's 2 billion-quantitative Transformer is one of the language models commonly used in the project. This is a command to adjust the weight of weights, with a distinction between the hint format and suitability when comparing the base version.

Select with `model=gemma2_2b_it`. Pre-set weights are marked `google/gemma-2-2b-it`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma2_2b_pt_l25_canonical`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-2-2b-it)

### 2.3 `gemma2_9b_it`

Gemma 2 ' s 9 billion-quantitative Transformer, for a larger analysis than 2B, requires the use of a corresponding check point. This is a command to shift the weight of the weight of the weight, with a distinction between the hint format and suitability when comparing the base version.

Select with `model=gemma2_9b_it`. Pre-set weights are `google/gemma-2-9b-it`. Enter the sequence after text and division, which is primarily read to activate between the continuation fraction and the selected position. Default SAE is `gemma2_9b_it_scope_resid`. You can also choose `gemma2_9b_it_scope_resid_l31_131k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-2-9b-it)

### 2.4 `gemma2_9b_pt`

Gemma 2 ' s 9 billion-volume Transformer for a larger analysis than 2B, with a corresponding check point, with the base pre-training weight as the subject of the experiment.

Select with `model=gemma2_9b_pt`. Pre-set weights are marked `google/gemma-2-9b`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `gemma2_9b_pt_scope_resid`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/google/gemma-2-9b)

### 2.5 `openresearch_gemma2_2b_base`

The Gemma 2 2B base model for the OpenResearch research process is pre-set. The project registers it separately from other Gemma weights to track compatibility and run records.

Select with `model=openresearch_gemma2_2b_base`. Base weights are solved by the corresponding adaptor to the item. Enter the sequence after the text and the semiword, which is mainly active as the middle of the extension fraction and the selected position. Default SAE is `gemma2_2b_pt_l25_canonical`. Preset layer is `gemma2_2b_pt_l25_canonical` 25. You can also select `gemma2_2b_it_matryoshka_l12_32k`, `gemma2_2b_it_scope_resid_l12`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml)

## 3. GPT-2

### 3.1 `gpt2small`

GPT-2 Small self-return Transformer is a classic contrast for mechanisms such as indirect benevolent recognition.

Select with `model=gpt2small`. Pre-weighted weights are `gpt2-small`. Entering is the sequence of text and after-words, which is mainly active as the middle of the extension fraction and the selected location. Default SAE is `gpt2small_res_jb_l8`. Pre-set level 8 is `gpt2small_res_jb_l8`. Option `gpt2small_fs_12288_l8`, `gpt2small_fs_24576_l8`, `gpt2small_fs_6144_l8`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml)

## 4. Llama 3.1

### 4.1 `llama31_8b`

Llama 3.1's 8 billion-volume self-regression Transformer can be used for question-and-answer behaviour and SAE intervention.

Select with `model=llama31_8b`. Pre-weighted weights are `meta-llama/Llama-3.1-8B`. Entering is a sequence of text and after-words, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `llama31_8b_lxr_8x`. Pre-set layer is `llama31_8b_lxr_8x` Layer 21. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/meta-llama/Llama-3.1-8B)

### 4.2 `llama31_8b_instruct`

Llama 3.1's 8 billion-volume self-regression Transformer, which can be used for question and answer behaviour and SAE interventions, is the directive to shift the weight of the weight of the weight, with a distinction between the format and suitability of the hint when comparing the base version.

Select with `model=llama31_8b_instruct`. Pre-set weights are `meta-llama/Llama-3.1-8B-Instruct`. Enter the sequence after text and division, which is primarily read to activate between the continuation fraction and the selected position. Default SAE is `llama31_8b_instruct_andyrdt_l19_k64`. You can also choose `llama31_8b_instruct_andyrdt_l23_k64`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## 5. Mistral

### 5.1 `mistral7b_instruct`

The V0.1 command version of Mistral 7B, which is used for text behaviour and SAE analysis, is determined by the model adaptor. This is the command that adjusts the weight of the weight, with a distinction between the hint format and suitability when comparing the base version.

Select with `model=mistral7b_instruct`. Pre-set weights are marked `mistralai/Mistral-7B-Instruct-v0.1`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `mistral7b_instruct_tc_l16_131k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

## 6. Pythia

### 6.1 `pythia410m`

Pythia 's $411 million to retrain the data version to compare the impact of the model 's size with the 70M option.

Select with `model=pythia410m`. Pre-set weights are marked `pythia-410m-deduped`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `pythia410m_mlp_topk`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml)

### 6.2 `pythia70m`

Seven million of Pythia ' s 7 million go-trained data versions are suitable for low-cost entry and small-scale interventions.

Select with `model=pythia70m`. Pre-set weights are `pythia-70m-deduped`. Enter the sequence after text and division, which is primarily read to activate between the continuation fraction and the selected position. Default SAE is `pythia70m_resid_sm`. You can also choose `pythia70m_saebench_gated_l3`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml)

## 7. Qwen

### 7.1 `qwen25_3b`

Qwen2.5 is a 3 billion-quantitative version. The base and command versions use independent model identities and check points.

Select with `model=qwen25_3b`. Preset weights are marked `Qwen/Qwen2.5-3B`. Enter the sequence after the text and the fraction, which is primarily read to activate between the continuation fraction and the selected location. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen2.5-3B)

### 7.2 `qwen25_3b_it`

Qwen2.5 uses an independent model identity and check point. This is a command shift of weight, which requires a distinction between a hint format and suitability when comparing the base version.

Select with `model=qwen25_3b_it`. Preset weights are marked `Qwen/Qwen2.5-3B-Instruct`. Enter the sequence after the text and the fraction, which is primarily read to activate between the continuation fraction and the selected location. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

### 7.3 `qwen35_27b`

Qwen3.5's 27 billion-quantitative version requires a separate precheck of selected layers, weighting layouts and resources.

Select with `model=qwen35_27b`. Pre-set weights are marked `Qwen/Qwen3.5-27B`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen35_27b_resid_w80k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3.5-27B)

### 7.4 `qwen35_2b`

Qwen3.5 ' s 2 billion-volume base version for small-scale characterization and question-and-answer experiments.

Select with `model=qwen35_2b`. Pre-set weights are marked `Qwen/Qwen3.5-2B-Base`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen35_2b_resid_w32k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3.5-2B-Base)

### 7.5 `qwen35_4b`

Qwen3.5's 4 billion-quantitative version has been used for BBH precheck and natural question and answer capture.

Select with `model=qwen35_4b`. Pre-set weights are marked `Qwen/Qwen3.5-4B`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen35_4b_matryoshka_l15_65k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3.5-4B)

### 7.6 `qwen35_9b`

Qwen3.5's 9 billion-quantitative base version, configured for a stand-alone residual feature experiment.

Select with `model=qwen35_9b`. Preset weights are marked with `Qwen/Qwen3.5-9B-Base`. Entering is the sequence of text and after-words, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen35_9b_resid_w64k`. Preset layer is `qwen35_9b_resid_w64k` Level 16. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3.5-9B-Base)

### 7.7 `qwen3_1_7b`

Qwen3 has a 1.7 billion-volume version with a separate local configuration for the same size base.

Select with `model=qwen3_1_7b`. Pre-set weights are marked `Qwen/Qwen3-1.7B`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen3_1_7b_ak_resid_l14_65k`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3-1.7B)

### 7.8 `qwen3_1_7b_base`

The Qwen3 version of 1.7 billion volume-scale, with a separate local configuration of the same size base version, uses the base pre-training weight as the subject of the experiment.

Select with `model=qwen3_1_7b_base`. Preset weights are marked with `Qwen/Qwen3-1.7B-Base`. Entering is the sequence after text and semiwords, which is primarily read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen3_1_7b_base_resid_w32k`. Preset layer is `qwen3_1_7b_base_resid_w32k_l0_100` 14th floor, `qwen3_1_7b_base_resid_w32k` 14th floor. Alternatively, you can choose `qwen3_1_7b_base_resid_w32k_l0_100`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

### 7.9 `qwen3_8b`

Qwen3 selects the 8 billion-quantitative base version of the project according to the independent residual SSE.

Select with `model=qwen3_8b`. Preset weights are marked `Qwen/Qwen3-8B-Base`. Entering is the sequence of text and after-words, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `qwen3_8b_resid_w64k`. Preset layer is `qwen3_8b_resid_w64k` Layer 18. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/Qwen/Qwen3-8B-Base)

## 8. DeepSeek-R1 distillation model

### 8.1 `r1_distill_llama8b`

DeepSeek-R1 distilled version Llama 8B, which compares text behaviour of the distilled model.

Select with `model=r1_distill_llama8b`. Preset weights are marked with `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. Enter a sequence after text and division, which is primarily active as the middle of the extension fraction and the selected position. Default SAE is `r1_distill_llama8b_llamascope_l23`. Preset layer is `r1_distill_llama8b_llamascope_l23` 23rd Layer. Alternatively, you can select `r1_distill_llama8b_llamascope_l15_openr1`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

### 8.2 `r1_distill_qwen15b`

DeepSeek-R1 distilled version Qwen 1.5B to observe the behaviour of the distillation model under lower resources.

Select with `model=r1_distill_qwen15b`. Pre-set weights are marked `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Enter the sequence after the text and the fraction, which is mainly read to activate as the middle of the extension fraction and the selected position. Default SAE is `r1_distill_qwen15b_resid_l20`. The meaning of the same name or feature number depends on the actual weight and SAE.

[Project preset](../conf/models/catalog.yaml) · [Source of weights](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
