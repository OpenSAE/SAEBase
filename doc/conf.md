# SAEBase Configuration Parameters

## Purpose

Indicates where data, models and tasks are selected, and how changes to a given set will change a run. `conf/` Actual documentation and correspondence module organization.

## Scope

To select the current formulation in SAEBase, overwrite this parameter, access the task's specific settings, and interpret the historical results according to the configuration in effect at the time.

## Profile Navigator

Up to two-tier directories under `conf/`, one file for each configuration branch. For example, the default values, batches and filter rules for capture tasks are `conf/tasks/capture/task.yaml` and can be viewed `task`, `matrix` and `sar_filter_policy`, respectively.

Generates by the actual configuration directory. The use and parameters of each file are viewed in the body section.

<!-- BEGIN CONFIG_NAVIGATION -->

| Configure Paths | When? |
| --- | --- |
| [`conf/config.yaml`](../conf/config.yaml) | [Configure Entry and Entry Order](#1-configure-entry-and-entry-order-confconfigyaml),See default combinations, devices and running methods. |
| [`conf/datasets/`](../conf/datasets/) | [Data sets and hints](#2-data-sets-and-hints-confdatasets),selects how to prepare the research input and hint. |
| [`conf/models/`](../conf/models/) | [Model](#3-model-confmodels),selects the base model and default SAE. |
| [`conf/tasks/`](../conf/tasks/) | [Task Configuration](#4-task-configuration-conftasks),selects the task, uses its configuration and looks at the task formulation. |
| [`conf/base/`](../conf/base/) | [Public interface and running](#5-public-interface-and-running-confbase),access to SAE interfaces, operating environments, default test names and data roots. |

<!-- END CONFIG_NAVIGATION -->

## 1 Configure Entry and Entry Order `conf/config.yaml`

**Purpose.** See default combinations, devices and running methods.

View the synthetic configuration from the root of the project `saebase task=capture --cfg job`. The main configuration selects each component group, the component formulation provides the corresponding default, and commands the line to overwrite the current operation. View the configuration using `--cfg job` and select the actual operation by the corresponding entry when performing the task. The historical results use the original running list and configuration snapshot. [Main configuration](../conf/config.yaml)] [Application registration](../base/configuration/__init__.py)] [Experience report](experiment.md)]

The main formulation defaults `dataset=ioi`, `model=openresearch_gemma2_2b_base` and `task=capture`. For example, you want `dataset=bbh` for data and `task.capture.mode=non_whiten` for the capture method.

For configuration-only runs with registered components, each invocation gets a separate record. See the [experiment records guide](experiment.md) for the manifest and result readers.

### 1.1 Operating methods, equipment and numerical strategies

| Settings | Default value | Behaviour |
| --- | --- | --- |
| `seed` | `42` | The main running random feed must be named after a non-negative integer and participate in the running directory. `dataset.sampling_seed`. |
| `runtime.mode` | `null` | New at `new` when omitted. You can select `resume` or `overwrite` in a visible way, and you can check the original list and the running identity. |
| `runtime.numerics` | `vllm` | `native` uses the primary calculator policy. |
| `runtime.device` | `auto` | Shared implementation equipment, automatic mode preference for available CUDAs and CPUs. |
| `runtime.cpu_threads` | `4` | The linear budget used for calculations such as Torch must be positive integer at a value. |

The directories are run by data set, model, task, experimental purpose and optional cross-role organization. The shared directories are repeated for the same purpose, and the results are maintained by the execution record. The historical results are read by the original list and path. [configuration check](../base/runtime/configuration/config_validate.py)] [run paths and running](../base/runtime/experiment_paths.py)]

## 2 Data sets and hints `conf/datasets/`

**Purpose.** selects how to prepare the research input and hint.

[Open Profile Directory](../conf/datasets/)

### 2.1 Data and tip recipe `conf/datasets/catalog.yaml`

Configure the preparation of inputs and tips in `dataset` and `prompt_source`, respectively. [Open formulation](../conf/datasets/catalog.yaml)]

#### 2.1.1 Data set presets

`dataset=<preset_name>` selects the source of the sample. The data set name, default sample and upstream sub-configuration are brought into the preset. The data set description includes samples, labels and applicable tasks, such as comparing BBH's reasoning with IOI's human name projection. [Data set formulation](../conf/datasets/catalog.yaml)] [Data set description](datasets.md)]

| To change what? | Settings and Impacts |
| --- | --- |
| How many questions are taken from the data that are ready? | `dataset.sample_count`, default changes with the data set. BBH defaults 500. Changes the number of questions will change the assessment coverage. |
| Question fixed. | `dataset.sampling_seed`, BBH presets 42. With data versions, presets and sample numbers, the reusable selection conditions are determined. |
| Use raw training and testing separation | `dataset.use_original_split`.BBH Preset as `false`,The project uses a fixed sample record by mission. `true` , the data module retains the upstream division. Capture The fixed sample path follows the registered theme group, MMLU-Pro If you need to set up a separate set, check the task entry and the valid sample list.. |
| Read a specific hint or cache | `dataset.prompt_file` Default Empty, `dataset.data_path` Default is generated by the data root and data set name. Checks the model, terminator and input record when specifying the file or path. |

#### 2.1.2 Hint preparation presets

The preparatory process for using this directory can specify the upstream library, data disaggregation and budget generation. When reading the preparatory product for the experiment, the data set is usually selected; to change the subject(s) to be prepared, the source(s) of the corresponding reminder(s). For example, `prompt_source.prepare.n_generate` determines the number of initial candidates, `min_kept` determines how many qualified questions should be left, and the actual number retained is affected by model and word filtering. `categories`, `balance_per_category`, `max_prompt_tokens` and `n_shot` changes the selection or hints only in the formula that supports these settings. [data catalogue](../conf/datasets/catalog.yaml)] [indicator of origin](../conf/datasets/catalog.yaml)] [data registration](../datasets/plugins.py)]

## 3 Model `conf/models/`

**Purpose.** selects the base model and default SAE.

[Open Profile Directory](../conf/models/)

### 3.1 Model pre-set `conf/models/catalog.yaml`

Select the model with `model=<preset_name>`. The model formula maintains the base weight, matching SAE, default check points and layers. The tasks to support a visible location can be covered by `task.sae_key` and `task.layer`. Both are omitted, and are analysed according to the default rules of the current task and model. In exchange for the model, SAE or layer, the feature number and the identity entered after preparation are reconfirmed.

The normal task is to select the implementation device by `runtime.device`. The `model.device` and `task.device` in the old formulation are still read and are reported in error when multiple visible values are in conflict. SAE uses the planned equipment at the back end of the training, with a specific evaluation to assign the equipment to its agreement. [model formula](../conf/models/catalog.yaml)] [Equipment parsing](../models/language_model/loading_policy.py)] [Model and weight description](models.md)]

## 4 Task Configuration `conf/tasks/`

**Purpose.** selects the task, uses its configuration and looks at the task formulation.

[Open Profile Directory](../conf/tasks/)

### 4.1 Shared Configuration

#### 4.1.1 Mission entrance and resource selection

| Parameters | Formula Value | Purpose and conditions |
| --- | --- | --- |
| `task.extensions` | `[]` | A visible list of installed extensions used by the current task. Empty lists are not loaded for extensions. |
| `task.execute` | `false` | In the task of supporting the field, switch between planned inspections and actual implementation. |
| `task.sae_key` | `null` | Captures, interventions, etc. can be used to overwrite default SAE. Prechecks select the checker with `task.precheck.sae_list`. |
| `task.layer` | `null` | Captures, interventions, etc. can be used to select a model layer and verify it with the selected SAE. Prechecking determines a layer from the object and model formulation. |
| `task.prompt` | `""` | Provides a visible text. This is the basis for the capture to analyse the specified hint, precheck the selected data set. |

The task formulation states its own `task.name`, which is used for distribution on the running layer. The user selects the command to switch the task through a formula such as `task=capture`. Resource location and task validation are determined by the appropriate entry. For example, when the original protocol selects its own resource, set it at the agreed location. SAE [mission resource resolution and validation](../base/runtime/sae_operations.py)]

#### 4.1.2 Shared performance parameters

Capture, intervention, generation and precheck are stated in their respective documents `task.performance`.The following parameters can be used to adjust the number of candidates, to check the cost and consistency of the results. For example, when the number of candidates is increased, the value difference and the visible limit are checked. GPU Operation with `vllm` Numerical policy, which fixes the zero tolerance difference. `runtime.numerics=native` Available for value comparison `+task.performance.atol`,`+task.performance.logit_atol` and `+task.performance.rtol` Override the general output absolute error, the model output fraction absolute error and the relative error, respectively, and use them when omitted `0.000001`,`0.0001` and `0`.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.performance.enabled` | `true` | Enables the performance check for the corresponding task. |
| `task.performance.batch_size` | `auto` | Enables the project to select the actual execution batch according to its performance. |
| `task.performance.batch_sizes` | `[1, 2, 4, 8, 16, 32, 64]` | Lists the number of candidates to be compared. |
| `task.performance.warmup` | `1` | The number of preheats per candidate. |
| `task.performance.repeats` | `3` | Number of repeats of official count times per candidate. |
| `task.performance.probe_samples` | `3` | Number of samples used in the survey. |
| `task.performance.probe_states` | `64` | Survey the number of states used. |
| `task.performance.max_seconds` | `120` | Performance check time budget, in seconds. |
| `task.performance.memory_fraction` | `0.9` | Visible budget ratio for performance probes. |

### 4.2 Benchmark `conf/tasks/benchmark/`

[Open Profile Directory](../conf/tasks/benchmark/)

#### 4.2.1 Benchmark `conf/tasks/benchmark/task.yaml`

To select the assessment protocol and read out. For example, use the activated health check mode to read the statistics of the specified model on several prepared alerts. [Baseline task configuration](../conf/tasks/benchmark/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.benchmark.evaluation` | `null` | Specifies the specific assessment method. When `activation_health` is selected, add matching data hint sources, SAE and assessment protocols, such as `+prompt_source=ioi +sae=gemma2_2b_pt_l25_canonical +evaluation=activation_health`. |
| `task.benchmark.benchmark_config` | `""` | Specifies the baseline protocol that contains the method, input and scope. The built-in protocol is selected by `file_path#presets.name`, and the external independent agreement uses the file path. |
| `task.benchmark.protocol` | `null` | Provide indicators and methodological parameters, and model and data still use the corresponding component selection. |
| `task.resume` | `true` | Supports the re-use of saved results on mission terms. |

##### 4.2.1.1 Model comparison `presets.model_comparison`

Provides for models, SAEs, bits and co-inputs that are involved in the comparison. The indicators for re-establishing errors, explanations, and output changes are used to compare candidates for the SAE. In expanding the comparison, check `targets`, `metrics` and measure the budget. [Model compare formula](../conf/tasks/benchmark/task.yaml)]

##### 4.2.1.2 Model comparison with GPT-2 `presets.model_comparison_gpt2`

Save Organisation GPT-2 A comparative combination and corresponding indicators. GPT-2 For comparison of participation in this document `targets` and `metrics` Reference.[GPT-2 Compare Formula](../conf/tasks/benchmark/task.yaml)

##### 4.2.1.3 Pythia small-scale baseline `presets.pythia_benchmark_smoke`

Fixed Pythia-70M level 3, a SAE and a small sample to check the process from the reading of the sample to the output reconstruction and intervention indicators. View the plan with `saebase-benchmark --config conf/tasks/benchmark/task.yaml#presets.pythia_benchmark_smoke --dry-run`. [Small formulation](../conf/tasks/benchmark/task.yaml)]

##### 4.2.1.4 Pythia Group Baseline `presets.pythia_grouped_benchmark`

Use the same Pythia model and bits to produce results by an independent prompt group for subsequent review of reliability by group. For example, when comparing group results, confirm the hint grouping and measuring upper limit from here. [group formula](../conf/tasks/benchmark/task.yaml)]

##### 4.2.1.5 Activate health check-ups `evaluation`

Use a prepared hint to calculate the activation and reconstruction statistics. Select `task.benchmark.evaluation=activation_health` while selecting the corresponding source, model, and SAE. For example, use 5 hints to see if the activation is normal and adjust the subject matter to the study range. [activate the health formulation](../conf/tasks/benchmark/task.yaml)] [read to achieve](../tasks/benchmark/metrics/health.py)]

| Parameters | Default value | Purpose |
| --- | --- | --- |
| `evaluation.n_prompts` | `5` | Limit the number of ready alerts to check. |
| `evaluation.device` | `cuda` | Specifies the model for this specific evaluation and the SAE execution device. |
| `evaluation.require_prepared_prompts` | `true` | Request to read the tips that are ready and check the source. |

##### 4.2.1.6 Review against data `presets.m4_ground_truth`

Record the number of seeds, sample sizes, confidence levels and re-samplings used in the review. A review of the history check the 128 re-samplings and 95% confidence levels used at the time. [Creview the formula](../conf/tasks/benchmark/task.yaml)]

##### 4.2.1.7 Pythia Reliability Review `presets.pythia_reliability_audit`

Read for `saebase-audit --config` to check the sensitivity of the benchmark indicator to sampling and random seeds. For example, a change `sample_sizes` can compare stability at different sample sizes. [Pythia Reliability Review](../conf/tasks/benchmark/task.yaml)]

##### 4.2.1.8 Models and SAE Overrides Directory `site_catalog`

For benchmarking and SAE pre-testing, registration models, SAEs, layers, and resource sources. For example, a list of layers above the target can be used to check the planned spot. [override directory](../conf/tasks/benchmark/task.yaml)] [bits configuration read](../tasks/benchmark/sites/catalog.py)] [SAE pre-check](../tasks/precheck/sae_checks.py)]

##### 4.2.1.9 Specialized acceptance input

The bit-checking module keeps a fixed sample of the batch with the code for the same input group for the income header. It is used to check the source of the input when assessing the two points. [Special assessment input](../assets/dataset_cohorts/stage_d/files/stage_d_population.jsonl)]

##### 4.2.1.10 Specific acceptance target `site_targets`

Saves the combination and order of the specific acceptances used, including models, SAEs, layers, hook names and the identity of the check point. Checks whether a given point is used in the batch. [Special acceptance target](../conf/tasks/benchmark/task.yaml)]

### 4.3 Capture `conf/tasks/capture/`

[Open Profile Directory](../conf/tasks/capture/)

#### 4.3.1 Capture `conf/tasks/capture/task.yaml`

For example, on the same subject, you can first select a candidate for a standardized activation and then check this pool for answer behaviour. [According configuration](../conf/tasks/capture/task.yaml)]

##### 4.3.1.1 Method selection and output

Select by `task.capture.mode`.

The method determines the basis to be used to select the characteristic or measure the contribution. Other settings control the range and output of the observation.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.mode` | `whiten` | `non_whiten` Sort by original activation, `whiten` Sort by standardized activation, `g_s` Rating by answer behaviour, `sar` Returns with sample, `I` Round-up, `rss` Look for response characteristics for the same target in different contexts. |
| `task.capture.sample_ids` | `null` | Only designated observation topics are dealt with, and complete reference groups are retained. |
| `task.capture.output_scope` | `last_token` | Observe the last word. `all_tokens` Saves the word-by-temporal slender coding to match `task.capture.selection.candidate_count=all`. |
| `task.capture.max_tokens` | `128` | Limiting the number of words recorded to `all_tokens` only. |
| `task.capture.max_active_per_token` | `256` | The maximum number of active features for each word record is used for word-by-word results. |
| `task.capture.whiten_chunk_size` | `256` | The number of words in each batch for the calculation of standardized statistics. |
| `task.capture.output.feature_count` | `null` | Controls the output characteristics or characteristics to the quantity. The output is 10 when the new formulation of g_s is omitted; the rest follows the range of candidates. |
| `task.capture.model_answer.max_new_tokens` | `256` | All Capture saves the response of the model to the original input; this value limits the new creation of the characters. |
| `task.capture.model_answer.batch_size` | `8` | The original response uses the upstream generation method of the loaded model, input the same length and re-use KV Cache. |

##### 4.3.1.2 Shared candidate screening

The number of candidate pools is determined by `task.capture.selection.candidate_count` and the number of final outputs by `task.capture.output.feature_count`.

`non_whiten`,`whiten`,`g_s`,`I` and `sar` Default common candidate pool. Use terminologies and reference topics to filter and exclude Neuronpedia Primary active density is greater than 1% . Save only the non-zero character of the final word and the reference standard difference is greater than zero, in absolute terms z-score Most taken 20 . Use the actual number when the candidate is below the upper limit and zero is omitted from the presentation, I or SAR Record undefined and cause when less than two.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.selection.candidate_policy` | `active_whiten` | When the candidate is a positive integer, the same group of candidates is selected by each capture method from the feature that actually activates the subject and is filtered in absolute fractions after bleaching. `legacy` The number of candidates is the following: `all` To continue to apply the full-scope rules of the counterpart model. |
| `task.capture.selection.candidate_count` | `20` | When you set a positive integer number, you screen the candidate for the re-cantonment limit. `all` uses the full range of features and the relevant mode is still subject to calculation. |
| `task.capture.selection.feature_filter.enabled` | `true` | Enables the characterization filter for a mode with a numerical ceiling. |
| `task.capture.selection.feature_filter.min_native_nonzero_fraction` | `0.01` | The non-zero-activation threshold for the Neuronpedia language is 0 to 1. |
| `task.capture.selection.feature_filter.density_strict` | `true` | `true` excludes characteristics whose density is strict above the threshold. `false` also excludes characteristics that correspond to the threshold. |
| `task.capture.selection.cohort_filter.enabled` | `true` | Enables the frequency filtering of reference questions for models with numerical ceilings. |
| `task.capture.selection.cohort_filter.reference_sample_count` | `100` | Sets the number of reference questions shared by five models for standardized statistics and frequency screening. |
| `task.capture.selection.cohort_filter.zscore_threshold` | `1.0` | When the end-word standardization of a reference is activated beyond that value, the high-frequency judgement of the feature is taken into account. |
| `task.capture.selection.cohort_filter.min_question_fraction` | `0.5` | The proportion of the reference question that meets the above conditions triggers the frequency filter at least for this value. |
| `task.capture.selection.feature_selection` | `whiten` | When using the `legacy` candidate strategy, select the candidate ranking basis for `g_s`, `sar`, and `I`. |

##### 4.3.1.3 Shared answer rating

The set is also used when the SAR returns to select the full answer rating.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.answer.backend` | `full_context_tail` | Select the front-end method for scoring the answer, and use the end layer of the full context to calculate the path. The same calibration is used for comparison. |
| `task.capture.answer.reference_text` | `null` | Provides a standard text of the answer for a visible hint. The data set is executed using a sample answer. |

###### Whiten primary range

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.g_s.candidate_count` | `100` | Only `g_s` use `legacy` Sets when you are a candidate for a strategy. Whiten Primary range, must have a positive integer. |

Under the default candidate strategy, `g_s` is ranked in the common pool by Zero-Equal Characters. The output is controlled by `task.capture.output.feature_count`. When the old policy is used, the g(S) candidate ceiling covers the shared ceiling, and the numerical result is read by the original identity.

##### 4.3.1.4 Intensive interaction

Use this section of parameters after `task.capture.mode=I` selection. Only if the candidate is selected with a single characteristic behaviour and the candidate is insufficient is the candidate deficit rule applied.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.i.feature_selection` | `whiten` | When using the `legacy` candidate strategy, choose `whiten`, `native_activation` or `g_s` and may also inherit `selection.feature_selection` with an empty value. |
| `task.capture.i.g_s_shortfall` | `available` | When using the `legacy` policy and selecting a candidate by g(S), `available` uses the current number and `relax_filters` allows for a relaxed screening. |
| `task.capture.i.max_exhaustive_features` | `16` | If you have reached the ceiling, you will have to adjust the pool ceiling or overstretch the budget. |

##### 4.3.1.5 SAR Return

Select `task.capture.mode=sar` to use the parameter in this section. The sub-task is a stand-alone and pairing contribution by sampling the characteristics. Specifies the character number and location only when comparing the output fractions of a single word.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.sar.source` | `null` | Returns a trusted local original `sar.py` or a directory used by `sar`. |
| `task.capture.sar.M` | `384` | Number of feature subsets of sample regression assessment. |
| `task.capture.sar.n_fit` | `256` | The number of sub-sets to be distributed for return shall be divided into totals and assessments. |
| `task.capture.sar.p` | `0.5` | The probability of selection of a single feature when a subset of the characteristics is generated in the sample. |
| `task.capture.sar.top_pairs` | `0` | The corresponding amount is reported in pairs when positive numbers are set. |
| `task.capture.sar.pair_ranking` | `signed` | Return `sar` Pair-to-Pair Rules.`I` Sort using measured interactive ingens. |
| `task.capture.sar.readout` | `answer_mean_logprob` | Defaults to use the mean word logarithmic probability of a reference answer. The regression path also supports `logit_difference`. |
| `task.capture.sar.readout_position` | `-1` | Select the word position for the rating to read. Negative one is the last position. |
| `task.capture.sar.target_token_id` | `null` | The target number to be used by the visible hint in the word scoring is matched with the terminologist. |
| `task.capture.sar.reference_token_id` | `null` | The cross-character number used for the word fraction difference is to match the terminologist. |
| `task.capture.sar.require_affine_decode` | `true` | Requires the SAE used for the corresponding calculation to meet the conditions of the imitation decode. |

The catch description applies to selection, sorting or checking features. For example, a `task.capture.output.feature_count=1` feature can only be reported from the same candidate pool, and the original screening range is maintained. [A capture method, scoring and example](../tasks/capture/README.md)]

##### 4.3.1.6 Context comparison of RSS

The following table is used when `task.capture.mode=rss` is selected. This method includes a different context for the same target text and selects repeated and stable changes in direction.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.rss.source` | `null` | Specifies that a trusted local original RSS is achieved. |
| `task.capture.rss.contexts` | `[]` | Specifies at least two non-empty contexts, each of which forms an input to the same target text. |
| `task.capture.rss.M` | `400` | The maximum number of characteristics to enter the candidate pool by absolute change on average. |
| `task.capture.rss.K` | `20` | The maximum number of features that are ultimately retained after filtering. |
| `task.capture.rss.rho` | `0.8` | The lowest proportion of the changes that can be counted. |
| `task.capture.rss.kappa` | `0.8` | In the changed context, the lowest proportion of the main direction of change. |
| `task.capture.rss.tau` | `0.00001` | Absolute changes are included only if they are more strict than this threshold. |
| `task.capture.rss.report.dominance_ratio` | `5.0` | The characteristics selected for RSS are determined by the algorithm setting. |

##### 4.3.1.7 characterization notes

Captures can be accompanied by a source and description for the selected feature. `task.capture.neuronpedia.allow_network=false` Limiting access sources.

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.capture.neuronpedia.enabled` | `true` | Enables feature comment information. |
| `task.capture.neuronpedia.allow_network` | `true` | Controls can be accessed online for comments. |
| `task.capture.neuronpedia.cache_dir` | `null` | Visible specifies the cache path. Use the environment specified cache position when you leave space. |
| `task.capture.neuronpedia.timeout_seconds` | `15` | The number of extra-hour seconds for a single visit. |
| `task.capture.neuronpedia.total_timeout_seconds` | `120` | The total number of hours and seconds of all relevant accesses. |
| `task.capture.neuronpedia.max_download_bytes` | `67108864` | Download volume cap, 64 MiB. |
| `task.capture.neuronpedia.max_uncompressed_bytes` | `268435456` | Maximum size of post-enhanced information, 256 MiB. |

##### 4.3.1.8 Migration of old capture commands

Capture previously declared special field entry `task.capture`,For example: `task.mode=whiten` Replace with `task.capture.mode=whiten`.The following table shows the entry points where grouping or field names have been changed at the same time:.

| Keys in the old command | Keys currently in use |
| --- | --- |
| `task.filter` | `task.capture.selection.candidate_count` |
| Captured `task.capture_count` or `task.top_feature_num` | `task.capture.output.feature_count` |
| `task.sar.feature_filter.*`、`task.sar.cohort_filter.*`、`task.sar.feature_selection` | Corresponding fields are under `task.capture.selection`. |
| `task.sar.answer`、`task.sar.answer_backend` | For `task.capture.answer.reference_text` and `task.capture.answer.backend`, respectively. |
| `task.sar.i_feature_selection`、`task.sar.i_g_s_shortfall`、`task.sar.max_exhaustive_features` | For `task.capture.i.feature_selection`, `task.capture.i.g_s_shortfall` and `task.capture.i.max_exhaustive_features`, respectively. |
| SAR returns the special `task.sar.*` | Corresponding fields are under `task.capture.sar`. |
| `task.rss.*` | Corresponding fields are shown in `task.capture.rss` at `task.capture.rss.report.dominance_ratio` scale. |

The old keys of the history snapshot are mapd while reading. The original rigorous running is still checking the source version and configuration on the list.

##### 4.3.1.9 Capture Batch `matrix`

For example, to treat the whole batch only BBH, the `matrix.datasets` of the formula can be changed to `[bbh]`. [capture matrix formula](../conf/tasks/capture/task.yaml)] [capture matrix portal](../tasks/capture/matrix.py)]

| Parameters | Formula values or omitted behaviour | Purpose |
| --- | --- | --- |
| `matrix.models` | Models, SAEs and Layers in Files | Specifies where you want to capture the model. |
| `matrix.datasets` | List of data sets in the file | Specifies the data sets on which each model runs. |
| `matrix.seed` | `42` | Sets the random torrent for this batch to be executed. |
| `matrix.model_timeout_seconds` | `86400` | Limits the total length of the individual model sub-process in seconds. |
| `matrix.dataset_overrides` | Use the formulations of the data sets when omitted | Harmonized coverage of the data set for this instalment, e.g. by setting a common maximum number of samples. |

##### 4.3.1.10 SAR feature screening `sar_filter_policy`

Captures a filter file within a project to define a landmark, functional word and other syntax mode. If you need to explain why a feature is sifted, you can check the type of hit in the rule. [The catch selection rule](../conf/tasks/capture/task.yaml)]

### 4.4 Circuits `conf/tasks/circuits/`

[Open Profile Directory](../conf/tasks/circuits/)

#### 4.4.1 Circuit `conf/tasks/circuits/task.yaml`

For example, when comparing different price equivalents, you should fix a transcoder configuration that matches the model. [wire](../conf/tasks/circuits/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.circuits.approved` | `false` | The approved switch to which the circuit execution path is read is set according to the actual authorization. |
| `task.circuits.transcoder_set` | `gemma` | Select the transcoder set that matches the model. |
| `task.circuits.slug` | `gemma2b-trace` | Short names are provided for tracking products. |
| `task.timeout_seconds` | `1800` | Tracks the number of extra-hour seconds of the operation. |

### 4.5 Explanation `conf/tasks/explain/`

[Open Profile Directory](../conf/tasks/explain/)

#### 4.5.1 Explanation `conf/tasks/explain/task.yaml`

For example, the material is found to be used to generate the explanation and another to be used to check whether the explanation is applicable. [Interprets](../conf/tasks/explain/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.explain.approved` | `false` | Explains the approved switch to be read by the execution path, as actually authorized. |
| `task.explain.discovery_file` | `""` | Sample documents were found to provide a basis for the generation of the explanation. |
| `task.explain.heldout_file` | `""` | Samples are set aside for independent reconciliation of explanations. |
| `task.explain.site_id` | `""` | Indicates the model bits that need to be explained. |
| `task.explain.feature_id` | `0` | Specify the characterization number of the bit. |
| `task.explain.generator_model` | `""` | Generates model identification for interpretation. |
| `task.explain.generator_method` | `""` | Records the method name for the result. The actual generator is selected by `task.explain.generator` or the Python caller. |
| `task.explain.authorized_budget_usd` | `0.0` | Approved budget this time, in United States dollars. |
| `task.explain.estimated_cost_usd` | `0.0` | Projected cost before implementation, in United States dollars. |
| `task.explain.generator` | `""` | The empty value retains the Python call entrance with the visible name generator `saebase_generate`. |
| `task.explain.generator_max_new_tokens` | `24` | An explanation of the number of words that generate the maximum output. |

### 4.6 Generate `conf/tasks/generate/`

[Open Profile Directory](../conf/tasks/generate/)

#### 4.6.1 Generate `conf/tasks/generate/task.yaml`

For example, keep the same hint and temperature and compare the text generated by different additions. [Generating](../conf/tasks/generate/task.yaml) configuration]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.generate.intervention` | `add` | Select how to modify the selected feature, by default adding a value. |
| `task.generate.value` | `1.0` | Default increase range, as explained by the selected change. |
| `task.generate.max_new_tokens` | `8` | Maximum number of newly generated characters. |
| `task.generate.temperature` | `0.0` | Adjusts the fraction scale of the decoded sample. The default is zero temperature behaviour. |
| `task.generate.stop_at_context_limit` | `true` | Discontinue and record the reasons when the original context limits are met. |
| `task.generate.conditions` | `null` | Provide for multiple groups of primary intervention conditions, compared with the same generation mission. |
| `task.generate.feature_id` | `0` | . The number corresponds to this SAE and the model layer. |
| `task.generate.feature_selection` | `configured` | Use the specified feature by default. `capture_top` Select the catch ranking feature to read the saved catch result or to calculate it on site. You can also select a candidate from the last content word `last_active_content_token`. |
| `task.generate.top_feature_num` | `10` | The maximum number of features to be removed and changed on a case-by-case basis when a ranking selection strategy is used. |
| `task.generate.feature_ranking` | `whiten` | Select to sort by standard value or original activation, e. g. `non_whiten`. |
| `task.generate.capture_source` | `null` | Provides the already available catch results. Checks the tips, weights, bits and ranking conditions when using the reuse. |
| `task.generate.whiten_chunk_size` | `256` | The word series used to calculate the statistics required for standardization. |
| `task.resume` | `true` | Allows the task to be reused to meet the running condition. The processing of existing directories is still determined by `runtime.mode`. |
| `task.generate.inline_results` | `true` | Default to write details to the result. Read the task details file on close. |
| `task.generate.continue_on_error` | `false` | The remaining sub-items could be continued after the failure of the separate sub-items and the missing items could be addressed on a results-based basis. |
| `task.generate.reuse_prompt_observations` | `true` | Reuse a calculated observation when the full prefix of the same prompt matches. |

### 4.7 Intervention `conf/tasks/intervene/`

[Open Profile Directory](../conf/tasks/intervene/)

#### 4.7.1 Intervention `conf/tasks/intervene/task.yaml`

For example, you can read the score of the baseline and the zero condition separately using the same input and feature. [The intervention configuration](../conf/tasks/intervene/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.intervene.value` | `0.0` | The value of the individual intervention. The default is used to observe the result of setting the feature to zero. |
| `task.intervene.verify_identity` | `true` | Check to see if you read the same as the original calculation. |
| `task.intervene.protocol` | `null` | (c) The introduction of clear batch intervention conditions, budgets and comparative settings. |
| `task.intervene.feature_id` | `0` | . The number corresponds to this SAE and the model layer. |
| `task.intervene.feature_selection` | `configured` | Default uses the specified feature. `capture_top` Selects the catch ranking feature to read the saved catch result or to calculate it on site. |
| `task.intervene.top_feature_num` | `10` | The maximum number of features to be removed and changed on a case-by-case basis when a ranking selection strategy is used. |
| `task.intervene.feature_ranking` | `whiten` | Select to sort by standard value or original activation, e. g. `non_whiten`. |
| `task.intervene.capture_source` | `null` | Provides the already available catch results. Checks the tips, weights, bits and ranking conditions when using the reuse. |
| `task.intervene.whiten_chunk_size` | `256` | The word series used to calculate the statistics required for standardization. |
| `task.resume` | `true` | Allows the task to be reused to meet the running condition. The processing of existing directories is still determined by `runtime.mode`. |
| `task.intervene.inline_results` | `true` | Default to write details to the result. Read the task details file on close. |
| `task.intervene.continue_on_error` | `false` | The remaining sub-items could be continued after the failure of the separate sub-items and the missing items could be addressed on a results-based basis. |
| `task.intervene.reuse_prompt_observations` | `true` | Reuse a calculated observation when the full prefix of the same prompt matches. |

##### 4.7.1.1 Interventions and generation of batches `matrix`

It selects candidate characteristics for each model and data set and runs the assigned tasks. For example, to measure only the effects of the intervention, `matrix.tasks` is assigned to `[intervene]`. [intervention and generation matrix formula](../conf/tasks/intervene/task.yaml)] [intervention and generation batch portal](../tasks/intervene/matrix.py)

| Parameters | Formula Value | Purpose |
| --- | --- | --- |
| `matrix.models`、`matrix.datasets` | List of model bits and data sets in files | Defines the group of experiments for this batch. |
| `matrix.seed` | `42` | Sets the random torrent for this batch to be executed. |
| `matrix.model_timeout_seconds` | `86400` | Limits the total length of the individual model sub-process in seconds. |
| `matrix.tasks` | `[intervene, generate]` | Select the operational intervention, generation or both. |
| `matrix.feature_ranking` | `whiten` | Lists the candidate characteristics with a standardized activation value. |
| `matrix.feature_count` | `10` | Specifies the number of candidates for each entry that you want to intervene on a case-by-case basis. |
| `matrix.intervention_value` | `0.0` | Sets the intervention value for the batch. |
| `matrix.capture_source_tag` | `cohort_ranking` | Specifies the catch batch that you can try to reuse. The portal still checks if the actual results match this input. |
| `matrix.generation.intervention` | `ablate` | Decides how to modify the features to be used in their continuation. |
| `matrix.generation.max_new_tokens` | `8` | Limits the maximum number of new characters to continue. |
| `matrix.generation.temperature` | `0.0` | Sets the decoded temperature that is generated. |
| `matrix.max_passes` | `2` | Limit the number of batches. |
| `matrix.tag` | `whiten_feature_effects` | Name the experiment and locate the batch report. |

##### 4.7.1.2 Pythia characterization intervention `presets.pythia_feature_edits`

For `saebase-intervene --config` Read. Protocol declaration model, SAE,The word number of the sample, how to modify it, and how to measure the budget. `conditions` It's a way to clearly set out the conditions for comparison..[Pythia Intervention formulation](../conf/tasks/intervene/task.yaml)

### 4.8 precheck `conf/tasks/precheck/`

[Open Profile Directory](../conf/tasks/precheck/)

#### 4.8.1 precheck `conf/tasks/precheck/task.yaml`

For the purpose of confirming the combination and the conditions of implementation. For example, only one SAE will be checked first, and the selective energy model will compare the volume of the actual task. [precheck configuration](../conf/tasks/precheck/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.precheck.mode` | `sae` | Inspection SAE.The other range is `task`,`all`,`performance`. |
| `task.precheck.resource_only` | `false` | Search only SAE resources for real times, skip the language model. |
| `task.precheck.worker_mode` | `isolated` | The default quarantine is performed. `in_process` is used to provide an external timeout for a credible batch. |
| `task.precheck.sae_list` | `default` | Use the model default SAE, with the option `all` or the visible list. |
| `task.precheck.check_tasks` | `[]` | Specifies the list of tasks to be checked, to be filled in according to the checkable tasks registered under this item. |
| `task.precheck.auto_setup` | `true` | The preparation process is accessible when the required resources are lacking. |
| `task.precheck.allow_network` | `true` | The preparation process has access to the network. |
| `task.precheck.allow_install` | `true` | (c) Whether the preparation process can be installed without a lack of reliance. |
| `task.precheck.allow_download` | `true` | Whether the preparatory process can download the resources that are lacking. |
| `task.precheck.max_download_bytes` | `17179869184` | Missing document download budget, 16 GiB. |
| `task.precheck.prepare_timeout_seconds` | `1800` | Resource readiness time limit, in seconds. |
| `task.precheck.dataset_prepare_timeout_seconds` | `1800` | Data set preparation time limit, in seconds. |
| `task.precheck.check_sample_count` | `32` | A true input of the candidates for the survey is independent of the official number of questions in the data set. |
| `task.precheck.sampling_seed` | `42` | Select the random seed that you want to explore the sample. |
| `task.precheck.max_seconds` | `120` | Check the time budget, in seconds. |
| `task.precheck.max_samples` | `2` | Maximum sample size for small-scale inspection. |
| `task.precheck.max_tokens` | `32` | The word limit for small-scale checks. |
| `task.precheck.task_configs` | `{}` | Each target receives the full task profile under `task`. For example, by `+task.precheck.task_configs.capture.capture.mode=whiten` specify the catch check. |

The performance precheck default compares the captured whiteing, original activation, single characterizations and quantities in four ways. The `+task.precheck.capture_modes=[g_s,I]` can be used when narrowing the scope. The pre-checking instructions are used to explain the range of checks selected and to respond. For example, the candidate batch, time-consuming and numerical match in the performance report can be seen before the official capture. [Preview operation and report description](../tasks/precheck/README.md)]

##### 4.8.1.1 SAE common settings and name protocols `sae_checks`

When you want to adjust the size of an agreement, equipment or sample, you find the actual preset. [Baseline formula catalogue](../conf/tasks/precheck/task.yaml)]

### 4.9 SAE Training `conf/tasks/train/`

[Open Profile Directory](../conf/tasks/train/)

#### 4.9.1 SAE Training `conf/tasks/train/task.yaml`

For example, when you change the range of data involved in training, check the training plan and record the product with a new experiment name. [SAE training mission configuration](../conf/tasks/train/task.yaml)]

| Parameters | Default value | Purpose and conditions |
| --- | --- | --- |
| `task.train.approved` | `false` | SAE training path reads the approved switch, as actually authorized. |
| `task.train.train_plan` | `null` | Provides a clear SAE training plan and determines training input, objectives and budget. |
| `task.train.allow_large` | `false` | Controls allow training beyond the conventional boundaries. |
| `task.timeout_seconds` | `1800` | SAE training operation timeout seconds. |

##### 4.9.1.1 Training programme `task.train.train_plan`

A plan defines the data, objectives and budget for this training. For example, a Top-K SAE is trained to determine the exact snapshots and reading points of the underlying model and data set, and then to determine the width of the encoder, the maximum length of each input, and the number of steps. The model and data set selected by the command line is used to run the grouping and to train to actually read the snapshots specified in the table below.

| Parameters | Use and Defaults |
| --- | --- |
| `task.train.train_plan.backend` | It will be filled out. Select `saelens` or `sparsify`. The backend determines the structure and how to run the support. |
| `task.train.train_plan.model_path`、`task.train.train_plan.model_revision` | It is necessary to fill in the location of the underlying model and the fixed submission version. |
| `task.train.train_plan.dataset_path`、`task.train.train_plan.dataset_revision` | It is necessary to fill in the location of the training data and the fixed submission version. |
| `task.train.train_plan.hookpoint` | Always fill in, specify the internal bits of the active model. |
| `task.train.train_plan.architecture` | It will be filled out in the structure supported by the selected backend. Sparsify accepts `topk` and `transcoder`. |
| `task.train.train_plan.d_in`、`task.train.train_plan.d_sae`、`task.train.train_plan.k` | To fill, specify the number of active dimensions, the number of characteristics expressed in thinness and the Top-K thinness, respectively. |
| `task.train.train_plan.max_examples`、`task.train.train_plan.steps`、`task.train.train_plan.batch_size`、`task.train.train_plan.ctx_len` | It will be necessary to specify the maximum size of the sample, the number of training steps, the lot and the length of the input context. |
| `task.train.train_plan.run_name` | You're gonna have to fill in, train in the running name for the back end. |
| `task.train.train_plan.device`、`task.train.train_plan.distributed_gpus` | The backend defaults to use CUDA and a graphic card. Only adjusts the device or parallel number when supported by the selected backend. |
| `task.train.train_plan.resume_checkpoint` | Default empty, specify the existing check points that you want to read to continue training. |
| `task.train.train_plan.model_dtype`、`task.train.train_plan.sae_dtype` | The default is `float32`. Select the numerical precision of the basic model and SAELens training, respectively. |
| `task.train.train_plan.store_batch_size_prompts`、`task.train.train_plan.n_batches_in_buffer` | Default empty, determined by the backend. Applys to the amount of prompts and buffers that are activated through SAELENS control. |

[Definition of training programme](../tasks/train/execution/training.py)Indicate which structures and budget combinations are available. Sparsify Plan selection Top-K When the structure is in place, the required snapshot, bits and graphic settings can be checked at the definition point.

## 5 Public interface and running `conf/base/`

**Purpose.** access to SAE interfaces, operating environments, default test names and data roots.

[Open Profile Directory](../conf/base/)

### 5 SAE interface formulation `conf/base/interfaces/`

[Open Profile Directory](../conf/base/interfaces/)

#### 5.1.1 Encoder preset `conf/base/interfaces/sae_profiles.yaml`

The SAE formula saves the encoder weight source, compatible model, and layer definition, etc... When checking which encoder results a given feature results from, check the corresponding presets from here. For example, check whether the Gemma 2 default encoder model and 25 layers are consistent with the results list. [SAE] formulation](../conf/base/interfaces/sae_profiles.yaml)

`model=openresearch_gemma2_2b_base` Select Basic Model `google/gemma-2-2b`,Formula statement loads as required. Matchable SAE It's written in... `sae_profiles`,Default Selection `gemma2_2b_pt_l25_canonical`,Default Read No. 25 Layer. The model formulation lists the weights and the default selections allowed for the mix., SAE Formula to read the level of the specific encoder and the prepared hint requirements. Replace `model=qwen35_4b` , the default encoder is changed to `qwen35_4b_matryoshka_l15_65k`,I don't think so. 15 Layer by SAE Formula declaration.

### 6 Public Operations Configuration `conf/base/runtime/`

[Open Profile Directory](../conf/base/runtime/)

#### 6 Public run formulation `conf/base/runtime/runtime.yaml`

The same file management interpreter, experimental name and data root. [Opens the running formula](../conf/base/runtime/runtime.yaml)]

##### 6.1.1.1 Current and modelled environment `environments`

This file saves the default interpreter, the critical dependent version, and the environment that the model needs. When you switch to another environment, check the target model and the dependencies from this file, and then the actual interpreter and the software version from the running inventory. The results are based on the actual implementation environment. [Environmental configuration](../conf/base/runtime/runtime.yaml)] [Environmental document](environment/README.md)]

##### 6.1.1.2 Experiment name and default value `experiment`

The `experiment` section saves the test name and single run mark. The same experiment keeps a double measurement in an independent running directory.

| Parameters | Default value | Purpose |
| --- | --- | --- |
| `experiment.sub_name` | `null` | The main method not yet expressed under the task is omitted, e.g. `whiten` under the catch task. The purpose or key condition of the explicit name writing study, e.g. `answer_stability`. Seeds, dates and regular parameters are recorded in the running record. |
| `experiment.run_id` | `null` | Optional performer, e. g. `noise_control`. The common directory is used for the same purpose. The old configuration and the product are stored in the directory `executions/`. Runs on the current execution configuration. |
[Experiment Default](../conf/base/runtime/runtime.yaml)

##### 6.1.1.3 Root and data directories `paths`

In paragraph `paths`, `paths.root` defaults for the startup directory, `paths.data_dir` defaults for `data/` below it. The data on the external disk can be marked to set the data root. The output is still organized according to the current path to run the project. [Process formulation](../conf/base/runtime/runtime.yaml)]

## Current value in profile

The following are automatically updated from the actual configuration file. The usage of the documents and the meaning of the parameters are shown in the corresponding subsection.

<!-- BEGIN CONFIG_VALUES -->

<details>
<summary>base/configuration/__init__.py</summary>

[Open Full File](../base/configuration/__init__.py)

<!-- SHA256: 85e92e4d642a16cb905ddbe058ff4d064dd4b1a6a26f653ad23710542cdb9303 -->
<pre>&quot;&quot;&quot;Named component configurations for source and installed SAEBase.

Catalogs keep shared defaults once. Hydra owns selection and composition;
load_group provides the same presets to direct component consumers.
&quot;&quot;&quot;

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def workspace_at(directory: Path) -&gt; bool:
    &quot;&quot;&quot;Recognize a source workspace independently of the launch directory.&quot;&quot;&quot;
    return (
        (directory / &quot;pyproject.toml&quot;).is_file()
        and (directory / &quot;base/application/commands.py&quot;).is_file()
        and (directory / &quot;conf/config.yaml&quot;).is_file()
    )


def configuration_root() -&gt; Path:
    &quot;&quot;&quot;Locate physical recipes in a checkout and data files in distributions.&quot;&quot;&quot;
    source = Path(__file__).resolve().parents[2]
    if workspace_at(source):
        return source / &quot;conf&quot;
    return Path(__file__).absolute().parent


def launch_directory() -&gt; Path:
    &quot;&quot;&quot;Return the caller&#x27;s directory before Hydra changes into the run directory.&quot;&quot;&quot;
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    return Path(get_original_cwd() if HydraConfig.initialized() else Path.cwd()).resolve()


def default_workspace() -&gt; str | None:
    &quot;&quot;&quot;Prefer the caller&#x27;s SAEBase checkout, then the checkout supplying the package.&quot;&quot;&quot;
    launch = launch_directory()
    for directory in (launch, *launch.parents, configuration_root().parent):
        if workspace_at(directory):
            return str(directory)
    return None


def _default_data_directory(root: str | None) -&gt; str | None:
    return str(((launch_directory() / root).resolve() / &quot;assets/dataset_preparation/prompt_sets/files&quot;).resolve()) if root is not None else None


def _prepared_dataset_dir(root: str, name: str) -&gt; str:
    from saebase.datasets.artifacts import prepared_dataset_root

    return str(prepared_dataset_root(root, name))


OmegaConf.register_new_resolver(&quot;saebase_prepared_dir&quot;, _prepared_dataset_dir, replace=True)
OmegaConf.register_new_resolver(&quot;saebase_workspace&quot;, default_workspace, replace=True)
OmegaConf.register_new_resolver(&quot;saebase_data_dir&quot;, _default_data_directory, replace=True)


CATALOG_PATHS = {
    &quot;model&quot;: &quot;models/catalog.yaml&quot;,
    &quot;sae&quot;: &quot;base/interfaces/sae_profiles.yaml&quot;,
    &quot;dataset&quot;: &quot;datasets/catalog.yaml&quot;,
    &quot;prompt_source&quot;: &quot;datasets/catalog.yaml&quot;,
    &quot;benchmark&quot;: &quot;tasks/precheck/task.yaml&quot;,
}


def configuration_location(source: str | Path) -&gt; tuple[Path, str]:
    &quot;&quot;&quot;Split a physical configuration path and an optional named section.&quot;&quot;&quot;
    path = Path(source)
    filename, separator, section = str(path).rpartition(&quot;#&quot;)
    if separator and Path(filename).suffix.lower() in {&quot;.json&quot;, &quot;.yml&quot;, &quot;.yaml&quot;} and not path.is_file():
        if not section or any(not part.isidentifier() for part in section.split(&quot;.&quot;)):
            raise ValueError(f&quot;Invalid configuration section: {source}&quot;)
        return Path(filename), section
    return path, &quot;&quot;


def read_configuration(source: str | Path, *, resolve: bool = False):
    &quot;&quot;&quot;Read a standalone JSON/YAML recipe or a named mapping within a shared file.&quot;&quot;&quot;
    path, section = configuration_location(source)
    if path.suffix.lower() == &quot;.json&quot;:
        data = json.loads(path.read_text(encoding=&quot;utf-8&quot;))
    elif path.suffix.lower() in {&quot;.yaml&quot;, &quot;.yml&quot;}:
        data = (
            OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            if resolve
            else yaml.safe_load(path.read_text(encoding=&quot;utf-8&quot;))
        )
    else:
        raise ValueError(f&quot;Configuration must use JSON or YAML: {path}&quot;)
    if section:
        for key in section.split(&quot;.&quot;):
            if not isinstance(data, dict) or key not in data:
                raise ValueError(f&quot;Configuration section {section!r} is missing in {path}&quot;)
            data = data[key]
    return data


def bundled_configuration(relative: str, section: str = &quot;&quot;) -&gt; Path:
    &quot;&quot;&quot;Locate a bundled file in a checkout or an installed package.&quot;&quot;&quot;
    path = configuration_root() / relative
    return Path(f&quot;{path}#{section}&quot;) if section else Path(str(path))


def _catalog(group):
    config = OmegaConf.load(configuration_root() / CATALOG_PATHS[group])
    sections = {&quot;dataset&quot;: &quot;dataset&quot;, &quot;prompt_source&quot;: &quot;prompt_source&quot;, &quot;benchmark&quot;: &quot;sae_checks&quot;}
    return config[sections[group]] if group in sections else config


def load_task(name):
    &quot;&quot;&quot;Read a task recipe from the directory matching its implementation.&quot;&quot;&quot;
    config = OmegaConf.load(bundled_configuration(f&quot;tasks/{name}/task.yaml&quot;))
    return OmegaConf.masked_copy(config, [&quot;task&quot;])


def group_names(group):
    return tuple(_catalog(group).presets)


def load_group(group, name):
    catalog = _catalog(group)
    if name not in catalog.presets:
        raise KeyError(f&quot;Unknown {group} preset {name!r}; available: {list(catalog.presets)}&quot;)
    return OmegaConf.create({group: OmegaConf.merge(catalog.defaults, catalog.presets[name])})


def register_configs():
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    for group in (&quot;model&quot;, &quot;sae&quot;, &quot;dataset&quot;, &quot;prompt_source&quot;):
        catalog = _catalog(group)
        for name, values in catalog.presets.items():
            node = {group: OmegaConf.merge(catalog.defaults, values)}
            store.store(group=group, name=name, node=node, package=&quot;_global_&quot;, provider=&quot;saebase&quot;)

    for directory in (configuration_root() / &quot;tasks&quot;).iterdir():
        if (directory / &quot;task.yaml&quot;).is_file():
            store.store(
                group=&quot;task&quot;,
                name=directory.name,
                node=load_task(directory.name),
                package=&quot;_global_&quot;,
                provider=&quot;saebase&quot;,
            )

    for group, name, path in (
        (&quot;paths&quot;, &quot;default&quot;, &quot;base/runtime/runtime.yaml&quot;),
        (&quot;experiment&quot;, &quot;default&quot;, &quot;base/runtime/runtime.yaml&quot;),
        (&quot;evaluation&quot;, &quot;activation_health&quot;, &quot;tasks/benchmark/task.yaml&quot;),
    ):
        store.store(
            group=group,
            name=name,
            node={group: OmegaConf.load(configuration_root() / path)[group]},
            package=&quot;_global_&quot;,
            provider=&quot;saebase&quot;,
        )


register_configs()</pre>

</details>

<details>
<summary>conf/base/interfaces/sae_profiles.yaml</summary>

[Open Full File](../conf/base/interfaces/sae_profiles.yaml)

<!-- SHA256: b49addfa0462b548338b8bc262e802875fa06705f798deb7e61f2237d1ca4586 -->
<pre># Purpose and main function: Save SAE Public default and name preset for resources.
# Scope: Hydra Component selection, and conf.load_group Direct Read.
defaults:
  device: cuda
presets:
  gemma1b_it_l25_native_big:
    name: gemma1b_it_l25_native_big
    layer: 25
    require_prepared_prompts: true
  gemma1b_pt_l25_transfer_big:
    name: gemma1b_pt_l25_transfer_big
    layer: 25
    require_prepared_prompts: true
  gemma1b_pt_l25_transfer_small:
    name: gemma1b_pt_l25_transfer_small
    layer: 25
    require_prepared_prompts: true
  gemma1b_scope_resid:
    name: gemma1b_scope_resid
    layer: null
    require_prepared_prompts: true
  gemma270m_scope_resid:
    name: gemma270m_scope_resid
    layer: null
    require_prepared_prompts: true
  gemma2_2b_it_matryoshka_l12_32k:
    name: gemma2_2b_it_matryoshka_l12_32k
    layer: 12
    require_prepared_prompts: true
  gemma2_2b_it_scope_resid_l12:
    name: gemma2_2b_it_scope_resid_l12
    layer: 12
    require_prepared_prompts: true
  gemma2_2b_pt_l25_canonical:
    name: gemma2_2b_pt_l25_canonical
    layer: 25
    require_prepared_prompts: true
  gemma2_9b_it_scope_resid:
    name: gemma2_9b_it_scope_resid
    layer: 20
    require_prepared_prompts: true
  gemma2_9b_it_scope_resid_l31_131k:
    name: gemma2_9b_it_scope_resid_l31_131k
    layer: 31
    require_prepared_prompts: true
  gemma2_9b_pt_scope_resid:
    name: gemma2_9b_pt_scope_resid
    layer: 20
    require_prepared_prompts: true
  gemma3_12b_it_scope_resid:
    name: gemma3_12b_it_scope_resid
    layer: 24
    require_prepared_prompts: true
  gemma3_12b_pt_scope_resid:
    name: gemma3_12b_pt_scope_resid
    layer: 24
    require_prepared_prompts: true
  gemma3_27b_it_scope_resid:
    name: gemma3_27b_it_scope_resid
    layer: 31
    require_prepared_prompts: true
  gemma3_27b_pt_scope_resid:
    name: gemma3_27b_pt_scope_resid
    layer: 31
    require_prepared_prompts: true
  gemma3_4b_it_scope_resid:
    name: gemma3_4b_it_scope_resid
    layer: 29
    require_prepared_prompts: true
  gemma3_4b_pt_scope_resid:
    name: gemma3_4b_pt_scope_resid
    layer: 29
    require_prepared_prompts: true
  gpt2small_fs_12288_l8:
    name: gpt2small_fs_12288_l8
    layer: null
    require_prepared_prompts: true
  gpt2small_fs_24576_l8:
    name: gpt2small_fs_24576_l8
    layer: null
    require_prepared_prompts: true
  gpt2small_fs_6144_l8:
    name: gpt2small_fs_6144_l8
    layer: null
    require_prepared_prompts: true
  gpt2small_res_jb_l8:
    name: gpt2small_res_jb_l8
    layer: null
    require_prepared_prompts: true
  llama31_8b_instruct_andyrdt_l19_k64:
    name: llama31_8b_instruct_andyrdt_l19_k64
    layer: 19
    require_prepared_prompts: false
  llama31_8b_instruct_andyrdt_l23_k64:
    name: llama31_8b_instruct_andyrdt_l23_k64
    layer: 23
    require_prepared_prompts: false
  llama31_8b_lxr_8x:
    name: llama31_8b_lxr_8x
    layer: 21
    require_prepared_prompts: true
  mistral7b_instruct_tc_l16_131k:
    name: mistral7b_instruct_tc_l16_131k
    layer: 16
    require_prepared_prompts: true
  none:
    name: none
    layer: null
    require_prepared_prompts: false
  pythia410m_mlp_topk:
    name: pythia410m_mlp_topk
    layer: null
    require_prepared_prompts: true
  pythia70m_resid_sm:
    name: pythia70m_resid_sm
    layer: null
    require_prepared_prompts: true
  pythia70m_saebench_gated_l3:
    name: pythia70m_saebench_gated_l3
    layer: null
    require_prepared_prompts: true
  qwen35_27b_resid_w80k:
    name: qwen35_27b_resid_w80k
    layer: 32
    require_prepared_prompts: true
  qwen35_2b_resid_w32k:
    name: qwen35_2b_resid_w32k
    layer: 12
    require_prepared_prompts: true
  qwen35_4b_matryoshka_l15_65k:
    name: qwen35_4b_matryoshka_l15_65k
    layer: 15
    require_prepared_prompts: true
  qwen35_9b_resid_w64k:
    name: qwen35_9b_resid_w64k
    layer: 16
    require_prepared_prompts: true
  qwen3_1_7b_ak_resid_l14_65k:
    name: qwen3_1_7b_ak_resid_l14_65k
    layer: 14
    require_prepared_prompts: true
  qwen3_1_7b_base_resid_w32k:
    name: qwen3_1_7b_base_resid_w32k
    layer: 14
    require_prepared_prompts: true
  qwen3_8b_resid_w64k:
    name: qwen3_8b_resid_w64k
    layer: 18
    require_prepared_prompts: true
  r1_distill_llama8b_llamascope_l15_openr1:
    name: r1_distill_llama8b_llamascope_l15_openr1
    layer: 15
    require_prepared_prompts: true
  r1_distill_llama8b_llamascope_l23:
    name: r1_distill_llama8b_llamascope_l23
    layer: 23
    require_prepared_prompts: true
  r1_distill_qwen15b_resid_l20:
    name: r1_distill_qwen15b_resid_l20
    layer: 20
    require_prepared_prompts: true</pre>

</details>

<details>
<summary>conf/base/runtime/runtime.yaml</summary>

[Open Full File](../conf/base/runtime/runtime.yaml)

<!-- SHA256: 149ecfbe3cbaef811fa63f0dcb8d4ef42a1bcb46284d978c0c961d0b36bf95d9 -->
<pre># Purpose: select interpreters and define shared input and run defaults.
# Scope: local model launches and Hydra runtime composition.

environments:
  # Pilot interpreter and identified key dependent version; by Runtime Harmonization of selection and inspection.
  default:
    python: /venv/main/bin/python
    packages:
      torch: 2.11.0+cu128
      transformers: 4.57.6
      huggingface-hub: 0.36.2
      tokenizers: 0.22.2
  models:
    qwen35_9b: &amp;qwen35
      python: /workspace/envs/saebase/qwen35/bin/python
      packages:
        torch: 2.11.0+cu128
        transformers: 5.23.0
        huggingface-hub: 1.33.0
        tokenizers: 0.23.3
      architectures: [qwen3_5]
    qwen35_4b: *qwen35
    qwen35_2b: *qwen35
    qwen35_27b: *qwen35

paths:
  root: ${saebase_workspace:}
  data_dir: ${saebase_data_dir:${paths.root}}

experiment:
  sub_name: null
  run_id: null</pre>

</details>

<details>
<summary>conf/config.yaml</summary>

[Open Full File](../conf/config.yaml)

<!-- SHA256: 18be071ec5f428c6fff47e5fe17017d01fa2e9d298b00ca4d9e2a665883ea7c1 -->
<pre># SAEBase&#x27;s single master configuration. Component groups contain only their own settings.
defaults:
  - dataset: ioi
  - model: openresearch_gemma2_2b_base
  - paths: default
  - task: capture
  - experiment: default
  # Hydra&#x27;s default file handler creates ${output_dir}/train.log before
  # saebase.runtime.prepare_run. That empty file is then treated as a prior artifact.
  # Console logging leaves a clean leaf; task logs still go under logs/.
  - override hydra/job_logging: stdout
  - _self_

# Random seed for LLM SAE runs
seed: 42

# Run-directory cleanup is independent of the selected research task.
runtime:
  device: auto  # Shared model placement for the normal task dispatcher.
  mode: null  # null defaults to new; also accepts resume and overwrite
  numerics: vllm  # One stable GPU policy for precheck and tasks; native reproduces the old policy.
  cpu_threads: 4  # Bound Torch/BLAS CPU pools; uses the same setting in direct and matrix runs.

hydra:
  run:
    dir: ${saebase_run_path:leaf}
  sweep:
    dir: ${saebase_run_path:leaf}
    subdir: .
  output_subdir: null
  job:
    chdir: true</pre>

</details>

<details>
<summary>conf/datasets/catalog.yaml</summary>

[Open Full File](../conf/datasets/catalog.yaml)

<!-- SHA256: 9cf19111f843e586de1cb050a79345abdfe71e0573e3fa897b85d0579a9012c3 -->
<pre># Purpose: dataset selection and prompt preparation defaults and presets.
# Scope: dataset preparation and Hydra component selection.

dataset:
  # Purpose and main function: Public defaults and name presets for the preservation of data sets.
  # Scope: Hydra Component selection, and conf.load_group Direct Read.
  defaults:
    data_path: ${saebase_prepared_dir:${paths.data_dir},${dataset.name}}
    prompt_file: &#x27;&#x27;
  presets:
    arc_challenge:
      name: ARCChallenge
      source_config: ARC-Challenge
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    bbh:
      name: BBH
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    counterfact:
      name: CounterFact
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    greater_than:
      name: GreaterThan
    gsm8k:
      name: GSM8K
      source_config: main
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    harmbench:
      name: HarmBench
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    hotpotqa:
      name: HotpotQA
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
      source_config: distractor
    humaneval:
      name: HumanEval
      source_config: openai_humaneval
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    ioi:
      name: IOI
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    mmlu_pro:
      name: MMLUPro
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    ravel:
      name: RAVEL
      source_config: city
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    simple_copying:
      name: SimpleCopying
    truthfulqa:
      name: TruthfulQA
      source_config: multiple_choice
      sample_count: 500
      sampling_seed: 42
      use_original_split: false
    xstest:
      name: XSTest
      sample_count: 500
      sampling_seed: 42
      use_original_split: false

prompt_source:
  # Purpose and main function: Save public defaults and name presets of the source of the hint.
  # Scope: Hydra Component selection, and conf.load_group Direct Read.
  defaults: {}
  presets:
    arc_challenge:
      name: ARCChallenge
      data_path: ${saebase_prepared_dir:${paths.data_dir},ARCChallenge}
    bbh:
      name: BBH
      data_path: ${saebase_prepared_dir:${paths.data_dir},BBH}
      prepare:
        n_generate: 80
        min_kept: 10
        seed: 0
        source_path: /workspace/data/BBH
    counterfact:
      name: CounterFact
      data_path: ${saebase_prepared_dir:${paths.data_dir},CounterFact}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0
    gpqa_extended:
      name: GPQAExtended
      data_path: ${saebase_prepared_dir:${paths.data_dir},GPQAExtended}
    greater_than:
      name: GreaterThan
      data_path: ${saebase_prepared_dir:${paths.data_dir},GreaterThan}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0
    ioi:
      name: IOI
      data_path: ${saebase_prepared_dir:${paths.data_dir},IOI}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0
    livecodebench:
      name: LiveCodeBench
      data_path: ${saebase_prepared_dir:${paths.data_dir},LiveCodeBench}
    math500:
      name: MATH500
      data_path: ${saebase_prepared_dir:${paths.data_dir},MATH500}
    medqa:
      name: MedQA
      data_path: ${saebase_prepared_dir:${paths.data_dir},MedQA}
    mmlu_pro:
      name: MMLUPro
      data_path: ${saebase_prepared_dir:${paths.data_dir},MMLUPro}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0
        split: test
        categories: all
        n_shot: 0
        scoring: direct
        max_prompt_tokens: 2048
        drop_na_options: true
        balance_per_category: false
        source_path: /workspace/data/MMLU-Pro
        hf_name: TIGER-Lab/MMLU-Pro
    simple_copying:
      name: SimpleCopying
      data_path: ${saebase_prepared_dir:${paths.data_dir},SimpleCopying}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0
    subject_verb_agreement:
      name: SubjectVerbAgreement
      data_path: ${saebase_prepared_dir:${paths.data_dir},SubjectVerbAgreement}
      prepare:
        n_generate: 300
        min_kept: 20
        seed: 0</pre>

</details>

<details>
<summary>conf/models/catalog.yaml</summary>

[Open Full File](../conf/models/catalog.yaml)

<!-- SHA256: ae5a13cb67294c2ee501661fe8e2a1572b945e992287e5351f6f77b4d798224d -->
<pre># Purpose and main function: to save the model ’ s public default and name preset.
# Scope: Hydra Component selection, and conf.load_group Direct Read.
defaults:
  device: auto
presets:
  gemma1b:
    name: gemma1b
    pretrained_name: google/gemma-3-1b-pt
    family: gemma
    sae_profiles:
    - gemma1b_pt_l25_transfer_big
    - gemma1b_pt_l25_transfer_small
    - gemma1b_scope_resid
    default_saes:
    - gemma1b_scope_resid
    default_layers:
      gemma1b_scope_resid: 13
  gemma1b_it:
    name: gemma1b_it
    pretrained_name: google/gemma-3-1b-it
    family: gemma
    sae_profiles:
    - gemma1b_it_l25_native_big
    default_saes:
    - gemma1b_it_l25_native_big
  gemma270m:
    name: gemma270m
    pretrained_name: gemma-3-270m
    family: gemma
    sae_profiles:
    - gemma270m_scope_resid
    default_saes:
    - gemma270m_scope_resid
  gemma2_2b_base:
    name: gemma2_2b_base
    pretrained_name: google/gemma-2-2b
    family: gemma2
    sae_profiles:
    - gemma2_2b_it_matryoshka_l12_32k
    - gemma2_2b_it_scope_resid_l12
    - gemma2_2b_pt_l25_canonical
    default_saes:
    - gemma2_2b_pt_l25_canonical
    default_layers:
      gemma2_2b_pt_l25_canonical: 25
  gemma2_2b_it:
    name: gemma2_2b_it
    pretrained_name: google/gemma-2-2b-it
    family: gemma2
    sae_profiles:
    - gemma2_2b_pt_l25_canonical
    default_saes:
    - gemma2_2b_pt_l25_canonical
  gemma2_9b_it:
    name: gemma2_9b_it
    pretrained_name: google/gemma-2-9b-it
    family: gemma
    sae_profiles:
    - gemma2_9b_it_scope_resid
    - gemma2_9b_it_scope_resid_l31_131k
    default_saes:
    - gemma2_9b_it_scope_resid
  gemma2_9b_pt:
    name: gemma2_9b_pt
    pretrained_name: google/gemma-2-9b
    family: gemma
    sae_profiles:
    - gemma2_9b_pt_scope_resid
    default_saes:
    - gemma2_9b_pt_scope_resid
  gemma3_12b_it:
    name: gemma3_12b_it
    pretrained_name: google/gemma-3-12b-it
    family: gemma
    sae_profiles:
    - gemma3_12b_it_scope_resid
    default_saes:
    - gemma3_12b_it_scope_resid
    default_layers:
      gemma3_12b_it_scope_resid: 24
  gemma3_12b_pt:
    name: gemma3_12b_pt
    pretrained_name: google/gemma-3-12b-pt
    family: gemma
    sae_profiles:
    - gemma3_12b_pt_scope_resid
    default_saes:
    - gemma3_12b_pt_scope_resid
  gemma3_27b_it:
    name: gemma3_27b_it
    pretrained_name: google/gemma-3-27b-it
    family: gemma
    sae_profiles:
    - gemma3_27b_it_scope_resid
    default_saes:
    - gemma3_27b_it_scope_resid
  gemma3_27b_pt:
    name: gemma3_27b_pt
    pretrained_name: google/gemma-3-27b-pt
    family: gemma
    sae_profiles:
    - gemma3_27b_pt_scope_resid
    default_saes:
    - gemma3_27b_pt_scope_resid
  gemma3_4b_it:
    name: gemma3_4b_it
    pretrained_name: google/gemma-3-4b-it
    family: gemma
    sae_profiles:
    - gemma3_4b_it_scope_resid
    default_saes:
    - gemma3_4b_it_scope_resid
  gemma3_4b_pt:
    name: gemma3_4b_pt
    pretrained_name: google/gemma-3-4b-pt
    family: gemma
    sae_profiles:
    - gemma3_4b_pt_scope_resid
    default_saes:
    - gemma3_4b_pt_scope_resid
    default_layers:
      gemma3_4b_pt_scope_resid: 17
  gpt2small:
    name: gpt2small
    pretrained_name: gpt2-small
    hub_repo_id: openai-community/gpt2
    family: gpt2
    sae_profiles:
    - gpt2small_fs_12288_l8
    - gpt2small_fs_24576_l8
    - gpt2small_fs_6144_l8
    - gpt2small_res_jb_l8
    default_saes:
    - gpt2small_res_jb_l8
    default_layers:
      gpt2small_res_jb_l8: 8
  llama31_8b:
    name: llama31_8b
    pretrained_name: meta-llama/Llama-3.1-8B
    family: llama
    sae_profiles:
    - llama31_8b_lxr_8x
    default_saes:
    - llama31_8b_lxr_8x
    default_layers:
      llama31_8b_lxr_8x: 21
  llama31_8b_instruct:
    name: llama31_8b_instruct
    pretrained_name: meta-llama/Llama-3.1-8B-Instruct
    family: llama
    sae_profiles:
    - llama31_8b_instruct_andyrdt_l19_k64
    - llama31_8b_instruct_andyrdt_l23_k64
    default_saes:
    - llama31_8b_instruct_andyrdt_l19_k64
  mistral7b_instruct:
    name: mistral7b_instruct
    pretrained_name: mistralai/Mistral-7B-Instruct-v0.1
    family: mistral
    sae_profiles:
    - mistral7b_instruct_tc_l16_131k
    default_saes:
    - mistral7b_instruct_tc_l16_131k
  openresearch_gemma2_2b_base:
    name: gemma2_2b_base
    hf_id: google/gemma-2-2b
    lazy: true
    sae_profiles:
    - gemma2_2b_it_matryoshka_l12_32k
    - gemma2_2b_it_scope_resid_l12
    - gemma2_2b_pt_l25_canonical
    default_saes:
    - gemma2_2b_pt_l25_canonical
    default_layers:
      gemma2_2b_pt_l25_canonical: 25
  pythia410m:
    name: pythia410m
    pretrained_name: pythia-410m-deduped
    family: pythia
    sae_profiles:
    - pythia410m_mlp_topk
    default_saes:
    - pythia410m_mlp_topk
  pythia70m:
    name: pythia70m
    pretrained_name: pythia-70m-deduped
    family: pythia
    sae_profiles:
    - pythia70m_resid_sm
    - pythia70m_saebench_gated_l3
    default_saes:
    - pythia70m_resid_sm
  qwen25_3b:
    name: qwen25_3b
    pretrained_name: Qwen/Qwen2.5-3B
    family: qwen
    sae_profiles: []
    default_saes: []
  qwen25_3b_it:
    name: qwen25_3b_it
    pretrained_name: Qwen/Qwen2.5-3B-Instruct
    family: qwen
    sae_profiles: []
    default_saes: []
  qwen35_27b:
    name: qwen35_27b
    pretrained_name: Qwen/Qwen3.5-27B
    family: qwen
    sae_profiles:
    - qwen35_27b_resid_w80k
    default_saes:
    - qwen35_27b_resid_w80k
  qwen35_2b:
    name: qwen35_2b
    pretrained_name: Qwen/Qwen3.5-2B-Base
    family: qwen
    sae_profiles:
    - qwen35_2b_resid_w32k
    default_saes:
    - qwen35_2b_resid_w32k
  qwen35_4b:
    name: qwen35_4b
    pretrained_name: Qwen/Qwen3.5-4B
    family: qwen
    sae_profiles:
    - qwen35_4b_matryoshka_l15_65k
    default_saes:
    - qwen35_4b_matryoshka_l15_65k
  qwen35_9b:
    name: qwen35_9b
    pretrained_name: Qwen/Qwen3.5-9B-Base
    family: qwen
    sae_profiles:
    - qwen35_9b_resid_w64k
    default_saes:
    - qwen35_9b_resid_w64k
    default_layers:
      qwen35_9b_resid_w64k: 16
  qwen3_1_7b:
    name: qwen3_1_7b
    pretrained_name: Qwen/Qwen3-1.7B
    family: qwen
    sae_profiles:
    - qwen3_1_7b_ak_resid_l14_65k
    default_saes:
    - qwen3_1_7b_ak_resid_l14_65k
  qwen3_1_7b_base:
    name: qwen3_1_7b_base
    pretrained_name: Qwen/Qwen3-1.7B-Base
    family: qwen
    sae_profiles:
    - qwen3_1_7b_base_resid_w32k
    - qwen3_1_7b_base_resid_w32k_l0_100
    default_saes:
    - qwen3_1_7b_base_resid_w32k
    default_layers:
      qwen3_1_7b_base_resid_w32k_l0_100: 14
      qwen3_1_7b_base_resid_w32k: 14
  qwen3_8b:
    name: qwen3_8b
    pretrained_name: Qwen/Qwen3-8B-Base
    family: qwen
    sae_profiles:
    - qwen3_8b_resid_w64k
    default_saes:
    - qwen3_8b_resid_w64k
    default_layers:
      qwen3_8b_resid_w64k: 18
  r1_distill_llama8b:
    name: r1_distill_llama8b
    pretrained_name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    family: llama
    sae_profiles:
    - r1_distill_llama8b_llamascope_l15_openr1
    - r1_distill_llama8b_llamascope_l23
    default_saes:
    - r1_distill_llama8b_llamascope_l23
    default_layers:
      r1_distill_llama8b_llamascope_l23: 23
  r1_distill_qwen15b:
    name: r1_distill_qwen15b
    pretrained_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    family: qwen
    sae_profiles:
    - r1_distill_qwen15b_resid_l20
    default_saes:
    - r1_distill_qwen15b_resid_l20</pre>

</details>

<details>
<summary>conf/tasks/benchmark/task.yaml</summary>

[Open Full File](../conf/tasks/benchmark/task.yaml)

<!-- SHA256: 1b5adb863c531a64d37599d9c02381d723724532db75ebc5d1343c310f0dc3cc -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: benchmark
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  resume: true
  benchmark:
    evaluation: null
    benchmark_config: &#x27;&#x27;
    protocol: null  # Metric/method parameters; model and dataset remain global selections.

evaluation:
  name: activation_health
  n_prompts: 5
  device: cuda
  require_prepared_prompts: true

presets:
  model_comparison:
    schema: saebase.model_comparison
    status_schema: saebase.model_comparison.results
    name: SAEBase  cross-family real release acceptance
    targets:
    - model_key: pythia70m
      family: pythia
      site:
        layer: 3
        hook_name: blocks.3.hook_resid_post
      sae_keys:
      - pythia70m_resid_sm
      - pythia70m_saebench_gated_l3
      model_card: https://huggingface.co/EleutherAI/pythia-70m-deduped
      sae_cards:
        pythia70m_resid_sm: https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed
        pythia70m_saebench_gated_l3: https://github.com/adamkarvonen/SAEBench
    - model_key: gpt2small
      family: gpt2
      site:
        layer: 8
        hook_name: blocks.8.hook_resid_pre
      sae_keys:
      - gpt2small_res_jb_l8
      - gpt2small_fs_6144_l8
      model_card: https://huggingface.co/openai-community/gpt2
      sae_cards:
        gpt2small_res_jb_l8: https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted
        gpt2small_fs_6144_l8: https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
    prompts:
    - The city is full of beautiful buildings.
    - The weather was warm and sunny yesterday.
    - The scientist wrote the results on paper.
    - A small animal ran across the quiet garden.
    - The teacher explained the lesson to students.
    - The river flows through the mountains.
    metrics:
    - core.l0
    - core.reconstruction_mse
    - core.fvu
    - core.explained_variance
    - intervention.logit_delta_l2
    - intervention.kl_baseline_to_edited
    conditions:
    - name: identity
      operation: identity
    - name: reconstruction
      operation: reconstruction
    seed: 0
    max_measurements: 128
    max_runtime_seconds: 2400
    max_token_count: 80
    bootstrap_replicates: 64
  model_comparison_gpt2:
    schema: saebase.model_comparison
    status_schema: saebase.model_comparison.results
    name: Model comparison optional extended GPT2 widths and Pythia checkpoints
    targets:
    - model_key: pythia70m
      family: pythia
      site:
        layer: 3
        hook_name: blocks.3.hook_resid_post
      sae_keys:
      - pythia70m_resid_sm
      - pythia70m_saebench_gated_l3
      model_card: https://huggingface.co/EleutherAI/pythia-70m-deduped
      sae_cards:
        pythia70m_resid_sm: https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed
        pythia70m_saebench_gated_l3: https://github.com/adamkarvonen/SAEBench
    - model_key: gpt2small
      family: gpt2
      site:
        layer: 8
        hook_name: blocks.8.hook_resid_pre
      sae_keys:
      - gpt2small_res_jb_l8
      - gpt2small_fs_6144_l8
      - gpt2small_fs_12288_l8
      - gpt2small_fs_24576_l8
      model_card: https://huggingface.co/openai-community/gpt2
      sae_cards:
        gpt2small_res_jb_l8: https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted
        gpt2small_fs_6144_l8: https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
        gpt2small_fs_12288_l8: https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
        gpt2small_fs_24576_l8: https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
    prompts:
    - The city is full of beautiful buildings.
    - The weather was warm and sunny yesterday.
    - The scientist wrote the results on paper.
    - A small animal ran across the quiet garden.
    - The teacher explained the lesson to students.
    - The river flows through the mountains.
    metrics:
    - core.l0
    - core.reconstruction_mse
    - core.fvu
    - intervention.logit_delta_l2
    - intervention.kl_baseline_to_edited
    conditions:
    - name: identity
      operation: identity
    - name: reconstruction
      operation: reconstruction
    seed: 0
    max_measurements: 128
    max_runtime_seconds: 3600
    max_token_count: 80
    bootstrap_replicates: 64
  pythia_benchmark_smoke:
    schema: saebase.benchmark.config.v1
    name: Pythia M3 bounded smoke
    targets:
    - model_key: pythia70m
      sae_keys:
      - pythia70m_resid_sm
      layer: 3
      hook_name: blocks.3.hook_resid_post
    dataset:
      id: m0-example-tokens
      samples:
      - sample_id: m0-smoke-1
        token_ids:
        - 40587
        - 37083
        - 15458
        prompt_group_id: m0-1
        split: test
        target_token_id: 1
        reference_token_id: 0
    metrics:
    - core.l0
    - core.reconstruction_mse
    - core.fvu
    - intervention.logit_delta_l2
    - intervention.kl_baseline_to_edited
    conditions:
    - name: identity
      operation: identity
    - name: reconstruction
      operation: reconstruction
    seed: 0
    max_measurements: 16
    max_activation_rows: 8
    tolerance: 0.01
  pythia_grouped_benchmark:
    schema: saebase.benchmark.config.v1
    name: M4 Pythia-70M real grouped audit smoke
    targets:
    - model_key: pythia70m
      sae_keys:
      - pythia70m_resid_sm
      layer: 3
      hook_name: blocks.3.hook_resid_post
    dataset:
      id: m0-three-frozen-reference-prompts
      samples:
      - sample_id: simple_copying-000000
        token_ids:
        - 40587
        - 37083
        - 15458
        prompt_group_id: m0-group-000
        split: test
        target_token_id: 1
        reference_token_id: 0
      - sample_id: simple_copying-000002
        token_ids:
        - 6300
        - 249
        - 29040
        - 20839
        prompt_group_id: m0-group-002
        split: test
        target_token_id: 1
        reference_token_id: 0
      - sample_id: simple_copying-000003
        token_ids:
        - 53
        - 50177
        - 38150
        - 308
        - 50177
        prompt_group_id: m0-group-003
        split: test
        target_token_id: 1
        reference_token_id: 0
    metrics:
    - core.l0
    - core.reconstruction_mse
    - core.fvu
    - core.explained_variance
    - intervention.logit_delta_l2
    - intervention.kl_baseline_to_edited
    conditions:
    - name: identity
      operation: identity
    - name: reconstruction
      operation: reconstruction
    seed: 0
    max_measurements: 16
    max_activation_rows: 16
    tolerance: 0.01
    sample_batch: 1
  m4_ground_truth:
    schema: saebase.m4.audit.config.v1
    evaluation_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    bootstrap_replicates: 128
    confidence: 0.95
    sample_sizes:
    - 32
    - 64
    - 128
    top_k: 1
    training_seed: null
    metric_ids: []
  pythia_reliability_audit:
    schema: saebase.m4.audit.config.v1
    evaluation_seeds:
    - 0
    - 1
    - 2
    - 3
    - 4
    bootstrap_replicates: 96
    confidence: 0.95
    sample_sizes:
    - 2
    - 3
    - 32
    - 128
    - 512
    top_k: 1
    training_seed: null
    metric_ids:
    - core.l0
    - core.reconstruction_mse
    - core.fvu
    - core.explained_variance
    - intervention.logit_delta_l2
    - intervention.kl_baseline_to_edited

site_catalog:
  schema: saebase.coverage.stage_bc.v1
  objective: Representative open model/SAE coverage with exact layer topology, source
    files and independent real-weight status
  model_pair_work: CPU only for small ungated models; all large/Gemma gated model
    pairs deferred to D
  targets:
  - model_key: gemma270m
    sae_key: gemma270m_scope_resid
    size: 270M
    n_layers: 18
    architecture: jumprelu
    selected_layers:
    - 5
    - 9
    - 12
    cpu_probe_layers:
    - 5
    - 9
    access: gated_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: gemma1b
    sae_key: gemma1b_scope_resid
    size: 1B
    n_layers: 26
    architecture: jumprelu
    selected_layers:
    - 7
    - 13
    - 17
    cpu_probe_layers:
    - 7
    - 13
    access: gated_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: gemma2_2b_base
    sae_key: gemma2_2b_pt_l25_canonical
    size: 2B
    n_layers: 26
    architecture: jumprelu
    selected_layers:
    - 13
    - 25
    cpu_probe_layers:
    - 25
    access: gated_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: qwen35_2b
    sae_key: qwen35_2b_resid_w32k
    size: 2B
    n_layers: 24
    architecture: topk
    selected_layers:
    - 6
    - 12
    - 15
    - 20
    cpu_probe_layers:
    - 12
    access: open
    tier: core
    model_pair_cpu_candidate: false
  - model_key: qwen3_8b
    sae_key: qwen3_8b_resid_w64k
    size: 8B
    n_layers: 36
    architecture: topk
    selected_layers:
    - 9
    - 18
    - 23
    - 31
    cpu_probe_layers:
    - 18
    access: large_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: llama31_8b
    sae_key: llama31_8b_lxr_8x
    size: 8B
    n_layers: 32
    architecture: jumprelu
    selected_layers:
    - 8
    - 16
    - 21
    - 27
    cpu_probe_layers:
    - 16
    access: gated_large_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: pythia410m
    sae_key: pythia410m_mlp_topk
    size: 410M
    n_layers: 24
    architecture: topk
    selected_layers:
    - 4
    - 12
    - 20
    cpu_probe_layers:
    - 4
    - 12
    access: open
    tier: core
    model_pair_cpu_candidate: true
  - model_key: qwen3_1_7b_base
    sae_key: qwen3_1_7b_base_resid_w32k
    size: 1.7B
    n_layers: 28
    architecture: topk
    selected_layers:
    - 7
    - 14
    - 18
    - 23
    cpu_probe_layers:
    - 7
    - 14
    access: open
    tier: core
    model_pair_cpu_candidate: true
  - model_key: r1_distill_qwen15b
    sae_key: r1_distill_qwen15b_resid_l20
    size: 1.5B
    n_layers: 28
    architecture: topk
    selected_layers:
    - 20
    cpu_probe_layers:
    - 20
    access: open
    tier: core
    model_pair_cpu_candidate: true
  - model_key: gemma3_4b_pt
    sae_key: gemma3_4b_pt_scope_resid
    size: 4B
    n_layers: 34
    architecture: jumprelu
    selected_layers:
    - 9
    - 17
    - 22
    - 29
    cpu_probe_layers:
    - 17
    access: gated_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: gemma3_4b_it
    sae_key: gemma3_4b_it_scope_resid
    size: 4B
    n_layers: 34
    architecture: jumprelu
    selected_layers:
    - 9
    - 17
    - 22
    - 29
    cpu_probe_layers:
    - 17
    access: gated_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: mistral7b_instruct
    sae_key: mistral7b_instruct_tc_l16_131k
    size: 7B
    n_layers: 32
    architecture: topk
    selected_layers:
    - 16
    cpu_probe_layers: []
    access: large_sae_4gb
    tier: core
    model_pair_cpu_candidate: false
  - model_key: gemma2_9b_pt
    sae_key: gemma2_9b_pt_scope_resid
    size: 9B
    n_layers: 42
    architecture: jumprelu
    selected_layers:
    - 9
    - 20
    - 31
    cpu_probe_layers:
    - 20
    access: large_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: qwen35_9b
    sae_key: qwen35_9b_resid_w64k
    size: 9B
    n_layers: 32
    architecture: topk
    selected_layers:
    - 8
    - 16
    - 21
    - 27
    cpu_probe_layers:
    - 16
    access: large_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: r1_distill_llama8b
    sae_key: r1_distill_llama8b_llamascope_l23
    size: 8B
    n_layers: 32
    architecture: jumprelu
    selected_layers:
    - 23
    cpu_probe_layers:
    - 23
    access: large_model
    tier: core
    model_pair_cpu_candidate: false
  - model_key: qwen3_1_7b
    sae_key: qwen3_1_7b_ak_resid_l14_65k
    size: 1.7B
    n_layers: 28
    architecture: batchtopk
    selected_layers:
    - 14
    cpu_probe_layers:
    - 14
    access: open
    tier: architecture
    model_pair_cpu_candidate: false
  - model_key: gemma2_2b_it
    sae_key: gemma2_2b_it_matryoshka_l12_32k
    size: 2B
    n_layers: 26
    architecture: matryoshka_batchtopk
    selected_layers:
    - 12
    cpu_probe_layers:
    - 12
    access: gated_model
    tier: architecture
    model_pair_cpu_candidate: false
  - model_key: qwen35_4b
    sae_key: qwen35_4b_matryoshka_l15_65k
    size: 4B
    n_layers: 32
    architecture: matryoshka_batchtopk
    selected_layers:
    - 15
    cpu_probe_layers:
    - 15
    access: large_model
    tier: architecture
    model_pair_cpu_candidate: false
site_targets:
  join_keys:
  - model_key
  - sae_key
  - layer
  - hook_name
  - checkpoint_sha
  model_order:
  - qwen35_2b
  - pythia410m
  - qwen3_1_7b_base
  - qwen3_1_7b
  - qwen35_4b
  - qwen3_8b
  - mistral7b_instruct
  - r1_distill_llama8b
  - qwen35_9b
  - gemma270m
  - gemma1b
  - gemma2_2b_base
  - gemma2_2b_it
  - gemma3_4b_pt
  - gemma3_4b_it
  - llama31_8b
  - gemma2_9b_pt
  - r1_distill_qwen15b
  n_sites: 48
  prior_pair_verified:
  - layer: 4
    model_key: pythia410m
    sae_key: pythia410m_mlp_topk
  - layer: 12
    model_key: qwen35_2b
    sae_key: qwen35_2b_resid_w32k
  - layer: 7
    model_key: qwen3_1_7b_base
    sae_key: qwen3_1_7b_base_resid_w32k
  - layer: 20
    model_key: r1_distill_qwen15b
    sae_key: r1_distill_qwen15b_resid_l20
  schema: saebase.coverage.stage_d_targets.v1
  sites:
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.5.hook_resid_post
    last_attempt_utc: null
    layer: 5
    model_key: gemma270m
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-270m-pt
    sae_key: gemma270m_scope_resid
    sae_state_sha256: null
    size: 270M
    skip_by_default: false
    source_file_id: resid_post/layer_5_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.9.hook_resid_post
    last_attempt_utc: null
    layer: 9
    model_key: gemma270m
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-270m-pt
    sae_key: gemma270m_scope_resid
    sae_state_sha256: null
    size: 270M
    skip_by_default: false
    source_file_id: resid_post/layer_9_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.12.hook_resid_post
    last_attempt_utc: null
    layer: 12
    model_key: gemma270m
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-270m-pt
    sae_key: gemma270m_scope_resid
    sae_state_sha256: null
    size: 270M
    skip_by_default: false
    source_file_id: resid_post/layer_12_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.7.hook_resid_post
    last_attempt_utc: null
    layer: 7
    model_key: gemma1b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-1b-pt
    sae_key: gemma1b_scope_resid
    sae_state_sha256: null
    size: 1B
    skip_by_default: false
    source_file_id: resid_post/layer_7_width_16k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.13.hook_resid_post
    last_attempt_utc: null
    layer: 13
    model_key: gemma1b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-1b-pt
    sae_key: gemma1b_scope_resid
    sae_state_sha256: null
    size: 1B
    skip_by_default: false
    source_file_id: resid_post/layer_13_width_16k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.17.hook_resid_post
    last_attempt_utc: null
    layer: 17
    model_key: gemma1b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-1b-pt
    sae_key: gemma1b_scope_resid
    sae_state_sha256: null
    size: 1B
    skip_by_default: false
    source_file_id: resid_post/layer_17_width_16k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.13.hook_resid_post
    last_attempt_utc: null
    layer: 13
    model_key: gemma2_2b_base
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2b-pt-res
    sae_key: gemma2_2b_pt_l25_canonical
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: layer_13/width_16k/canonical
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.25.hook_resid_post
    last_attempt_utc: null
    layer: 25
    model_key: gemma2_2b_base
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2b-pt-res
    sae_key: gemma2_2b_pt_l25_canonical
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: layer_25/width_16k/canonical
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: d1_calibration
    hook_name: blocks.6.hook_resid_post
    last_attempt_utc: null
    layer: 6
    model_key: qwen35_2b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50
    sae_key: qwen35_2b_resid_w32k
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: layer6.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: d1_calibration
    hook_name: blocks.12.hook_resid_post
    last_attempt_utc: null
    layer: 12
    model_key: qwen35_2b
    model_state_sha256: null
    prior_pair_verified: true
    repo_id: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50
    sae_key: qwen35_2b_resid_w32k
    sae_state_sha256: null
    size: 2B
    skip_by_default: true
    source_file_id: layer12.sae.pt
    status: skipped_prior
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: d1_calibration
    hook_name: blocks.15.hook_resid_post
    last_attempt_utc: null
    layer: 15
    model_key: qwen35_2b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50
    sae_key: qwen35_2b_resid_w32k
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: layer15.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: d1_calibration
    hook_name: blocks.20.hook_resid_post
    last_attempt_utc: null
    layer: 20
    model_key: qwen35_2b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50
    sae_key: qwen35_2b_resid_w32k
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: layer20.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.9.hook_resid_post
    last_attempt_utc: null
    layer: 9
    model_key: qwen3_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
    sae_key: qwen3_8b_resid_w64k
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: layer9.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.18.hook_resid_post
    last_attempt_utc: null
    layer: 18
    model_key: qwen3_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
    sae_key: qwen3_8b_resid_w64k
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: layer18.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.23.hook_resid_post
    last_attempt_utc: null
    layer: 23
    model_key: qwen3_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
    sae_key: qwen3_8b_resid_w64k
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: layer23.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.31.hook_resid_post
    last_attempt_utc: null
    layer: 31
    model_key: qwen3_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
    sae_key: qwen3_8b_resid_w64k
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: layer31.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.8.hook_resid_post
    last_attempt_utc: null
    layer: 8
    model_key: llama31_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
    sae_key: llama31_8b_lxr_8x
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: Llama3_1-8B-Base-L8R-8x
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.16.hook_resid_post
    last_attempt_utc: null
    layer: 16
    model_key: llama31_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
    sae_key: llama31_8b_lxr_8x
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: Llama3_1-8B-Base-L16R-8x
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.21.hook_resid_post
    last_attempt_utc: null
    layer: 21
    model_key: llama31_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
    sae_key: llama31_8b_lxr_8x
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: Llama3_1-8B-Base-L21R-8x
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.27.hook_resid_post
    last_attempt_utc: null
    layer: 27
    model_key: llama31_8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
    sae_key: llama31_8b_lxr_8x
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: Llama3_1-8B-Base-L27R-8x
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.4.hook_mlp_out
    last_attempt_utc: null
    layer: 4
    model_key: pythia410m
    model_state_sha256: null
    prior_pair_verified: true
    repo_id: EleutherAI/sae-pythia-410m-65k
    sae_key: pythia410m_mlp_topk
    sae_state_sha256: null
    size: 410M
    skip_by_default: true
    source_file_id: layers.4.mlp
    status: skipped_prior
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.12.hook_mlp_out
    last_attempt_utc: null
    layer: 12
    model_key: pythia410m
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: EleutherAI/sae-pythia-410m-65k
    sae_key: pythia410m_mlp_topk
    sae_state_sha256: null
    size: 410M
    skip_by_default: false
    source_file_id: layers.12.mlp
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.20.hook_mlp_out
    last_attempt_utc: null
    layer: 20
    model_key: pythia410m
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: EleutherAI/sae-pythia-410m-65k
    sae_key: pythia410m_mlp_topk
    sae_state_sha256: null
    size: 410M
    skip_by_default: false
    source_file_id: layers.20.mlp
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.7.hook_resid_post
    last_attempt_utc: null
    layer: 7
    model_key: qwen3_1_7b_base
    model_state_sha256: null
    prior_pair_verified: true
    repo_id: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
    sae_key: qwen3_1_7b_base_resid_w32k
    sae_state_sha256: null
    size: 1.7B
    skip_by_default: true
    source_file_id: layer7.sae.pt
    status: skipped_prior
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.14.hook_resid_post
    last_attempt_utc: null
    layer: 14
    model_key: qwen3_1_7b_base
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
    sae_key: qwen3_1_7b_base_resid_w32k
    sae_state_sha256: null
    size: 1.7B
    skip_by_default: false
    source_file_id: layer14.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.18.hook_resid_post
    last_attempt_utc: null
    layer: 18
    model_key: qwen3_1_7b_base
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
    sae_key: qwen3_1_7b_base_resid_w32k
    sae_state_sha256: null
    size: 1.7B
    skip_by_default: false
    source_file_id: layer18.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.23.hook_resid_post
    last_attempt_utc: null
    layer: 23
    model_key: qwen3_1_7b_base
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
    sae_key: qwen3_1_7b_base_resid_w32k
    sae_state_sha256: null
    size: 1.7B
    skip_by_default: false
    source_file_id: layer23.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.20.hook_resid_post
    last_attempt_utc: null
    layer: 20
    model_key: r1_distill_qwen15b
    model_state_sha256: null
    prior_pair_verified: true
    repo_id: analist/sae-r1-distill-qwen-1.5b-resid-l20-reasoning
    sae_key: r1_distill_qwen15b_resid_l20
    sae_state_sha256: null
    size: 1.5B
    skip_by_default: true
    source_file_id: resid_post_layer20
    status: skipped_prior
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.9.hook_resid_post
    last_attempt_utc: null
    layer: 9
    model_key: gemma3_4b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-pt
    sae_key: gemma3_4b_pt_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_9_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.17.hook_resid_post
    last_attempt_utc: null
    layer: 17
    model_key: gemma3_4b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-pt
    sae_key: gemma3_4b_pt_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_17_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.22.hook_resid_post
    last_attempt_utc: null
    layer: 22
    model_key: gemma3_4b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-pt
    sae_key: gemma3_4b_pt_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_22_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.29.hook_resid_post
    last_attempt_utc: null
    layer: 29
    model_key: gemma3_4b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-pt
    sae_key: gemma3_4b_pt_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_29_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.9.hook_resid_post
    last_attempt_utc: null
    layer: 9
    model_key: gemma3_4b_it
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-it
    sae_key: gemma3_4b_it_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_9_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.17.hook_resid_post
    last_attempt_utc: null
    layer: 17
    model_key: gemma3_4b_it
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-it
    sae_key: gemma3_4b_it_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_17_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.22.hook_resid_post
    last_attempt_utc: null
    layer: 22
    model_key: gemma3_4b_it
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-it
    sae_key: gemma3_4b_it_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_22_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: gated_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.29.hook_resid_post
    last_attempt_utc: null
    layer: 29
    model_key: gemma3_4b_it
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-2-4b-it
    sae_key: gemma3_4b_it_scope_resid
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: resid_post/layer_29_width_65k_l0_medium
    status: registered
    tier: core
  - acceptance_path: null
    access: large_sae_4gb
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.16.hook_mlp_out
    last_attempt_utc: null
    layer: 16
    model_key: mistral7b_instruct
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: tylercosgrove/mistral-7b-sparse-autoencoder-layer16
    sae_key: mistral7b_instruct_tc_l16_131k
    sae_state_sha256: null
    size: 7B
    skip_by_default: false
    source_file_id: .
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.9.hook_resid_post
    last_attempt_utc: null
    layer: 9
    model_key: gemma2_9b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-9b-pt-res
    sae_key: gemma2_9b_pt_scope_resid
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer_9/width_16k/average_l0_51
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.20.hook_resid_post
    last_attempt_utc: null
    layer: 20
    model_key: gemma2_9b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-9b-pt-res
    sae_key: gemma2_9b_pt_scope_resid
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer_20/width_16k/average_l0_58
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.31.hook_resid_post
    last_attempt_utc: null
    layer: 31
    model_key: gemma2_9b_pt
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: google/gemma-scope-9b-pt-res
    sae_key: gemma2_9b_pt_scope_resid
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer_31/width_16k/average_l0_35
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.8.hook_resid_post
    last_attempt_utc: null
    layer: 8
    model_key: qwen35_9b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
    sae_key: qwen35_9b_resid_w64k
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer8.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.16.hook_resid_post
    last_attempt_utc: null
    layer: 16
    model_key: qwen35_9b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
    sae_key: qwen35_9b_resid_w64k
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer16.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.21.hook_resid_post
    last_attempt_utc: null
    layer: 21
    model_key: qwen35_9b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
    sae_key: qwen35_9b_resid_w64k
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer21.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: topk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.27.hook_resid_post
    last_attempt_utc: null
    layer: 27
    model_key: qwen35_9b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
    sae_key: qwen35_9b_resid_w64k
    sae_state_sha256: null
    size: 9B
    skip_by_default: false
    source_file_id: layer27.sae.pt
    status: registered
    tier: core
  - acceptance_path: null
    access: large_model
    architecture: jumprelu
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: large
    hook_name: blocks.23.hook_resid_post
    last_attempt_utc: null
    layer: 23
    model_key: r1_distill_llama8b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: fnlp/Llama-Scope-R1-Distill
    sae_key: r1_distill_llama8b_llamascope_l23
    sae_state_sha256: null
    size: 8B
    skip_by_default: false
    source_file_id: 400M-Slimpajama-400M-OpenR1-Math-220k/L23R
    status: registered
    tier: core
  - acceptance_path: null
    access: open
    architecture: batchtopk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.14.hook_resid_post
    last_attempt_utc: null
    layer: 14
    model_key: qwen3_1_7b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: adamkarvonen/qwen3-1.7b-saes
    sae_key: qwen3_1_7b_ak_resid_l14_65k
    sae_state_sha256: null
    size: 1.7B
    skip_by_default: false
    source_file_id: saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_14/trainer_2
    status: registered
    tier: architecture
  - acceptance_path: null
    access: gated_model
    architecture: matryoshka_batchtopk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: gated
    hook_name: blocks.12.hook_resid_post
    last_attempt_utc: null
    layer: 12
    model_key: gemma2_2b_it
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40
    sae_key: gemma2_2b_it_matryoshka_l12_32k
    sae_state_sha256: null
    size: 2B
    skip_by_default: false
    source_file_id: blocks.12.hook_resid_post
    status: registered
    tier: architecture
  - acceptance_path: null
    access: large_model
    architecture: matryoshka_batchtopk
    checkpoint_sha: null
    error: null
    error_type: null
    execution_wave: open_small
    hook_name: blocks.15.hook_resid_post
    last_attempt_utc: null
    layer: 15
    model_key: qwen35_4b
    model_state_sha256: null
    prior_pair_verified: false
    repo_id: decoderesearch/qwen-3.5-saes
    sae_key: qwen35_4b_matryoshka_l15_65k
    sae_state_sha256: null
    size: 4B
    skip_by_default: false
    source_file_id: qwen-3.5-4b/btk-mat-layer-15-k-100
    status: registered
    tier: architecture
  source_catalog_schema: saebase.coverage.stage_bc.v1</pre>

</details>

<details>
<summary>conf/tasks/capture/task.yaml</summary>

[Open Full File](../conf/tasks/capture/task.yaml)

<!-- SHA256: 35ced6365b53760f01f18a9f4d50323263d77e34848ac734d6862834766c8f1d -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: capture
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  capture:
    sample_ids: null  # Optional observation subset; preserves the full reference cohort.
    output_scope: last_token  # Or all_tokens: complete native sparse codes at each token.
    max_tokens: 128
    max_active_per_token: 256
    mode: whiten  # non_whiten | whiten | g_s | sar (regression) | I (exhaustive) | rss
    whiten_chunk_size: 256  # Native SAE tokenwise encoding batch; statistics use FP64.
    neuronpedia:
      enabled: true
      allow_network: true
      cache_dir: null  # Uses assets/annotation_cache/neuronpedia_explanations/files under the checkout.
      timeout_seconds: 15
      total_timeout_seconds: 120
      max_download_bytes: 67108864
      max_uncompressed_bytes: 268435456

    model_answer:
      max_new_tokens: 256  # Unedited continuation of the exact captured input; always recorded.
      batch_size: 8  # Group equal-length inputs; reuse the loaded model and upstream KV cache.
    output:
      feature_count: null  # Output features/pairs; null follows the candidate count.
    selection:
      candidate_policy: active_whiten  # Shared per-question selection, or legacy mode-specific selection.
      candidate_count: 20  # &#x27;all&#x27; uses all features; a positive integer limits candidates.
      feature_filter:
        enabled: true  # Reused by all numeric-filter Capture modes.
        min_native_nonzero_fraction: 0.01  # Neuronpedia corpus frequency, not amplitude.
        density_strict: true  # Exclude only densities strictly above the threshold.
      cohort_filter:
        enabled: true  # Exclude frequent last-token z &gt; threshold when numeric filter is used.
        reference_sample_count: 100  # Reference questions for mean/std and frequency; SAR uses dataset.sample_count.
        zscore_threshold: 1.0
        min_question_fraction: 0.5  # Inclusive: at least this fraction of all selected questions.
      feature_selection: whiten  # Candidate ranking for legacy g_s/SAR/I.
    answer:
      backend: full_context_tail  # Complete-context tail verified against native full scoring.
      reference_text: null  # Explicit canonical text for task.prompt; Dataset references otherwise.
    g_s:
      candidate_count: 100  # Legacy Whiten Top N before scoring single deletions by g(S).
    i:
      feature_selection: whiten  # Legacy I candidate ranking; null inherits selection.
      g_s_shortfall: available  # Legacy g_s candidates: available | relax_filters.
      max_exhaustive_features: 16  # Validate the finite exhaustive candidate pool before measurement.
    sar:
      source: null  # Explicit trusted local original sar.py (or its checkout directory).
      top_pairs: 0  # Positive values also report pairwise fit coefficients.
      pair_ranking: signed  # mode=sar only; mode=I always uses largest signed measured I.
      M: 384
      n_fit: 256
      p: 0.5
      readout: answer_mean_logprob  # Surrogate also supports logit_difference.
      readout_position: -1
      target_token_id: null  # For an explicit task.prompt; dataset labels otherwise.
      reference_token_id: null
      require_affine_decode: true
    rss:
      source: null  # Trusted local upstream rss.py; for this case study: tasks/capture/scoring/stable_support.py.
      contexts: []  # Each full variant is context + space + task.prompt.
      M: 400
      K: 20
      rho: 0.8
      kappa: 0.8
      tau: 0.00001
      report:
        dominance_ratio: 5.0  # Required ratio for display candidates.
  performance:
    enabled: true
    batch_size: auto
    batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    warmup: 1
    repeats: 3
    probe_samples: 3
    probe_states: 64
    max_seconds: 120
    memory_fraction: 0.9

# Explicit research batch selections; availability is owned by the component Registry.
matrix:
  models:
  - model: gpt2small
    sae_key: gpt2small_res_jb_l8
    layer: 8
  - model: gemma2_2b_base
    sae_key: gemma2_2b_pt_l25_canonical
    layer: 25
  - model: gemma1b
    sae_key: gemma1b_scope_resid
    layer: 13
  - model: gemma3_4b_pt
    sae_key: gemma3_4b_pt_scope_resid
    layer: 17
  - model: gemma3_12b_it
    offload_unused_layers: true
    sae_key: gemma3_12b_it_scope_resid
    layer: 24
  - model: llama31_8b
    sae_key: llama31_8b_lxr_8x
    layer: 16
  - model: qwen3_1_7b_base
    sae_key: qwen3_1_7b_base_resid_w32k_l0_100
    layer: 14
  - model: qwen3_8b
    sae_key: qwen3_8b_resid_w64k
    layer: 18
  - model: qwen35_9b
    sae_key: qwen35_9b_resid_w64k
    layer: 16
  - model: r1_distill_llama8b
    sae_key: r1_distill_llama8b_llamascope_l23
    layer: 23
  datasets:
  - hotpotqa
  - bbh
  - ioi
  - ravel
  - counterfact
  - arc_challenge
  - mmlu_pro
  - gsm8k
  - humaneval
  - truthfulqa
  - harmbench
  - xstest
  seed: 42
  model_timeout_seconds: 86400

sar_filter_policy:
  version: 2
  punctuation_rules:
    punctuation: \bpunctuation\b
    colon_semicolon: \b(?:colons?|semicolons?)\b
    comma: \bcommas?\b
    apostrophe: \bapostrophes?\b
    quotation_marks: \b(?:quotation marks?|quote marks?|double quotes?|single quotes?|inverted
      commas?)\b
    parentheses_brackets: \b(?:parentheses|parenthesis|parenthetical|brackets?)\b
    hyphen_dash: \b(?:hyphens?|em[ -]?dashes?|en[ -]?dashes?)\b
    ellipsis: \b(?:ellipsis|ellipses)\b
    question_exclamation_marks: \b(?:question marks?|exclamation (?:marks?|points?))\b
    full_stop_period: \bfull stops?\b|\b(?:sentence[- ]ending|punctuation) periods?\b|\bperiods?
      (?:as punctuation|at the end of sentences)\b
    sentence_boundary: \bsentence (?:boundaries|endings?|termination)\b|\bend[- ]of[-
      ]sentence (?:markers?|punctuation)\b
    syntax_braces_slash: \b(?:braces?|slashes?|dashes?)\b.{0,60}\b(?:punctuation|syntax|delimiters?|text|sentences?)\b|\b(?:punctuation|syntax|delimiters?)\b.{0,60}\b(?:braces?|slashes?|dashes?)\b
  structural_rules:
    function_word_class: \b(?:prepositions?|pronouns?|conjunctions?|determiners?|function
      words?|auxiliary verbs?|modal verbs?)\b
    grammatical_article: \b(?:definite|indefinite) articles?\b|\barticles? (?:and|or)
      (?:prepositions?|pronouns?|determiners?)\b
    natural_language_grammar: \b(?:grammar|grammatical|syntactic|syntax|parts? of
      speech)\b
    sentence_structure: \bsentence (?:structures?|construction|patterns?)\b|\b(?:relative|subordinate|subordinating|main|adverbial|conditional)
      clauses?\b
    verb_morphology: \b(?:verb tenses?|past tense|present tense|future tense|perfect
      tense|conjugations?|inflections?|subject[- ]verb agreement)\b|\bforms? of (?:the
      )?verb\s+[\&quot;\u201c\u2018\x27]?(?:to\s+)?(?:be|have|do)\b
    generic_question_form: \b(?:interrogative|question forms?|question structures?|question
      phrases?|yes[- ]no questions?|rhetorical questions?)\b|\bquestions? (?:in|and)
      dialogue\b
    reported_speech_structure: \b(?:reported speech|direct speech|indirect speech|quotative
      constructions?)\b
    layout_only: \b(?:whitespace|white space|line breaks?|newlines?|indentation)\b
  named_function_words:
  - a
  - about
  - above
  - across
  - after
  - against
  - along
  - although
  - am
  - among
  - an
  - and
  - are
  - as
  - at
  - be
  - because
  - been
  - before
  - being
  - between
  - but
  - by
  - can
  - could
  - despite
  - did
  - do
  - does
  - during
  - for
  - from
  - had
  - has
  - have
  - he
  - her
  - him
  - his
  - i
  - if
  - in
  - into
  - is
  - it
  - its
  - may
  - me
  - might
  - must
  - my
  - near
  - nor
  - not
  - of
  - &#x27;on&#x27;
  - onto
  - or
  - our
  - over
  - shall
  - she
  - should
  - so
  - that
  - the
  - their
  - them
  - these
  - they
  - this
  - those
  - though
  - through
  - throughout
  - to
  - under
  - unless
  - us
  - was
  - we
  - were
  - what
  - whereas
  - which
  - while
  - who
  - whom
  - whose
  - will
  - with
  - within
  - without
  - would
  - yet
  - you
  - your
  min_native_nonzero_fraction: 0.5
  matching_semantics: Filter explicit punctuation, function-word classes, named function
    words, natural-language grammar, generic question forms, reported-speech structure,
    and whitespace layout. Programming syntax alone and topical/proper nouns are not
    automatically classified as linguistic structure.
  origin: SAR/to_ziye/filtering/policy.json; v2 rules, applied to the selected SAE
    source</pre>

</details>

<details>
<summary>conf/tasks/circuits/task.yaml</summary>

[Open Full File](../conf/tasks/circuits/task.yaml)

<!-- SHA256: 6b861b899ec810919328e9febd7966854b5ba32d5ac3559a43b8ad91aa263da9 -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: circuits
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  timeout_seconds: 1800
  circuits:
    approved: false
    transcoder_set: gemma
    slug: gemma2b-trace</pre>

</details>

<details>
<summary>conf/tasks/explain/task.yaml</summary>

[Open Full File](../conf/tasks/explain/task.yaml)

<!-- SHA256: 9480f81ccb09a3c4279ca5ce6bd821e5cc78b2f2c2c01cc79450e99dfa5d6f93 -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: explain
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  explain:
    approved: false
    discovery_file: &#x27;&#x27;
    heldout_file: &#x27;&#x27;
    site_id: &#x27;&#x27;
    feature_id: 0
    generator_model: &#x27;&#x27;
    generator_method: &#x27;&#x27;
    authorized_budget_usd: 0.0
    estimated_cost_usd: 0.0
    # Empty keeps the Python callable entry. The only allowlisted name is
    # saebase_generate; other strings are rejected and never executed.
    generator: &#x27;&#x27;
    generator_max_new_tokens: 24</pre>

</details>

<details>
<summary>conf/tasks/generate/task.yaml</summary>

[Open Full File](../conf/tasks/generate/task.yaml)

<!-- SHA256: 1bb46c8b1f5281aab263b9fd66c7d0457fcf9da3aabe9f572d417f80686bd0ea -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: generate
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  resume: true
  performance:
    enabled: true
    batch_size: auto
    batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    warmup: 1
    repeats: 3
    probe_samples: 3
    probe_states: 64
    max_seconds: 120
    memory_fraction: 0.9
  generate:
    feature_id: 0
    intervention: add
    value: 1.0
    max_new_tokens: 8
    temperature: 0.0
    stop_at_context_limit: true  # Keep the original prompt; stop with a recorded reason at the native limit.

    feature_selection: configured  # configured | last_active_content_token | capture_top
    top_feature_num: 10
    feature_ranking: whiten  # non_whiten | whiten: select once from the original prompt.
    capture_source: null  # Optional saved capture run with exact prompt/checkpoint/site/ranking matches.
    whiten_chunk_size: 256
    inline_results: true  # False keeps full rows in reports/generate_features.jsonl.
    continue_on_error: false
    conditions: null  # Optional list of native intervention conditions.
    reuse_prompt_observations: true  # Exact full-prefix reuse within each prompt; no KV cache.</pre>

</details>

<details>
<summary>conf/tasks/intervene/task.yaml</summary>

[Open Full File](../conf/tasks/intervene/task.yaml)

<!-- SHA256: 0fdb2ca11b5d19ab7d740a3fe65990174b9e216de70e78234cf274fec5860848 -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: intervene
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  resume: true
  performance:
    enabled: true
    batch_size: auto
    batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    warmup: 1
    repeats: 3
    probe_samples: 3
    probe_states: 64
    max_seconds: 120
    memory_fraction: 0.9
  intervene:
    feature_id: 0
    value: 0.0
    verify_identity: true
    feature_selection: configured  # configured | capture_top: independent edits of each Top N feature.
    top_feature_num: 10
    feature_ranking: whiten  # non_whiten | whiten: same ranking as capture.
    capture_source: null  # Optional saved capture run; requires matching prompts/checkpoints/site/ranking.
    whiten_chunk_size: 256
    inline_results: true  # False keeps full rows in reports/intervene_features.jsonl.
    continue_on_error: false  # Large matrices may retain failures and continue independent features.

    protocol: null  # Batched conditions/budgets; token samples belong to dataset.samples.
    reuse_prompt_observations: true  # Exact full-prefix reuse within each prompt; no KV cache.

# Purpose: Complete implementation of the main survey of the configuration Top 10 Independent Zero Experiment.
# Scope: 10 A model, 12 Data sets, intervene/generate;Llama Use current L21.
matrix:
  models:
  - model: gpt2small
    sae_key: gpt2small_res_jb_l8
    layer: 8
  - model: gemma2_2b_base
    sae_key: gemma2_2b_pt_l25_canonical
    layer: 25
  - model: gemma1b
    sae_key: gemma1b_scope_resid
    layer: 13
  - model: gemma3_4b_pt
    sae_key: gemma3_4b_pt_scope_resid
    layer: 17
  - model: gemma3_12b_it
    sae_key: gemma3_12b_it_scope_resid
    layer: 24
  - model: llama31_8b
    sae_key: llama31_8b_lxr_8x
    layer: 21
  - model: qwen3_1_7b_base
    sae_key: qwen3_1_7b_base_resid_w32k_l0_100
    layer: 14
  - model: qwen3_8b
    sae_key: qwen3_8b_resid_w64k
    layer: 18
  - model: qwen35_9b
    sae_key: qwen35_9b_resid_w64k
    layer: 16
  - model: r1_distill_llama8b
    sae_key: r1_distill_llama8b_llamascope_l23
    layer: 23
  datasets:
  - ioi
  - ravel
  - counterfact
  - humaneval
  - arc_challenge
  - truthfulqa
  - gsm8k
  - bbh
  - hotpotqa
  - mmlu_pro
  - harmbench
  - xstest
  seed: 42
  model_timeout_seconds: 86400
  tasks:
  - intervene
  - generate
  feature_ranking: whiten
  feature_count: 10
  intervention_value: 0.0
  capture_source_tag: cohort_ranking
  generation:
    intervention: ablate
    max_new_tokens: 8
    temperature: 0.0
  max_passes: 2
  tag: whiten_feature_effects

presets:
  pythia_feature_edits:
    model_key: pythia70m
    sae_key: pythia70m_resid_sm
    layer: 3
    sample_batch: 4
    tolerance: 0.001
    microbatch: 8
    max_measurements: 10000
    max_generated_tokens: 100
    samples:
    - sample_id: example-user-token-ids
      token_ids:
      - 40587
      - 37083
      - 15458
      target_token_id: 1
      reference_token_id: 0
    conditions:
    - name: identity
      operation: identity
    - name: ablate_feature_0
      operation: ablate
      feature_ids:
      - 0
    - name: scale_feature_0
      operation: scale
      feature_ids:
      - 0
      value: 0.5
    - name: replace_with_sae_reconstruction
      operation: reconstruction
    steering:
      condition_name: ablate_feature_0
      max_new_tokens: 3
      seed: 0
      temperature: 0.0</pre>

</details>

<details>
<summary>conf/tasks/precheck/task.yaml</summary>

[Open Full File](../conf/tasks/precheck/task.yaml)

<!-- SHA256: 1ff0cfc95028f7670c4cc4cd483975b728aca791c99ec63459b26902311bdab6 -->
<pre># @package _global_
# Setup only prepares assets; Precheck owns every real test.
task:
  name: precheck
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  prompt: &#x27;&#x27;  # Empty checks the selected Dataset; explicit text checks that text.
  device: auto
  performance:
    enabled: true
    batch_size: auto
    batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    warmup: 1
    repeats: 3
    probe_samples: 3
    probe_states: 64
    max_seconds: 120
    memory_fraction: 0.9
  precheck:
    resource_only: false  # True: load/probe only the SAE, without a language model.
    worker_mode: isolated  # in_process only for trusted batches with an outer process timeout.
    mode: sae             # sae | task | all | performance
    sae_list: default     # default | all | list of Model-owned SAE keys
    check_tasks: []       # subset of capture/intervene/generate/circuits/explain/benchmark/train
    auto_setup: true
    allow_network: true
    allow_install: true
    allow_download: true
    max_download_bytes: 17179869184  # 16 GiB of missing files, not entire SAE collections.
    prepare_timeout_seconds: 1800
    dataset_prepare_timeout_seconds: 1800
    check_sample_count: 32  # Real input probes; independent of Dataset cohort size.
    sampling_seed: 42
    max_seconds: 120
    max_samples: 2
    max_tokens: 32

    task_configs: {}  # Actual task overrides for performance checks.

sae_checks:
  # Purpose: named real-pair checks and bounded benchmark protocols.
  # Scope: explicitly selected precheck and benchmark executions.
  defaults:
    max_measurements: 24
    seed: 0
    tolerance: 0.01
    sample_batch: 1
    microbatch: 1
    audit:
      evaluation_seeds:
      - 0
      - 1
      - 2
      bootstrap_replicates: 32
      metric_ids:
      - core.l0
      - intervention.logit_delta_l2
  presets:
    cpu_pair:
      audit_count: 3
      prompts:
      - The capital of France is
      - The scientist explained the experiment.
      device: cpu
      max_activation_rows: 16
      sample_sizes:
      - 2
      - 3
      condition_name: active_feature_ablation
    reference_pair:
      prompts:
      - The capital of France is
      - A scientist explained the results carefully.
      - Small birds fly across the morning sky.
      model_key: qwen35_2b
      max_activation_rows: 24
      sample_sizes:
      - 2
      - 3
      condition_name: feature-ablation
  reference_pairs:
    gemma270m:
      sae_key: gemma270m_scope_resid
      layer: 5
      width: 640
    gemma1b:
      sae_key: gemma1b_scope_resid
      layer: 13
      width: 1152
    gemma2_2b:
      sae_key: gemma2_2b_pt_l25_canonical
      layer: 25
      width: 2304
    qwen35_2b:
      sae_key: qwen35_2b_resid_w32k
      layer: 12
      width: 2048
    qwen3_8b:
      sae_key: qwen3_8b_resid_w64k
      layer: 18
      width: 4096
    llama31_8b:
      sae_key: llama31_8b_lxr_8x
      layer: 16
      width: 4096</pre>

</details>

<details>
<summary>conf/tasks/train/task.yaml</summary>

[Open Full File](../conf/tasks/train/task.yaml)

<!-- SHA256: d5ac81066c0b87866faee18f39f2b7214c1276e4d2ca21bc8b9c1c2d09420cc5 -->
<pre># @package _global_
# Each task is model/dataset independent; compose selections through Hydra.
task:
  name: train
  extensions: []  # Explicit installed entry-point names; no automatic discovery.
  execute: false
  sae_key: null
  layer: null
  device: auto
  prompt: &#x27;&#x27;
  timeout_seconds: 1800
  train:
    approved: false
    train_plan: null
    allow_large: false</pre>

</details>

<!-- END CONFIG_VALUES -->

## Historical acceptance information

Read the summary to determine the version and coverage, then look at the evidence by model, SAE, and layer. Release tools and models support the document to read the original records. [Assisting](https://github.com/OpenSAE/SAEBase/blob/ed52342b706d8ad6c0e04cbe2108e69cc0dc9348/conf/coverage/stage_d_evidence/final/stage_d_summary.md)] [CPU] model search](../assets/history_experiment/benchmark_evidence/files/stage_b_cpu_verified.json) [CPU SAE place check](https://github.com/OpenSAE/SAEBase/blob/ed52342b706d8ad6c0e04cbe2108e69cc0dc9348/conf/coverage/stage_c_cpu_verified.json)]
