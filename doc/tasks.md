# Tasks

## Purpose

Description of the purpose, selection, output and limitation of the task.

## Scope

For the selection of actual operations, the interpretation of results and the maintenance of project descriptions.

## Purposes and access

The task is a formal task that can be selected at `saebase task=<name>`.

## Shared Parameters

`task.execute` decides whether the task to support the switch is actually being performed. `task.sae_key` and `task.layer` select features and layers. See [item parameter](conf.md) (for configuration files, default values and overlay rules, see how the model's operating equipment is set).

## Component

### `precheck`

Checks whether data and weights support a physical experiment.

Select with `task=precheck`. Use a range of checks to determine the availability of the environment and combinations. The time and conclusions in the report should be checked only for the range. [Target configuration](../conf/tasks/precheck/task.yaml)] [achievement](../tasks/precheck/task.py)]

### `capture`

Captures a layer from the input hint and encodes by SAE. You can save end words active, word-sorted, integratorized or behaviour-changed character selection. The result is a candidate, a value and the selected reference group.

All Capture mode defaults that the original model continues to be greedy of the same input, saves it at `model_answer` and is available downstream with the feature results; answers fail to preserve the feature and record the missing item. Generates a default of 256 for length, a default of 8 for batch, with parameters `task.capture.model_answer.max_new_tokens` and `task.capture.model_answer.batch_size`. Results are available to call `CaptureTask.refresh_model_answers()` only. [Model answers and additional interface](../tasks/capture/README.md#each-capture-includes_original_model_reply)].

Select with `task=capture`. The ranking is influenced by data, layers, reference groups and filters. High-activation and high-activity influences are explained separately. [Task configuration](../conf/tasks/capture/task.yaml)] [achievement](../tasks/capture/task.py)]

### `intervene`

Changes SAE to the selected characteristics and conditions of intervention indicate that the score is compared with the target answer after the intervention. It can be entered in the capture candidate, executed independently of multiple candidates and preserved the conditions.

Select with `task=intervene`. Check first the numerical policy, feature selection and baseline. The fraction change under the same conditions reflects the effect of the operation on the model's reading. [Task configuration](../conf/tasks/intervene/task.yaml) > > >](../tasks/intervene/task.py)]

### `generate`

The text of the given reminder is generated word by word, and you can add value to the selected SAE feature during the process of generation. The report saves the text, each condition, and the reason for termination.

Select with `task=generate`. Generates the upper limit, temperature, the length of the context and the selected position to decide the output together, and the conditions for comparison should be retroactive. [Target configuration](../conf/tasks/generate/task.yaml)] [Achieving](../tasks/generate/task.py)]

### `circuits`

The selected text and model are handed over to the configured transcoder circuit tracker to generate a reviewable chart tracking to analyse some upstream impact relationships of the selected output.

Select with `task=circuits`. Matching the model and transcoder is required to perform task approval and timeout control. The image is dependent on the selected hint and tracking configuration. [Target configuration](../conf/tasks/circuits/task.yaml)] [achievement](../tasks/circuits/task.py)]

### `explain`

The characterization-based sample produces a description of the natural language, which is then combined with a data check for interpretation.

Select with `task=explain`. Explanation is the assumption to be checked, the accuracy of which is based on the leave of sample evidence. [Target configuration](../conf/tasks/explain/task.yaml)] [achievement](../tasks/explain/task.py)]

### `benchmark`

A reviewable assessment is organized by a designated agreement to read models, data, SAE and evaluation settings and to preserve the corresponding metrics, scores or comparative results.

Select with `task=benchmark`. The replacement of indicators, models, methods or assessment sources can change meaning, and historical comparisons require corresponding protocols. [Target configuration](../conf/tasks/benchmark/task.yaml)] [achievement](../tasks/benchmark/task.py)]

### `train`

The training SAE is activated with the configured training plan and the selected model. The goal is to obtain parameters and training records that can be used for diluted coding.

Select with `task=train`. The model layer, input data, equipment, budget and output path needs to be checked. Weights only match the clearly matched activation point. When `saelens` is selected, the interpreter needs to be able to import `sae_lens`. When `sparsify` is selected, a new and available Sparsify backend is needed, which supports Top-K and Transcoder programs here. The training pre-check will confirm the selected backend, model snapshot and data input. [Task configuration](../conf/tasks/train/task.yaml) >](../tasks/train/task.py)]


## Characteristic capture method

Select with `task=capture task.capture.mode=<mode>`. The specific parameters for the SAR regression, over-representation and RSS context comparison are read from `task.capture.sar`, `task.capture.i` and `task.capture.rss`, respectively. The common selection and answer rating are assigned to the capture task `task.capture.selection` and `task.capture.answer`, respectively. The number of candidates is adjusted by `task.capture.selection.candidate_count` and the number of outputs is adjusted by `task.capture.output.feature_count`. For example, the number of returns to the intended sample is adjusted by `task.capture.sar.n_fit`. All the features to be captured in each model can be the characteristics of the same model location, and the model determines the evidence for selection or rating.

### `non_whiten`

The range is influenced by the weight and scale of the SAE.

### `whiten`

Use the reference issue averages and standard deviations to indicate the current degree of activation of deviation from a common level, and to facilitate the identification of candidates who are not normally active at a higher level.

### `g_s`

Default to list each single candidate in the shared pool, sorting by the deleted g(S). The shared pool excludes features with a strict density of more than 1% and filters them with terms and cross-references, then extracts up to 20 of the absolute z-score from the original non-zero feature of the topic. The old independent primary range is enabled by `candidate_policy=legacy`.

The target output is rated by the behaviour of directly observing the retention or change of candidate characteristics. It is appropriate to find out which candidates are more influential in the given answer, using the same objective and reading.

### `sar`

For candidate sub-samples and regressions, the difference described by the single characterization is estimated, with a ratio by configuration.

### `I`

An estimation of the symbolic interaction of the specified feature is to be read in the actual sub-activation, requiring a list of candidates for the defined feature.

### `rss`

Compares the characteristics of the same target text in multiple contexts, selecting the characteristics by range, frequency and direction consistency. Captures using a visible hint, at least two context and end-word complete feature. [A capture usage instruction](../tasks/capture/README.md) (for setting the selection and rating for capture, e.g. comparing the number of candidates with the actual output)

## precheck

Select with `task=precheck task.precheck.mode=<mode>`.

### `sae`

Checks whether the selected model location and SAE are available and compatible, and defaults on the list of SAE recommended by the model.

### `task`

Select the tasks and their configurations to be checked in a frame, with a small number of inputs for actual reading and computational problems.

### `all`

Check SAE and check the tasks according to this selected range.

### `performance`

The formal experiment still records the conditions that are eventually applied. [Preview configuration](../conf/tasks/precheck/task.yaml)]
