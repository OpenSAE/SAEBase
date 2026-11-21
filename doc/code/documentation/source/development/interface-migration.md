# Current realization and interface migration

## Purpose

Describe the sole attribution of scientific achievement, the current interface, and the way in which the old transfer is made to the current realization.

## Scope

The developers of the current SAEBase source code, configuration and plugin are moving to call; the scientific records continue to retain original values, sources and calculate identities.

This document preserves the commands and parameters of the historical migration. Examples of currently captured subtasks, open parameters and executables can be found in [configuration parameter](../../../../conf.md), which describes how to use `task.capture.sar` to return and how to select an exhaustive candidate with `task.capture.i`.

## Current entrance

Eight types of tasks are at the root level. Activated health assessments are provided by benchmark, `saebase` commands that are directly connected to the use. Retain pre-emptive tips, environmental inspections, interventions, evaluation, reliability analysis and synthetic control commands. Multi-models are compared to `saebase-benchmark --config <comparison_config> --dry-run/--real/--report-only`. Formal execution and model downloads continue to require the corresponding authorization.

| Old realization portal | Current realization portal |
| --- | --- |
| saebase.interventions | It's not like it's a good idea, but it's not a good idea, isn't it? |
| saebase.benchmarks | tasks.benchmark.execution |
| saebase.reliability | tasks.benchmark.reliability |
| saebase.integrations | tasks.benchmark.methods |
| saebase.generate | tasks.generate.serving |
| saebase.circuits.graph | tasks.circuits.attribution.circuits |
| saebase.explore | tasks.capture.explore |
| comparison calculations in saebase.release | tasks.benchmark.comparison |
| tasks.benchmark.extensions | saebase.extensions |
| saebase.data | saebase.datasets, physical achievement at root level |

Plugin only uses `opensae.saebase.extensions`. The model shares the existing extended object and registration status with SAE, the indicator. The old plugin group name exits the current access and the example in the current issue uses the same name as the document.

The task entry returns `EvaluationResult`; the specific operation function maintains its own measurement. The results of the old Task dictionary are now stored in the outer layer of the results summary. Precheck continues to express its inspection status by PASS, FAIL, BLONKED, NOT_RUN.

Common configuration no longer requires training. max_epochs. The running directory cleanup option is runtime. clear_experiment_leaf, and the training-specific plan continues on a training mission. The SAE and layer default values for Task are determined by the selected model and resource.

## Retentional capacity and removal mechanisms

### Number of Capture features

`task.top_feature_num` Controls the number of features returned by Capture, default 10. Returns do not exceed the actual number of features, sequences continue to use active absolute values and records the active values of the original symbol. Zero, negative, boolean and non-integer values are rejected before loading the model. The parameter is held by the Capture task, the direct operational entry is re-checked again, without changing SAE code or model activation.

The current software release uses the existing manual distribution process, which is aimed at verifying the exact source code and the current distribution package. Multimodels compare the retention of weighting identities, caches, statistical analyses and summaries of results, and release software versions from fixed requirements of at least two model communities.

Primary training is combined into an achievement within the mission, retaining precision settings, data entry checks, primary recovery and check point identities. The model's flattened catalogue exit activity code, and the model key still points to the counterpart and the example.

NNsight works to keep independently available baseline generation, running checks, layers detection, and active export. Reliance on editing of research modules that no longer exist to remove the teacher-force branch; feature editing using the current generate/intervene interface. The program gives a clear error to unsupported editing requests and does not replace them with baseline calculations.

## Scientific identity and text

Tool files are named after their purpose. Numerical identity is used to maintain the original values of fixed-format strings that have been used in fingerprinting, including the identifiers for activation, coding, sample and intervention measurements. They express the current single calculation identity and do not include the old version of the selection branch. Changes to these strings change the cache key and are treated separately from the normal file.

There are real measurements and source data that are raw bytes. No experiment or runs directories have been found in the current working area and no conversion of results has been carried out in this round. The external mounted migration of the actual Manifest and content identity of the user after the result has been specified clearly by the user; the unknown data location is not extrapolated.

## Authentication status

Reconstruct the candidate to complete only the source static review. New and updated test source code covers the task contract, configuration independence, results search and indicator integrity, but does not perform pytest, build, install, model download or GPU acceptance. Values equal and installation are reported on the basis of evidence of actual execution.

## Description of migration for improved functional consistency

This function improves the performance of offline product inspections and installation inspections, with the results reported in the current delivery record; synthesis certification does not imply real weight or scientific acceptance approval.

1. Capture 's `task.top_feature_num` is effective; the default value is still 10, returning all features when exceeding the total number of features.
2. Multilayer SAE no longer takes the last item of the hidden layer list. Select clearly by `task.layer`, or by `layer` or `model.default_layers.<sae_key>` in the pre-check list. The default master configuration has clearly selected the original Gemma layer 25.
3. precheck and implementation of the same resources. External resources need to be pre-registered; quarantine precheck needs to be installed `task.extensions` entry point Name. The provider can declare capacity and check the actual capacity after loading.
4. SAEBASE is launched from `conf/config.yaml`, with all tasks in an explicit combination of model, dataset and task. Benchmark's `task.protocol` only describes the evaluation method; models, data and visible samples in `task.benchmark_config` files are rejected when they conflict with the master control, and global selection cannot be overridden or deleted. The summary records the actual object and method source, and the relative input path is analysed by the original starter directory.
5. Add a completed summary and format version of the mission results; the old records are readable. Foot should have read the performance status and completed summary at the same time, processed exit codes 0, 1, 2, and no longer extrapolated all work from the existence of the report or the normal return of the process.
6. `runtime.mode` Supports `new`, `resume`, `overwrite` with an empty configuration equivalent to `new`. The old `experiment.overwrite` is converted to a corresponding intent and the conflict configuration is rejected. Restoring the original configuration, the account book and the wrong history; the old operation without an identity restoration will not be automatically restored.
7. The result is a return candidate instead of a re-entry permit. When the old book lacks a sample or protocol identity, the reasons are more uncomparable and the margin is not forged.`compare_runs(..., pairs=[((model_a, sae_a), (model_b, sae_b))])` Visible selection across resource pairs.
8. Research Lab forwards only completed abstracts approved by the original release process. Old records are missing when they are not carried with a summary and are not presumed to be complete; this time they will not be open for local running or private input.

Activity-health preparation samples are checked by existing PromptSource and its actual reading path; SAE layers are also analysed by public choice. `verify_matrix` for comparison of primary models still requires at least two model communities, at least two SAEs per model, and complete indicators and audits. This round does not replace software implementation with scientific acceptance approval; the “demolition model community requirement” in the previous note does not represent the release of the original acceptance condition.

This session does not change historical values, existing indicator formulae and cache identification formats, replace official Atlas, and does not automatically upload or release it.

The parallel research startup configuration and the Benchmark special startup configuration have been deleted and the old startup aliases are not maintained. The current caller is changed to the default master control or `compose(config_name="config")`; action-health by the components required by the same master package. Gemma 's default layer 25 returns to its own model configuration, and the master control does not contain the layer settings of the specific model.

`saebase` The Lord's orders are now directly into the public domain. Runtime,use `model=... dataset=... task=...`.The conversion of specific parameters to activate the health assessment is still held by its scientific tool, which is then combined with the same master control; the evaluation tool is no longer used SAEBase Main entrance.

The directory is unified from a global selection; the logic of Task customises the public directory and deletes the global model/dataset has been removed. The model collection uses `model.members`, and the tasks must be declared to support the input capability; the collection does not trigger Runtme assigned subtasks. Old directories and historical scientific results do not automatically move or rewrite, and source code and configuration identity changes do not quietly bypass the restoration verification.

## Precheck Autoprep

The default attempt to install missing analysis dependencies, download selected weights and prepare selected data after Precheck is executed. The network, installation and download default is open, and the offline mode is still visible; configuration and presentation only does not trigger preparation. Preparation failed to keep a specific phase, actual attempt and complete log and return to BLONKED.

Removes `allow_compute`, `allow_gpu`, `allow_native` switches; the selected inspection plan continues when resources are available. The equipment and primary plans still need to be valid, and the actual calculated budget, such as training inspections, remains valid.

Data sets such as IOI are generated and filtered through the existing Dataset preparation method. The minimum preparation budget is clearly recorded at 32 candidates, with at least 1 retained sample; it does not relax the original filter conditions and does not refer to the minimum check data as a complete scientific data set. The data are available in original bytes; new data are submitted through temporary storage, validation and cataloguing.


## Capture's non_white/white mode

`task.mode` is a Capture method parameter, default `non_whiten`, history has not stated that this field is run according to non_whiten. This default maintains the original activation absolute Top-K; `whiten` using the same selected Dataset, Model, SAE, Hook all prompt token position to group the feature average to the overall standard, then to rank at the end of token 's absolute z-score and to follow the feature ID ascending order. The two models use different method identities and cannot skip each other as a completed result.

Example: `python -m saebase.runtime.train dataset=humaneval model=gpt2small task=capture task.execute=true task.mode=whiten experiment.sub_name=whiten_run`. Toggle the old method to use `task.mode=non_whiten`. `task.whiten_chunk_size` Default 256, only control the size of the original tokenwise SAE code and write the statistical identity, FP64 cumulative central rectangles with the total number of token N as the differential. Use the currently selected complete data set to combine descriptive statistics, including primary zero-activation; BOS with real input and hint suffix, without adding answers or padding.

Whiten does not change featurity ID, SAE weight, original activating value or decode. Each result is retained at `activation`, adding `zscore`, `mean`, `std`, `zero_native_activation`, `whiten_eligible`. The standard difference is 0-hour and excludes Top-K; the valid item returns all valid items when it is insufficient top_feaute_num. All dimensions are given mathematically undefined at constant times. The original zero-activation item can be selected by negative z-score. `active_features` continues to indicate non-native zero items; `top_features` is ranked according to the mode. There is no SAR-specific language blocklist.

Statistical Save As `metadata/capture_normalization.npz` with `capture_normalization.json`,Record average, variance, standard deviation, active dimension, token Total, model/SAE Byte Identity, Input token The statistical document is a necessary product of the result. prompt Pending CPU End token snapshot,The ranking stage is reused to avoid a second model forward. Any input failure at the statistical stage will fail the current round and will not release standard parameters for part of the data.. BatchTopK Waitover token Competition encoder needs to define a separate statistical agreement, which is currently not supported by explicit reporting.


The Capture model follows the same Dataset/Model/capture directory level and saves every run using an independent source. sub_name. Manifest adds a common tab_mode metadata, and has list_runs already supported filters with name_mode, such as list_runs (root, tab= "capture", tab_mode= "whiten"). When the old capture record lacks Mode, it is recognized as non_whiten when read with the default configuration rule of CaptureTask, and the historical file is not written back. Whiten compares with non_whiten's mode, statistical identity, configuration, and does not replace either.


## Neuronpedia Renames Repository

The `fnlp/Llama3_1-8B-Base-LXR-8x` of Llama Spope LXR-8x and `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` have been verified as pointing to the same repository and submission, explaining that the source matches the explicit change of name for the identification. The SAE physical folder, dimension and Hook are still required; other same-name warehouses do not automatically recognize the same resource. This fixes the existing interpretation acquisition path for precheck and Capture. The new operation automatically captures the corresponding explanation, and the historical results are not read for operation rewrite.

This HotpotQA saves two captures `reports/capture_explanations.json` to supplement the record with the original Capture account SHA-256 binding explanation, and does not recalculate or rewrite the original activation as explained by the original Capture ID query and its source. The open source unexplained feature retains the missing state and does not generate alternative labels.


## Attribution of tasks and key inspection adjustments (2026-11-14)

The public entrance calls Task's own parameter validation before preparing the running directory, and gives the verified Task case to the executive entrance to avoid the normal link calling again the same check. `base/runtime/sae_operations.py` Only the shared resource operation and a small number of input type support functions are maintained; the specific parameter rules are located at `operation.py` in each Task. The direct use of Task or `execute_task` is still checking the configuration at the entry. The generic executive layer does not replace the task by writing " Done " ; the specific task declares its own summary of completion.

Automatic precheck Yeah. capture／intervene／generate Keep Current Real SAE Sampling, yes. benchmark／circuits／explain／train Call the task itself `prepare_execution`,Prepare protocols, input, and backends without repeating the official scientific calculations. Training readiness no longer requires a formal plan of up to two steps; explain that preparation no longer requires a hypothesis to be produced. The original mission ' s own calculations and resource budget continue to be valid. Primary training, interpretation, and mapping tasks do not have any weights on downloading external model labels..

The existing single input method for intervene/generate/circuits explicitly includes in the result `input_scope`: the number of visible prompt or Dataset Article 1, requests and processing. This change does not include all Dataset evaluation protocols for these tasks. Public prompt reads no more than a uniform limit of 4,000 characters; the actual limitations of the context and the original backend are responsible for the corresponding execution.

The active Dataset input file content summary enters an existing sample selection key; a change in the same path content creates a new choice, and the historical choice does not rewrite. The relative input path resolves the original startup directory. Samples, original labels, scientific formulae and numerical accuracy of natural data across the sprit remain the same.

SAE loads records with the same publicly available source of explanation. Llama Spope uses the stability key `llama_scope_lxr_8x` associated physical SAE for the release of SAE Lens and the interpretor library to remove the specific old and new repository name conversion in the explanation module. When no source has been registered, the explanation is clearly recorded as not obtained and is no longer considered as complete.

CaptureTask provides `refresh_annotations(folder)` and `read_samples(folder)`. The former repeats the saved feature ID and the real load-up of the identity to supplement the explanation, without load weight, and does not re-activate; the latter combines the additional account with the supplementary account when reading. The regular `read_run`/`inspect_run` for Runtime uses the small completed update provided by Task. History and activated books are not rewrited and do not run through all the samples at the time of regular operation.

Re-use of the embedded observation session of the same loaded resource is authorized to recapitulate and interpret snapshots, with the original final check performed at the end of the outermost layer. IOI The sample no longer triggers unnecessary. SAE CPU/GPU Round trip..

This time no changes will be made to the lists, test, research Labs, and Atlas; no parallel registration, running or test frames will be added. The assertion that the old warehouse name has been changed, the meaning of the old explanation completed, and the old internal call for signature needs to be updated, and its failure cannot be interpreted as a failure of actual scientific calculations.


## SAR mode and feature number renamed for Capture

Capture's configuration parameter `task.top_k` is renamed `task.top_feature_num`, and the default value is still 10. The old parameter is clearly reported as error; the caller should modify the configuration or command. Generates a sample, SAE architecture and reliability analysis to maintain the original meaning of `top_k`. The historical running record maintains the original content and the new configuration does not automatically repeat the old running.

`task.mode` Supports `non_whiten`, `whiten`, `sar`. SAR selects `top_feature_num` characteristics according to original activation absolute values, with the same values sorted by feature ID; it measures the combination of the whole set of characteristics relative to the specified behaviour and does not produce a single SAR ranking. The original feature activates, the Neuronpedia comment and the normal Capture result remain.

```bash
saebase model=pythia70m dataset=ioi task=capture \
  task.mode=sar task.layer=3 task.top_feature_num=16 \
  task.sar.source=/absolute/path/to/SAR/sar.py
```

The above is the only configuration entry; the actual calculation requires `task.execute=true`, the available resources and the corresponding execution authorization. `task.sar.source` refers to the user's reliable original version `sar.py` or its warehouse directory; the configuration verification only checks the input and does not execute the document. Imports its `coalitions` and `sar` at the time of execution, keeps and checks the source file SHA256, does not copy or rewrite the estimate.

Default extract per sample 384 Subset, 256 For the preparation of, 128 Numbers used to set aside the assessment, the probability of each feature being retained is 0.5.The number of proposed samples should be covered at least `1 + K + K*(K-1)/2` Second-order coefficients, with a requirement to combine at least two samples each. K Must at least be 2 and no more than SAE Dimensions, not in SAR Mode is silently cut. Random seeds come from the global context `seed`.

The behaviour temporarily supports the target to specify the output position as different from the control token logit. The Dataset sample provides `target_token_id`, `reference_token_id`, or the single token `tokenization.correct_ids`, `tokenization.distractor_ids`. The visible `task.prompt` uses `task.sar.target_token_id` and `task.sar.reference_token_id`. The label conflicts, multiple tokens, missing or illegal fail explicitly, without guessing from the predicted results, and without aggregating the answers. Dataset still holds the sample range and source.

Intervention uses snapshot of the same capture to remove only unreserved selected features from the sub-centre, while the rest of the features remain in their original meaning with the original refactor. Default decodeable and original reset check; `require_affine_decode=false` explicitly uses another decoder-mediate protocol identity. The check is not a mathematical proof of the linear nature of any decoder.

Add `sar` to each sample result, containing indicators, undefined indicator names, features ID, behaviour, sampling parameters and source file identification. `reports/sar/*.npz` Saves masks, proposed/indicated, responsive, original baselines, tokens, capture-activated and selected values and registers to required_articles. Constants are marked as mathematically undefined, following the existing EvaluationResult; notes refreshing the mathematical status.

SAR achieves the `tasks/capture/response_surface.py` and `sar_mode.py` original document loads the duplicate `base/interfaces/paper_sources.py`. `tasks.benchmark.methods` continues to export the same `evaluate_sar` for existing callers; the old `tasks.benchmark.methods.response_surface` file path moves to Capture. Public Runtime and Research Lab is not modified. At this stage, the cluster intervention is carried out sequentially and does not include a special upstream batch or postfixing acceleration.

The SAR decodeability diagnosis activates the use of FP32 plus minus to the FP16/BF16 model to avoid miscalculating the low accuracy of multiple independent patches to the non-linearity of the decoding device. The upgrade is used only for diagnostic purposes, and the formal subset intervention still uses the original snapshot and model raw accuracy; the default refusal of the non-linear decoder to remain valid.


## Tool capacity assigned to formal assignments (2026-11-14)

This section continues with the task attribution adjustment above: this round has been processed tils, tests and scripts. The name of the public order has been changed and calls the corresponding Task realization `saebase-benchmark`, `saebase-intervene`, `saebase-audit` and `saebase-ground-truth` respectively. Benchmark and the true calculation of the bulk intervention have been harmonized in Runtume to record the operation, execute the automatic precheck and transmit the completed state; `--output` indicates the active workspace root directory at the time of real calculation, with the product using the existing Dataset/Model/Task level.

- The bulk intervention method configuration is at `task.protocol`, the model and SAE are taken from the global selection, token enter using `dataset.samples`. The JSON command entry only converts the original experiment file to this configuration and does not perform a separate scientific process.
- All token's diluted feature captures are used by `task=capture task.output_scope=all_tokens task.mode=non_whiten`. `task.max_tokens` and `task.max_active_per_token` state, respectively, the upper limit of input length and non-zero line per token complete, which is clearly failed and does not interrupt the thin line; the result remains the final token's summary. The output range does not currently support whiten/SAR and cannot be interpreted as having been achieved.
- Multi-condition generation automatically selects the features on the end-end content token by `task=generate task.feature_selection=last_active_content_token`, or in `task.conditions` you clearly specify the existing original intervention conditions. The result records the baseline, modified token and intervention steps for each condition.
- The SAE is prepared individually and the original encode/decode detectable at precheck:`task=precheck task.mode=sae task.resource_only=true`; the independent entrance is `python -m tasks.precheck.checkpoint --sae-key <key> --layer <layer> --device auto --output <directory>` and does not load the base model. It only proves that the software is available using clearly marked composite input. The historical workflow provides `--coverage-catalog` retention of the source field of the original overwritten protocol.
- The login and comparison of numerical references is provided by `saebase.api.reference`; site coverage and upstream SAEBench work processes belong to Benchmark. The fixed reference entry that is retained uses these realizations and no longer maintains independent algorithms.
- Multi-model, multiDataset capture schedule is `python -m tasks.capture.matrix --capture-mode whiten --tag <run_tag>`. The selection list is `conf/tasks/capture/task.yaml`, and each group still constructs the SAEBase main configuration, with the public entry point auto precheck post-capture. The duplicate of the old fixed batch drive has been moved to the official matrix entry.

The old Qwen/vLLM temporary incapable sorting, repeat GPU/resource detection, completed experimental handover scripts and shell files only packaged with other commands have been removed. Tests for retention of protometric values, identity/samples, failed dissemination, recovery and installation; no longer determine the correct function by deleting the number of packages, files or migration files. Completed Atlas source catalogue migration claims are retired; Atlas originals, central verifications and official release processes are not modified. Historical scientific records are kept in the original language with frozen documents; the old commands in them only describe the execution at that time.

The current development document navigation check only validates the link to the active document; the central historical archive continues to be responsible for freezing summaries of the original language and not repeated in the project navigation check. The exclusive maintenance of Research Lab continues to be suspended.


## Achievement and results expressed in functional name (2026-11-14)

The activity code description, error reporting, reporting and workflow options are changed to the actual functional name: Numerical Reference to Checkpoints, Activation and Character Encoding Cache, Bulk Intervention, Benchmark, Reliability Analysis, SAE Training, Backtracking and Remote Tracking. Users are no longer required to know the development phase number first.

Reliability report title changed to `Reliability Audit`. The reference function for the point of inspection is `compare_reference_activation_probe`; the internal reading function for the reliability analysis is `_read_benchmark`, and the internal cache and measurement query is also named for the purpose. The existing public CLI command remains unchanged, and the reference acceptance is `--reference`, and the reliability analysis is `--results-dir`. The old parameter aliases are used only for the resolution of the call already made. The manual real reference workflow options are `reference`, `activation-cache`, `interventions` and `all`.

The saved results, format identifiers, protocol identities, cache directories and source fields are not rewrited; these machine identifiers continue to be read in the current format. Model names, layers, slants, squares, numerical precisions, etc. retain their original meaning. Freezing documents keep their historical originals and Research Lab is not maintained. The above adjustments do not change numerical calculations and sample selection.


## The complete correct answer behavior of SAR

`task.sar.readout=answer_mean_logprob` Use `mean_t log P(y_t* | x, y_<t*, patched_h)`. Dataset `answers.py` is responsible for extracting the correct answer and source field from the true label; a visible single entry can be `task.sar.answer` to provide a standard answer. Selecting solves the question to the correct option text, the final answer after the CounterFact uses target_true, GSM8K uses ###### , and HumanEval preserves the indentation and replacement of the standard code. HarmBench 's description of behaviour and XST 's classification labels are not used as a standard answer, and unsupported when the reference answer is missing.

Characteristics end from the original problem token Catch once; answer not involved SAE Encoding or feature selection. The answer continues with a separate text. `add_special_tokens=false` Encoding and Adding to Original prompt token after;forward input only with prefixes to the answer, calculates each answer using a strictly moving tab token . No extra BOS/EOS.Spaces for text answers, line breaks for code completion are recorded in a clear continuation format. Intervention always locates the end of the original question token,Do not move to the answer with the length of the answer token.FP32 log_softmax After FP64 Average, keep uninvolved conduct check for re-alignment.

Full answer token ID, length, original answer and source, format prefix and aggregation definition is written to SAR behavioral identity, original npz added an answer_token_ids. Original SAR sampling, formulation, setting aside and indicator definition remains unchanged. This achieves the SAR measurement on the correct answer support, where the user gives a copy of the Slement Wins, Production, Search table as a methodological reference and this round does not declare it as a duplicate result.


## Model implementation window harmonization (2026-11-14)

The new load model uses `min(native_model_context_limit, 4096)`. The window rules are managed by Model `models/loading_policy.py` to read the actual model text configuration; and do not use tokenizer’s infinity position lengths. TransformerLens before building into `n_ctx` to raise the original 2048 default window of Llama to 4096, and avoid creating an excessive attention buffer. Hugging Face layer provides the same `cfg.n_ctx` public input and complete answer rating.

The executive window for this GPT-2 Small is still 1024, and the nine models selected for Gemma, Llama and Qwen are 4096. Visible token input beyond the executive window is clearly not supported; the cut-off of the existing TransformerLens string is still intended, and SAR still requires complete titles and answers. Old results and stopped batches do not rewrite and do not automatically rerun. Window changes affect actual input, new test results cannot be described as complete input results according to their actual token identity records.


## SAR pairing results for Whitten feature (2026-11-14)

`task.mode=sar task.sar.feature_selection=whiten` Use the selected data set to aggregate the available standardized statistics and select `task.top_feature_num` features on an absolute standardized score; these IDs are used directly by SAR ' s combination of interventions. Defaults remain in the order of primary activation and do not change old experiments.

After setting `task.sar.top_pairs=5`, add `sar.pairwise` to the result. The second stage `design` of the original version of SAR and the existing combination, the coefficient describes the two interactive items that retain the indicator variables in absolute order; this is the second-stage proxy model coefficient, not one SAR fraction for each of the features. The original whole set of indicators and leave the assessment to remain the original meaning, with the results recorded. Each set saves the original activity at the two ends, the standardized fraction and the Neuronpedia public explanation; a separate cross-direction is indicated as the intended explanation. The full set of IDs and coefficients are stored in the original SAR group product, explaining the replacement of the official interface following Capture.


## Run record reading and continuing processing

`EvaluationResult.diagnostics` The specific cause of the preservation problem, the impact on the mission and the next recommendation. Pre-check `reason`, `repair` and the actual evidence are used directly to form a statement. The old record supplements this view when read, and the original document remains as it is; the reasons for the missing old state are not yet documented, and alerts to the mission log and precheck report.

The formal status of implementation of the result is `EvaluationResult.status`, `completion`, which continues to convey the completion of the request. The new result `meta` retains additional information such as torrents; the caller will read `meta.execution_status` or `meta.executed` instead of `status` or judgement `status == "executed"`, respectively.

Runs the internal product using a relative path to save, `read_run(folder)` returns the file position based on the current directory. The old task result is absolute path to identify its original active directory by the location of the official mission outcome document and map it back to the current directory; external sources continue to keep their actual position. Read the process to keep the historical file as it is.

`manifest.configuration` is the official source of the current configuration, `original_configuration` retains the initial settings with the first `config.yaml`; restores the attempt to continue. The current configuration summary uses `configuration_sha256` in a uniform way. The new writing process cleans the replaced `latest_configuration.json` field with the old `config_sha256` field.

`list_runs(root, errors=issues, **filters)` returns readable running and adds unread paths, specific causes, effects, and disposal suggestions to `issues`; the default calls indicate these anomalies by warning. Separately read the specified running still returns the actual error to locate the caller.

## Motivation of streamlined projects

- Model calls using `models.plugins.get_model_adapter(key)` or registered model interfaces, with `specs.py` saving statements in the series; examples `model.py` and `compatibility.py` exit at forward location.
- Prompts that identity, path, usability and reading and writing are uniformly imported from `saebase.datasets.artifacts`, which is moved to that location by `artifact_layout`, `prompt_schema` and `prompt_store`, and that the data content and format identification is maintained at the original value.
- Component captures imports from `saebase.api.components`, explaining how to generate imports from `tasks.explain.interpretation`; generate a backend using `tasks.generate.serving`.
- The code for reading a single component YAML is used at `conf.load_group(group, name)`; Hydra selects the syntax to keep, presets in the table of contents to synthesize the original public configuration. The new same-series model does not need to add a new case forward package.
- Batch is initiated by `tasks.capture.matrix` or `scripts/capture_queue.py`. Old experiments and measurements remain in place; changes in source-code identity are still processed under the original restoration conditions.

The KV cache in fixed decoder direction is generated using `tasks.generate.serving.decoder_steering`; the SAE position editing in interp-engine is used at `tasks.generate.serving.sae_interventions`. The two independent scientific capabilities retain the original algorithms, parameters and protocols, and the actual backend depends on the delay in loading as it is.

The default position of the prompting product harmonizes the rules of the original normal reading entry: the source code operates using `data/` and `experiment/` under the root of the project, and is installed using the call directory; the visible path is used to maintain its original meaning. The builder recreates the output of the package and removes the deleted module that remains in the old build catalogue; and distributes the check to confirm the integrity of the actual component table of contents and the normal installation call.


## Tool portal to function

The capability of the original tool directory has been moved into the module. Data preparation and environment displays keep the name of the original command. `python -m tasks.precheck.sae_checks`,Numerical Reference Usage `python -m saebase.api.reference`.Short assessment passed. `saebase-benchmark --pair-protocol cpu_pair|reference_pair --real` Implementation; CPU protocol selection `--sae-key`,Reference protocol selection `--model`.

The original parameters and output content continue to apply for the storage of real weights and for the re-entry of interventions using `python -m tests.product.test_activation_store` and `python -m tests.product.test_feature_interventions`, respectively. Details of the project and Lab commands are given in the [Contribution Guide](../../../../../.github/CONTRIBUTING.md)]. The old caller is using these current access points. The format in the historical results retains the original intent.

## Example and evaluation module attribution

The original example package is called to two source-code demonstration scripts or corresponding tutorials. The test is read by `tests.fixtures.synthetic` for composite components. The original assessment of public type and planning functions is imported by `tasks.benchmark.execution`, the indicator is registered at `tasks.benchmark.metrics.registry`, the method is accessed at `tasks.benchmark.methods`, and the measurements are read by `tasks.benchmark.results.metrics`.

The statistical function continues to be provided by `tasks.benchmark.reliability`: sample group re-sampling is the same as Bootstream `bootstrap.py`, known factor and degradation measurement is the same as `ground_truth.py`, SAR and RSS are the same as `controls.py` and statistical reports are read and written `tasks.benchmark.results.audit`.

Bitpoint Batch `python -m tasks.benchmark.sites.matrix`,Check for single bits `python -m tasks.benchmark.sites.checks`,Actual evaluation used `python -m tasks.benchmark.sites.runner`.Catalogue statements and source queries `sites.catalog`,Restore Record Return `sites.records`.Use of evidence `python -m tasks.benchmark.results.evidence --output <Run Directory>`,Use of the package `python -m tasks.benchmark.results.archive --help`.These operations continue to read original scientific identities and protocols; bits, precision, seeds, division and results format remain as intended.
