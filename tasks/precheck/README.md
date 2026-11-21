# Precheck Resource Preparedness and Physical Inspection

## Purpose

Prepare and check selected data, models, SAEs and mission conditions, providing implementation judgement and specific reasons.

## Scope

Automatic check for Precheck or official tasks that are clearly selected; resource actions are processed according to current request and configuration.

Precheck uses the sole master control `conf/config.yaml` of SAEBase to reuse the existing Dataset, Model, SAE and scientific interfaces. No new downloads, tasks management or registry have been added.

## Implementation process

`task.execute=false` or `--cfg job` keep the configuration for no installation, download or calculation. After setting `task.execute=true`:

1. Parsing accurate models, SAEs, layers and inspection targets.
2. Try to install the missing analytical dependencies, fix the installed numeric stack versions of the Torch, NumPy and others, and validate the import in the new process.
3. Prefer local complete caches; attempt to download the selected model and the required SAE file when a file is missing, recording the repository version and cache position. A SAE check does not download all layers of the warehouse.
4. When no visible hint is given, complete the data by selecting the dataset preparation interface, reusing the existing generation, coding, filtering and best rules.
5. Checks data products and reruns the selected SAE or task check. Resource preparation is not equal to physical check passed.
6. Maintain preparation records, original logs, inspection results and product audits.

```bash
saebase task=precheck task.execute=true \
  task.precheck.mode=task 'task.precheck.check_tasks=[capture]' \
  runtime.device=cpu experiment.sub_name=basic
```

Dataset and Model are still selected from General Control `dataset=...`, `model=...`. `task.precheck.mode` Supports `sae`, `task`, `all`. The list of tasks is to follow all seven scientific tasks when available; benchmark, Circuits, explain, Train must provide practical parameters, and Precheck does not produce a fictional plan.

## Default Preparatory Policy

- `auto_setup`,`allow_network`,`allow_install`,`allow_download` Default on. Visible network closure or preparation can still be used for offline diagnostics.
- The missing file download budget of the model and SAE defaults 16 GiB; prep 1,800 seconds and data ready 1,800 seconds. Physical check still uses `max_seconds`, `max_samples`, `max_tokens`.
- Default data ready to be generated 32 candidates for at least 1 retention. It is the budget for the preparation of the minimum inspection data that is clearly recorded, without changing the original dataset filter conditions, and does not represent a full assessment of data or scientific acceptance.
- Complete data are directly validated and reused; data are ready to be submitted with locks and temporary directories, and incomplete products are not covered.
- The equipment is specified by `runtime.device` and the physical inspection schedule is specified by `task.precheck.task_configs`. Delete `allow_compute`, `allow_gpu`, `allow_native` the second confirmation switch; the execution request will not wait for manual confirmation after download.
- The actual budgetary requirements, such as a maximum of two steps for primary training, are retained.

## Record & Failed

The result is saved in the current Runtime directory:

- `metadata/precheck_result.json`: Preparation phase and actual inspection results.
- `metadata/precheck_audit.json`: Report document integrity check.
- `tmp/precheck_jobs/<stage>/metadata/`: Request and subprocess back. The official mission automatically prechecks are stored at `metadata/automatic_precheck/` for the first time, and then retry to write to its separate subdirectories `attempts/`. Each time a list is executed, this return is indicated. The shared model or SAE loading failure will also write back.
- `tmp/precheck_jobs/<stage>/logs/`: Full standard output, error output, and installation, import and download events.

Read the request range, phase and actual object, and then read the corresponding results and logs. The specific causes and recommendations in the result are used to supplement dependency, resources or inputs, and the completed content is reused as per the task protocol.

BLOCKED 's reasons are recorded for failure in preparation for installation, source access, downloading, data preparation, budget shortfalls, etc. The software implementation error is still maintained as FAIL, which is not disguised as an environmental problem.

These records demonstrate where the software is going; they do not automatically demonstrate model performance or independent scientific acceptance. Offline tests use clearly marked synthetic responses and models, which do not replace real chain operations.

## Performance precheck and auto-modification

`task.precheck.mode=performance` Yeah. `task.precheck.check_tasks=[capture,intervene,generate]` The real task assigned to perform small-scale performance tests. Capture Default check `whiten`,`non_whiten`,`g_s`,`I`,Available `+task.precheck.capture_modes=[g_s,I]` Zooms out. One-time pre-check selects one model, one. SAE and layers; multiple resources run separately.

```bash
python -m saebase.runtime.train model=gpt2small dataset=gsm8k \
  task=precheck task.precheck.mode=performance task.execute=true \
  'task.precheck.check_tasks=[capture,intervene,generate]' \
  '+task.precheck.capture_modes=[g_s,I]' experiment.sub_name=performance_probe
```

The actual `capture`, `intervene`, `generate` defaults on `task.performance.enabled=true`, and the official Runtime autopretest will start running the matching performance probe and recommend a watch to task. Set for false to keep the cursor to work without performance adjustments; the GPU numerics policy is still determined by runtime. Numerics. Configure checks (`--cfg job` / `execute=false`) do not execute the probe. Qwen3.5 direct commands are subject to a dedicated interpreter in `conf/base/runtime/runtime.yaml`; the matrix entry is automatically selected through Runtume.

The public budget is covered by `base/runtime/performance.py`, by default:

| Parameters (at challenge.performance) | Default value | Purpose |
| --- | --- | --- |
| enabled | True (standard task configuration) | Enable automatic pre-checking and reporting of performance |
| batch_size | auto | automatic selection;or you can specify positive integers |
| batch_sizes | [1,2,4,8,16,32,64] | Check candidates and add the actual unit ceiling |
| probe_samples | 3 | Number of performance probes;no changes to the number of Datasets produced batches |
| probe_states | 64 | Status controls for Capture batch searches; the actual probe I still fully calculates 1024 states |
| warmup / repeats | 1 / 3 | The baseline and candidate are preheated and measured several times. |
| max_seconds | 120 | Soft time budget for each candidate search; forwards are completed for ongoing implementation |
| memory_fraction | 0.9 | Current Process Peak Reserved Visible Upper limit for total display of devices |
| atol / rtol | native: 1e-6 / 0；vllm: 0 / 0 | Numerical examination threshold for the answer average log probability |
| logit_atol | native: 1e-4；vllm: 0 | Intervention/generation of floating point indicator thresholds; generating token, ranking, string, etc. discrete output must be fully consistent |

The budget that is not included in YAML is covered by Hydra `+task.performance.<field>=...`. `task.precheck.task_configs` is transferable to the actual task configuration, such as zero-generated `generate: {generate: {intervention: ablate, value: 0, max_new_tokens: 8, reuse_prompt_observations: true}}`. A comparison of the same parameters should be used for the task to be run.

Optimizing ** the same input, and ** the same workload, selecting the highest candidate for a numerical check, retaining the remaining amount, and measuring the highest; keeping the original configuration under automatic mode if the candidate is slower than the original configuration. OOM records and stops searching for a larger watch, and actually runs OOM reduces the bat and performs a numerical check. The result is to give the best actual value in the budget and candidate, without a commitment to the best of the whole. Repeats the parameters by model, task, and length; decisions without net gain will also slow down. Normal Runtime automatically pre-screens these recommendations to missions that follow, without having to place all the mandatory lengths in the same bat. Capture continues to check the representative control of each subject and feature tasks; this is a set of physical controls, not proof of all possible inputs.

Major engineering optimization:

- TL pre/ back position and first half model cache for HF decoder; HF stops at the selected layer when preparing the cache, avoiding useless tail layer and full wordout.
- Capture status is a genuine batch, same original code status is reset, the decode baseline is reused, and the four quartiles are interactively ranked. Decoding is still line-by-line to original SAE, keeping the order of operations and LN statistics.
- Public observation and inspection of intervene check reuse, tail-layer re-laying, and batch processing of feature conditions.
- Generate's public baseline generation, batch processing of progenital feature conditions; butch1 is the same as larger battling. Each generating track retains its full prefix and original random seeds, and reuses do not contain obsolete editing after KV.
- Inference_mode, keep only the desired logits slices and avoid the view of the cache vectors to keep the entire Sequence xvocab space; models and SAE reuse and checkpoint verification leases are shared within the same pre-check range.

`reports/performance_precheck.md` is the performance precheck master report, JSON saves the stage distribution of the probes, logs the sample by sample, and the reasons for all candidates ' insinuation/surveillance/in error/rejection. Each task saves the actual running time at `reports/performance.json`, `performance.md`. The report provides average, standard deviation, minimum/maximum, P50/P90/P95/P99, distinguishing between amortization time for Question, material comparison and answer state; stages are embedded and cannot be directly associated.

The benefits are calculated on the basis of the original configuration of the same probes and the median preheat of the candidate, the candidate cache preparation being included in the stowage comparison, the search, the true execution of the entire state and the sharing of separate records of pre-processing. The report also shows the amount of time spent on each item after the first modulation, the amount spent and the number of units expected to be returned. The fractions of a few probes are only descriptive statistics; they are not full results of the 500 questions or the overall confidence interval. There is no item-by-item optimization that only reports the benefit of the portfolio and does not duplicate the same benefit because of multiple optimizations.

This real-real-performance probe of the four main models, its original source and its comparative calibre are presented in the [Performance Performance Assessment Report](../../assets/history_experiment/maintenance_evidence/files/task_performance/performance_report.md).

Standard operating usage`runtime.numerics=vllm`,Unified reuse before loading the model vLLM Stabilization strategies, strict comparison of scoring and output, and the difference must be 0;Do not count silent regression values as pass. History value paths`runtime.numerics=native`,You can't mix the same work process.[Consistency and acceleration of batch ratings](../../doc/code/documentation/source/development/batch-invariant-execution.md).

The normal Runtme portal loads only one model and SAE, automatically pre-checks with the official mission and directly consumes the recommendation. The calculation version writes into the configuration before the running judgement; the version is recalculated and the same version continues to be restored according to the mission completion strategy. The performance precheck of I also takes all 1024 states from a saved feature, recalculates independently and requests the same item by item.

Standard Project Settings have been fixed to `runtime.cpu_threads=4`, with unified control over the Torch and BLAS threads and additional parallels to the cutter; Intervene/Generate Default to open observation re-uses. Normal Runtime, pre-check and matrixworker uses the same settings to save the actual values with the results and performance reports. The precision of calculation, stabilization of the Attention, existing compilation paths, significant surplus and single model sequences are validated, as specified in [project default](../../doc/code/documentation/source/development/batch-invariant-execution.md#engineering_defaults)].

Release unused partitioner caches before the visible candidate measurement, reset and measure peaks; 90% of the display caps continue to include the reserved space that is really needed for this job. This treatment avoids prefixing or larger cascading, resulting in all candidates being wrongly rejected. The candidate records include `reserved_before_cleanup_bytes` and `memory_measurement_policy` to identify the measured calculus; numerical consistency conditions remain the original value.
