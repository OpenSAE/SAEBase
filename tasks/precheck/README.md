# Precheck Resource Preparation and Execution

## Purpose and Main Functions

Prepare and check the selected data, models, SAEs, and task conditions, then report whether execution can proceed and give concrete reasons.

## Scope

Use this guide for an explicitly selected Precheck or an automatic check for a production task. Resource actions follow the current request and configuration.

Precheck uses SAEBase's single master configuration, `conf/config.yaml`, and reuses the existing Dataset, Model, SAE, and research operation interfaces. It introduces no download task, management task, or registry.

## Execution Flow

`task.execute=false` and `--cfg job` are for configuration inspection. They perform no installation, download, or computation. After setting `task.execute=true`:

1. Resolve the exact model, SAE, layer, and check target.
2. Attempt to install missing analysis dependencies. Keep installed versions of Torch, NumPy, and other parts of the model's numerical stack fixed. Verify imports in a new process.
3. Check for a complete local cache first. When files are missing, attempt to download the selected model and the required SAE files. Record the repository revision and cache location. Checking one SAE does not download all layers in the repository.
4. When no explicit prompt is provided, use the selected Dataset's preparation interface to supply the data. Reuse its existing generation, encoding, filtering, and manifest rules.
5. Check the data artifacts, then run the selected SAE or task check. Preparing resources alone does not establish that the check passed.
6. Save the preparation record, original logs, check results, and artifact audit.

```bash
saebase task=precheck task.execute=true \
  task.precheck.mode=task 'task.precheck.check_tasks=[capture]' \
  runtime.device=cpu experiment.sub_name=basic
```

Select the Dataset and Model using the master configuration's `dataset=...` and `model=...`. Supported values for `task.precheck.mode` are `sae`, `task`, and `all`. An empty task check list selects all seven research tasks. Benchmark, circuits, explain, and train require actual parameters when they need a native plan. Precheck does not fabricate plans.

## Default Preparation Policy

- `auto_setup`, `allow_network`, `allow_install`, and `allow_download` are enabled by default. You can explicitly disable network access or preparation for offline diagnosis.
- The default download budget for missing model and SAE files is 16 GiB. Preparation has a timeout of 1800 seconds, and data preparation has a separate 1800-second timeout. Actual checks still use `max_seconds`, `max_samples`, and `max_tokens`.
- By default, data preparation generates 32 candidates and requires at least 1 to be retained. This is the recorded minimum data preparation budget for a check. The Dataset's filters still apply, and this budget does not define a full evaluation dataset or establish research acceptance.
- Validate and reuse existing complete data. Data preparation uses a lock and commits from a temporary directory. Existing incomplete artifacts are preserved.
- Set the device with `runtime.device` and provide the actual check plan in `task.precheck.task_configs`. The secondary confirmation switches `allow_compute`, `allow_gpu`, and `allow_native` have been removed. An execution request proceeds after download without another manual confirmation.
- Native training checks retain actual budget limits, including a maximum of two steps. Missing input plans or unavailable hardware are reported with specific reasons.

## Records and Failures

Results are saved in the current Runtime run directory:

- `metadata/precheck_result.json`: Preparation stages and actual check results.
- `metadata/precheck_audit.json`: Report-file integrity check.
- `tmp/precheck_jobs/<stage>/metadata/`: Requests and subprocess receipts. The first automatic precheck for a production task is saved under `metadata/automatic_precheck/`. Retries are written to separate `attempts/` subdirectories. Each execution's manifest points to its own receipt. A failure to load a shared model or SAE also produces a receipt.
- `tmp/precheck_jobs/<stage>/logs/`: Complete standard output, standard error, and installation, import, and download events.

Start with the request scope, stage, and actual check target, then read the corresponding result and log. Use the concrete reasons and suggested actions to supply missing dependencies, resources, or inputs. Reuse completed work according to the owning task's contract. When developing checks and their records, also design how those records will be read and used after the run.

Failures during installation, source access, download, data preparation, or budget allocation record the specific BLOCKED reason. Software execution errors remain FAIL. A timeout stops the entire worker process group so that installation and native compute subprocesses do not remain in the background.

These records establish how far the software executed. Establishing model performance or independent research acceptance requires separate evidence. Offline tests use explicitly marked synthetic responses and models and do not substitute for running the real workflow.

## Performance Precheck and Automatic Tuning

With `task.precheck.mode=performance`, run a small performance trial on the real tasks selected by `task.precheck.check_tasks=[capture,intervene,generate]`. Capture checks `whiten`, `non_whiten`, `g_s`, and `I` by default. Narrow the scope with `+task.precheck.capture_modes=[g_s,I]`. Each performance precheck selects one model, one SAE, and one layer. Run multiple resource combinations separately.

```bash
python -m saebase.runtime.train model=gpt2small dataset=gsm8k \
  task=precheck task.precheck.mode=performance task.execute=true \
  'task.precheck.check_tasks=[capture,intervene,generate]' \
  '+task.precheck.capture_modes=[g_s,I]' experiment.sub_name=performance_probe
```

The actual `capture`, `intervene`, and `generate` tasks default to `task.performance.enabled=true`. The production Runtime's automatic precheck runs the relevant performance probe first and passes the recommended batch size to the task. Setting it to false retains scalar execution and disables performance tuning. runtime.numerics still determines the GPU numerical policy. Configuration inspection (`--cfg job` / `execute=false`) runs no probes. Direct Qwen3.5 commands need the dedicated interpreter in `conf/base/runtime/runtime.yaml`. The matrix entry point selects it automatically through Runtime.

`base/runtime/performance.py` owns the shared budget. The defaults are:

| Parameter (under task.performance) | Default | Purpose |
| --- | --- | --- |
| enabled | true (standard task configuration) | Enable automatic performance precheck and reporting |
| batch_size | auto | Select automatically or specify a positive integer |
| batch_sizes | [1,2,4,8,16,32,64] | Measure these candidates and include the actual unit-count limit |
| probe_samples | 3 | Questions in the performance probe. The production batch's Dataset size stays unchanged |
| probe_states | 64 | Control states for the Capture batch search. A real I probe still computes all 1024 states |
| warmup / repeats | 1 / 3 | Warm up and measure both the baseline and the candidates repeatedly |
| max_seconds | 120 | Soft time budget for each candidate search. In-progress forward passes finish |
| memory_fraction | 0.9 | Maximum ratio of peak process-reserved GPU memory to total device memory |
| atol / rtol | native: 1e-6 / 0, vllm: 0 / 0 | Numerical tolerances for mean answer log probability |
| logit_atol | native: 1e-4, vllm: 0 | Floating-point tolerance for intervention and generation metrics. Discrete outputs such as generated tokens, rankings, and strings must match exactly |

For budgets omitted from YAML, use a Hydra override of the form `+task.performance.<field>=...`. `task.precheck.task_configs` can provide the real task configuration, such as `generate: {generate: {intervention: ablate, value: 0, max_new_tokens: 8, reuse_prompt_observations: true}}` for generation with ablation. Compare with the same parameters as the task that will run.

Tuning compares **the same input and the same workload** and selects the highest measured throughput among candidates that pass numerical checks and leave GPU memory headroom. In automatic mode, keep the original configuration if the candidate is slower. An out-of-memory error is recorded and ends the search for larger batches. If it occurs during the actual run, reduce the batch size and perform the numerical checks. The result is the best measured option under the given budget and candidate set. Reuse parameters by model, task, and length band. Decisions with no net benefit are also cached. The normal Runtime automatic precheck hands the recommendations to the immediately following task and allows different batch sizes for different lengths. Per-question Capture checks and representative controls for feature tasks still run. These measured controls do not establish numerical equivalence for every possible input.

Main engineering optimizations:

- Cache the first portion of the model at TL residual pre/post sites and in the HF decoder. During HF cache preparation, stop at the selected layer to avoid unnecessary tail layers and full-vocabulary outputs.
- Batch actual Capture states, deduplicate identical native code states, reuse invariant decoding baselines, and vectorize ranking of the four-state interactions. Decoding still calls the native SAE row by row to preserve operation order and LN statistics.
- Reuse shared observations and identity checks during Intervene, replay the tail layers, and batch feature conditions.
- Share baseline generation and batch native feature conditions during Generate. Batch size 1 and larger batches use the same implementation. Each generation trace retains its own complete prefix and original random seed. Reuse excludes stale edited KV caches.
- Use inference_mode and retain only the necessary logits slices so views of cached vectors do not hold the entire sequence×vocab tensor in memory. Model and SAE reuse and checkpoint verification leases are shared within the same precheck scope.

`reports/performance_precheck.md` is the overall performance precheck report. The JSON records stage distributions, per-sample records, and throughput, memory use, error, and rejection reasons for every candidate. Each task records actual elapsed times in `reports/performance.json` and `performance.md`. Reports give the mean, standard deviation, minimum, maximum, and P50/P90/P95/P99. They distinguish amortized time per question, feature comparison, and answer state. Stages can nest, so their times cannot be added directly.

Measure gains using the post-warmup medians for the original baseline and candidate on the same probe. Include candidate cache preparation in throughput comparisons. Record search, actual execution of all I states, and shared preprocessing separately. Reports give per-question times with and without initial tuning, search overhead, and the estimated number of units to break even. Quantiles from a small number of probes are descriptive statistics and do not represent the full 500-question results or population confidence intervals. When there is no per-optimization ablation, report the combined gain without attributing the same benefit to more than one optimization.

See the [measured performance report](../../assets/history_experiment/maintenance_evidence/files/task_performance/performance_report.md) for real performance probes on the four main models, original sources, and comparison criteria.

Standard runs use `runtime.numerics=vllm` and apply the shared stable vLLM execution policy before loading the model. Scores and outputs are compared strictly, with a required difference of 0. A silent fallback to scalar execution does not count as a pass. The historical numerical path uses `runtime.numerics=native`. A worker process cannot mix the two policies. See [batch scoring invariance and acceleration](../../doc/code/documentation/source/development/batch-invariant-execution.md) for the design and provenance.

The normal Runtime entry point loads the model and SAE once. The automatic precheck and production task share them, and the task consumes the recommendation directly. Record the compute version in the configuration before deciding whether to resume. Recompute if the version changes. For a matching version, resume according to the task's completion strategy. The I performance precheck also reads all 1024 states from a saved question artifact, independently recomputes them using a different batch size, and requires an exact match for every state.

The standard settings use `runtime.cpu_threads=4` to control both Torch and BLAS threads and disable the tokenizer's additional parallelism. Intervene and Generate reuse observations by default. The normal Runtime, precheck, and matrix workers use the same settings. Actual values are saved with results and performance reports. Compute precision, stable attention, the existing compilation path, GPU memory headroom, and sequential scheduling of one model at a time follow earlier validation. See [engineering defaults](../../doc/code/documentation/source/development/batch-invariant-execution.md#%E5%B7%A5%E7%A8%8B%E9%BB%98%E8%AE%A4%E5%80%BC) for the specific choices.

Before measuring a candidate's GPU memory use, release unused allocator cache, then reset and measure the peak. The 90% GPU memory limit still includes the reserved memory actually needed for the current workload. This prevents leftover cache from an earlier question or larger batch from incorrectly rejecting all candidates. Candidate records include `reserved_before_cleanup_bytes` and `memory_measurement_policy` to identify the measurement policy. Numerical consistency requirements retain their original values.
