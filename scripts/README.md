# Engineering maintenance and batch entrances

## Purpose

Description of how the software is used for its distribution, distribution, documentation, code checking and specific batch start-up.

## Scope

For software engineering maintenance and batch start-up as listed on this page; scientific methods and scientific results will be judged by the parent Task.

The engineering operation uses four portals to share their respective reading, processing and reporting procedures:

| File | Main Operations |
| --- | --- |
| `release.py` | Software version, source, conditions for issuance and asset register |
| `distribution.py` | `contents` Check package contents, `install` Check outside the warehouse installation, `roundtrip` Rebuild and inspect from the source package |
| `documentation.py` | `--check` Check Document Link.`conf --check` or `conf --write` Check the relative path responsible for the source code and update the path, current value and readable default values in the parameter document.`support --check` or `support --write` Check or update supporting statements based on assessed evidence. |
| `quality.py` | Format and static check of selected codes; `select --base <commit> --head <commit>` Select to continue integration check |

The scientific parameters are converted to the mission command under which the data are prepared using `saebase-prepare-prompts`. For the full use of the project development, see [Contribution Guide](../.github/CONTRIBUTING.md)].

Normal matrix, whiten and SAR queues share `base/runtime/environment.py` to start the model subprocess.
Current and Key Dependency Versions Harmonized Configuration at `conf/base/runtime/runtime.yaml`; Default use
`/venv/main/bin/python`, Qwen3.5 with `/workspace/envs/saebase/qwen35/bin/python`.
Runtime checks the version-dependent and the required model structure before weighting, and maintains a clear record of failure when the check is not passed.
The corresponding environment is also selected when using the `--model` parameter of the matrix directly.

The main process stays in the same environment and continues to run other models after the subprocess has ended using default environments.
Clears the process group that is created when time is exceeded and when you receive the stop signal. Saves it next to each model log
`run-<model>.environment.json`, containing interpreter, dependent version and actual package location; per experiment
The `metadata/manifest.json` also records the actual operating environment.
`/workspace/envs/saebase/`, the catalogue keeps a record of measurements and operations.

The new Capture batch uses formal capacity:

```bash
python -m tasks.capture.matrix --config conf/tasks/capture/task.yaml --capture-mode whiten --tag my_whiten_batch
```

Batch configurations are shown only for this model/SAE, data set selection and execution budget. Each group is still running with the public Runtime by SAEBase `conf/config.yaml`, automatically precheck, and then capture. Batching is reloaded with each model's loaded session and keeps a record of failed combinations and possible continuation.

Batch movements are maintained by `tasks.capture.matrix`. Configure selection models, SAEs, data sets and budgets, and the combinations are executed through official Runtime and Precheck.


## GPU free run Capture matrix

`capture_queue.py` Call formal. `tasks.capture.matrix`,Reads the specified matrix configuration Dataset and Model/SAE/Layer selection. Waiting for the external calculation process to leave before each model is started GPU,Visible occupancy does not exceed 1024 MiB,No more than utilization 5%,And keep it at least continuously. 300 seconds(5 min; command line does not accept below 300 Threshold;`--wait-pid` You can also wait to specify that you will exit in batches to avoid the model switching gap to the graphic card..

```bash
python -u scripts/capture_queue.py \
  --sar-source /absolute/path/to/SAR/sar.py \
  --report-dir /absolute/path/to/matrix-report \
  --tag sar_answer_matrix \
  --readout answer_mean_logprob
```

Default selection of 16 features, 384 subsets, of which 256 are proposed and 128 are set aside; Dataset still uses its own fixed sample selection. `answer_mean_logprob` Token do all standard answers toteacher forcing, calculation of average conditions log probability. The selection uses the correct option text, GSM8K uses the final answer, HumanEval uses the original standard code to complete it; entry and feature selection do not add to the answer. The record with no standard answer (including the current HarmBench, XST) is unsuppleted and cannot be used as a successful measure.

Run the product to save to `experiment/<Data Set>/<Model>/capture/<Test Name>/`.The name of the experiment is different from the capture mode, and an actual mode catalogue is added at the end.,`sar_answer_matrix` It's... SAR Member writing `experiment/<Data Set>/<Model>/capture/sar_answer_matrix/sar/`.Feeds are saved in the effective configuration. Report directory saves `effective_matrix.yaml`,`launch_history.jsonl`,`status.json`,Model logs and group-by-group completion records. Read group-by-group completions and reasons when viewing batches, then judge re-useable content using the official results of the corresponding operation; logs are used to locate actual errors. Maintaining the schedule and reporting function to ensure that these records support the continuation of batches and find the desired product. Restarting the same configuration will read completed records; failure grouping retains old directories and create new attempts., SAR We need a complete follow-up network. We need to shut down normal. capture . The default does not set the total time frame for a single model; available `--model-timeout-seconds` Set positive numbers to terminate process groups created by scripts only over time and continue with follow-up models.

`--check-config` Checks only all combinations without load weight, without executing the original estimater or waiting for GPU. For tmux, directs the standard output and error re-direction to the log directory running at the corresponding batch; views progress with `tmux attach -t <session_name>`.

`capture_queue.py --capture-mode sar` Use the original SAR source code; `--capture-mode whiten` and `--capture-mode non_whiten` to use the appropriate capture method. The number of combinations comes from configurations, and queues share the official matrix schedule.


## Runable Presentation

`scripts/examples/first_experiment.py` Assessment, statistics and reporting using synthetic models, supported `--dry-run` and `--output`.`scripts/examples/external_providers.py` Show Model, SAE and the registration and actual consumption of the indicator. `tests/fixtures/synthetic.py` Provision, General Call.[The first experimental course.](../doc/code/documentation/source/development/first-experiment.md).
