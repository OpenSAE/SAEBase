# Operational environmental information

## Purpose

Description of the source of the in-house interpreter, reliance on and resource information to assist in the re-use of the experimental environment and positioning operating conditions.

## Scope

Used to understand the actual environment in the experimental records, or to collect and compare environmental information according to user requirements.

## Run independently

Select Python 3.11 or 3.12 from the project root directory to execute `python -m pip install -e .`. Select the model backend and the preparation weights needed to load the task. The data directory can be selected by the item `paths.data_dir` parameter, running the record-keeping current interpreter, relying version and source. [Quick Start](../code/translated/README.md#-quick start)]

After installation, the configuration can be synthesized from the project root directory using the following commands, and the synthetic data example can generate an evaluation plan. Both commands rely on basic installation.

```bash
saebase task=capture --cfg job
python scripts/examples/first_experiment.py --dry-run
```

The first should list the selected data set, model and catch parameters. The second should list the synthetic prompt group, the model and the thin self-codator combination, and the planned number of times. The equipment, the dependent version and the performance status of the real check point can be checked against the corresponding experiment at `metadata/manifest.json`. [The experimental report](../experiment.md)]

## Optional ResearchHarness access

After access, you can bind the engine Python and the data and call the original entrance to SAEBASE via `rh projects run`.

If you need to collect the machine and process environment separately, you can run `rh utils probe --output <snapshot_path>`. When you have two snapshots, you can use `rh utils compare <previous_snapshot> <new_snapshot>` to check for changes in the conditions of dependence and equipment.

When operational conditions are missing, the specific dependencies, equipment or interpreter and its effects are processed within the delegated authority; environmental facts have been established by the re-use of the responsible module for the follow-up mission.

## Project Dependency and Operating Conditions

Baseline installation depends on [project configuration](../../pyproject.toml). Model environment selection is available [model operating environment configuration](../../conf/base/runtime/runtime.yaml), as compared to the actual interpreter of the operating inventory and the dependent version.

The 2B, 4B, 9B and 27B formulations in Qwen3.5 use a special interpreter. The batch entrance selects the interpreter by model and checks `qwen3_5` support for Transformers. The single run needs to be started with an interpreter in the formulation, e.g. `/workspace/envs/saebase/qwen35/bin/python -m saebase.runtime.train model=qwen35_2b --cfg job`.
