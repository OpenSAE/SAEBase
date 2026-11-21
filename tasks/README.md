# Mission statement

## Purpose

Description of the fulfilment, distribution and co-support responsibilities of the scientific mission.

## Scope

Read and develop as applicable to `tasks/`. Maintenance is carried out in accordance with the mandate of project root AGENTS.md.

- Each mission is responsible for its own algorithms, parameters, indicators, sources and product protocols; the configuration path follows the task source code and is placed at `conf/tasks/<task>/`.
- The task is to provide specific unfinished objects, known causes, impacts and suggestions for processing, and to maintain and access the information together. The check process covers actual result consumption and is based on the mission's need to verify that the operation continues, completes the missing steps or uses the results.
- `tasks/plugins.py` Access to Unified Registry, `base/application/runner.py` for distribution of tasks.
- `base/runtime/sae_operations.py` provides co-support for models, SAE and Prompt resources; common operating paths and records will be `base/runtime/`.
- Activate the health assessment of benchmark, which follows the existing EvaluationResult outer layer structure.
- Parameter checks are performed in the responsible interface, and the configuration checks are kept light; real GPU operations are performed according to the user's mandate.
- Use `maintain-task` skills to add, modify or remove tasks and their options.
