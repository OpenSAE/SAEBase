# Product and Integrated Testing

## Purpose

Description of the attribution, selection and use of the actual results of the test and assistance to the maintainer in identifying the affected function.

## Scope

To add, maintain or select the SAEBASE software test; check the scope with root AGENTS.md and current task requirements.

The `tests/product/` save product returns, and other special tests are organized around the corresponding function. Research Lab keeps its own local test, `tests/integration/` to check the project's interface with the centre and the input of the project's statement.

If you add, change or move the task configuration, you can run `python -m pytest -q tests/test_configuration.py` separately. Check the attribution of the location parameters from `conf/tasks/<task>/task.yaml` and `tasks/<task>/task.py`, indicating the missing corresponding files, inconsistent job identities and misfixed prefixed keys. The task-specific configuration uses `task.<task>.<parameter>` and captures the individually achieved SAR return, quick interaction, RSS and model original responses using the respective prefix under `task.capture`. The check also checks the grouping and realization of the files. The shared distribution, resources and operational fields are covered by the shared entry. The check also checks the full text of the field, default values and mission-specific parameters against the current parameter list in `doc/conf.md`.

The configuration file sets the path to the front two levels of the responsible source relative directory, with one file for each branch. Deeper duties are placed in the name section of the document. For example, the `tasks/capture/` configuration is placed at `conf/tasks/capture/`, the main configuration entry is `conf/config.yaml`. Run `python -m scripts.documentation conf --write` when new, modified, moved or deleted. It rebuilds the navigation and current values from the configuration directory and synchronises the default values that can be read directly from the formulation in the text table.

Submits an automatic synchronisation of the hook, and then runs `tests/test_configuration.py`. Lightweight CI, build release package and release process perform the same check, rejecting a routing error or unsynchronised configuration of the document. When you add a separate configuration, register the responsible source at `tests/test_configuration.py` when you confirm the responsible source from the normal reading access.

Prefer to re-entry and downstream use of selected scenarios. The preservation function also checks re-reading and product positioning, and the actual associated failure or partial completion of the inspection cause, impact and continuation method.

The central management test of ResearchHarness is kept at the centre `src/agent/tests/` and selected by the corresponding maintenance task.
