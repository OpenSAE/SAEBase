# Task Configuration Named by File Duty

## Purpose

Records the configuration prefix migration, document synchronization and check results for the eight tasks of SAEBase.

## Scope

This applies to the maintenance of `conf/task/`, corresponding tasks, precheck and batch entrances, and to seeing how old commands move.

## Maintenance history

2026-11-19 UTC. Based on `conf/task/<task>.yaml` and `tasks/<task>/task.py`, a special field is moved to `task.<task>` to keep the commonly used shared field and capture existing subtasks. Harmonize the actual reading and pre-check assembly. When entering an old screenshot or old overlay in the program, read or convert to the current field for the original purpose. The new command uses the new prefix.

2026-11-19 UTC. Synchronizes the main document, task document, pre-check description and examples of current commands. Expands the quick check to check the current task parameter table, the formulation default and the specific parameter coverage. Validation also fixes only the configuration of the active health assessment error entry.

## Authentication Results

- The associated regression of 432 passes, 3 skips.
- The final review of 35 inspections of the configuration, precheck and assessment portals was approved.
- Eight tasks are run by the only configuration of a formal order, and the new prefix and visible overlay values on the list are consistent with the command.
- This is a scientific experiment that does not contain real models.

The current field and its usage is in the [SAEBase configuration parameter](../../../conf.md).
