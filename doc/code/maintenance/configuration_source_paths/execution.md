# Configure Path Synchronization with Document

## Purpose

Recording of the actual source path migration configuration, as well as configuration checks and documents synchronized into the submission and deployment process.

## Scope

Applies to SAEBase configuration, Hydra entrance, task reading, document and release package maintenance.

## Maintenance of records

2026-11-19 UTC. Run a configuration check on an old directory, 12 passes. The path review found that the prefix and the document were validated by the original test and that the configuration was still acceptable in a directory that did not match the source code.

2026-11-19 UTC. Move configurations to the corresponding source catalogues under `conf/datasets/`, `conf/models/`, `conf/tasks/` and `conf/saebase/` to move the package and call the entry. Receiving and inspection inputs are maintained at the point of inspection, and historical acceptance evidence is transferred to `assets/history_experiment/benchmark_evidence/files/`. Environments, formulations and Hydra selects the entry to be read under the original name.

2026-11-19 UTC. Submit the hook automatically synchronizes the configuration document, and then executes the path, key and document static check. The light CI, the release package build and the release certification perform the check. The representative returns to a total of 308 passes. The configuration check 15 passes, verifying that the wrong path will be rejected and the source configuration changes will update the body default and the production content. The official command validates the only configuration path for performance precheck and activation of the health assessment.

[Configure Document](../../../conf.md) · [Quick Check](../../../../tests/test_configuration.py)
