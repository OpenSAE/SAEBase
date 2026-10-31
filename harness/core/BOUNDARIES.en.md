# Core: Implementation Boundaries

**Core** defines project architecture and management goals in `harness/core/project_graph.json`. It is separate from scientific theory or experimental evidence.

**Knowledge** records research questions, methods, and source evidence in `doc/knowledge/`; implemented code cannot itself establish a scientific result.

**Code** describes real Python entities extracted from the AST. Generated code nodes reference Core design and scientific Context by stable identifiers and must not overwrite source code.

**Environment** relies on measurements made on the execution machine. Synthetic examples remain clearly labeled.

**Experiments** are governed by `saebase/runtime/experiment_paths.py` and Hydra configuration in `conf/`. Harness interprets and validates output but must not become a training-loop dependency.

**Transfer** requires explicit source, destination, and authorization; machine credentials must not be committed.

**Visualization** provides read-only public views of the Atlas trees, with separately authorized local review functions. Publishing does not reveal private machine state.

For each work order, select only the relevant source nodes and paths. Compare expected Code Tree changes with the AST after implementation. Verify the resulting model, checkpoint, configuration, and experiment independently before marking a scientific goal complete.

Static views are not evidence that real training succeeded. Reused validation/test data does not constitute an independent test. Browser-level or Cursor-level restrictions are not an operating-system security boundary.

## Repository architecture and file ownership

Always identify the existing owner before editing. File edit permission and architectural ownership are independent requirements. Prefer extending established modules.

| Owner | Responsibility |
| --- | --- |
| `saebase/` | Shared API, Registry, Runtime and reusable mechanisms |
| `datasets/<Dataset>/` and `conf/dataset/` | Source-specific adapters and configuration |
| `models/<Model>/` and `conf/model/` | Model adapters and configuration |
| `tasks/<Task>/` and `conf/task/` | Reusable task operations and configuration |
| `experiments/<Study>/` and `conf/experiment/` | Study-specific orchestration and configuration |
| `scripts/` | Launch scripts and operational entrypoints |
| `tests/` | Product and integration tests |
| `experiment/` | Runtime-owned results |
| `harness/` | Governance, evidence and independent review |
| `.github/` | GitHub-native workflows, distinct from `harness/github/` |

A study is an instance, not a framework. Shared Dataset / Model / Task / Runtime modules must not import a particular study. Separate implementation, configuration, entrypoints and generated outputs.

Scripts and tests prefer either Work Plan grouping (`scripts/workplan/<id>.sh`, `tests/workplan/<id>/`) or experimental identity (`scripts/<dataset>/<model>/<task>/<sub_name>/` and the matching test path). Study-based `scripts/experiments/<study>/` is also valid. These are flexible conventions; an alternative needs documented ownership and purpose.

New root modules, independent registries, cross-component frameworks or dependency reversals require separate architecture approval. Every PR must document new/moved paths, existing alternatives, ownership, tests and evidence. Do not rewrite historical acceptance evidence merely to rename directories.


**Root guardrail:** Do not create a top-level `experiments/` study code package. Keep per-study glue under `conf/experiment/<study>/` and `scripts/experiments/<study>/`, tests under an owned `tests/` structure, and generated results under singular `experiment/`. Other new roots require separate architecture approval.
