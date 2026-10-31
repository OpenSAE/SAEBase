# SAEBase Research Harness

**Current Harness baseline (WP21.2 finalized):** [WP21 closeout](github/work_plan/21_harness_phase_closeout_20261030.en.md). The WP18–WP20 tests and prior Contract registry remain available; WP21's real-workflow exams, maintenance and archived round01/round02 results are linked from [Harness tests](tests/README.md). Remaining GPU or product fixes follow separate review.


The Harness provides auditable navigation, scientific knowledge management, experiment provenance, environment records, and scoped agent collaboration. Research Lab, datasets, models, tasks, and the public SAE API remain part of the unified SAEBase repository.

| Module | Responsibility |
| --- | --- |
| `core/` | Project architecture, Agent Flow, and review permissions |
| `knowledge/` | Scientific questions, evidence, and Context links |
| `code/` | Python AST structure, interface descriptions, and diffs |
| `experiments/` | Configuration, provenance, and artifact validation |
| `environment/` | Machine resource observations and comparisons |
| `transfer/` | Authorized and verified file transfers |
| `visualization/` | Interactive Core, Knowledge, Code, Environment, and Context trees |
| `github/work_plan/` | Historical plans, research reviews, and deferred acceptance work |
| `github/development_doc/` | Developer documentation and public reference material |
| `github/work_test/` | Integration and regression tests |

## Code architecture

Datasets, models, and tasks are organized directly under their corresponding top-level directories. `saebase/registry.py` and `saebase/runner.py` are the shared component registration and dispatch interfaces; `saebase/runtime/` owns experiment configuration and artifacts.

Legacy public API imports remain compatible where required. Historical release tags and experiment evidence are not newly verified merely because code moves.

## Local verification

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q harness/github/work_test/test_harness_migration.py
python -m saebase.runtime.train --cfg job
python harness/core/validate.py
python harness/knowledge/validate.py
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py --check
```

Actual results are established by CI logs for the corresponding commit. Real GPU runs and external-data checks require separate approval and evidence.

## Agent permissions

See `.cursor/hooks.json`, `.cursor/policy.json`, and [AGENT_FLOW](core/AGENT_FLOW.en.md). Unreviewed proposals cannot modify protected governance state or count as scientific acceptance.

## GitHub agents and ownership

See [the GitHub workspace](github/README.en.md) for engineering plans and tests, and [architecture boundaries](core/BOUNDARIES.en.md) before editing. Scripts and tests prefer Work Plan or Dataset/Model/Task/Sub-name layouts with justified extensions.
