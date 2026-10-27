# SAEBase Research Harness

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
| `github_work_plan/` | Historical plans, research reviews, and deferred acceptance work |
| `github_development_doc/` | Developer documentation and public reference material |
| `github_work_test/` | Integration and regression tests |

## Code architecture

Datasets, models, and tasks are organized directly under their corresponding top-level directories. `saebase/registry.py` and `saebase/runner.py` are the shared component registration and dispatch interfaces; `saebase/runtime/` owns experiment configuration and artifacts.

Legacy public API imports remain compatible where required. Historical release tags and experiment evidence are not newly verified merely because code moves.

## Local verification

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q harness/github_work_test/test_harness_migration.py
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
