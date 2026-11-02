# Code Tree: Implemented Interfaces and Structure

`harness/code/build.py` extracts real Python modules, classes, methods, and structural relationships from the AST. It writes `doc/code/code_tree.json`.

The presentation layer in `harness/code/presentation.py` provides readable, source-backed labels and separates individual datasets, models, and tasks. `harness/code/interfaces.py` extracts function parameters, types, defaults, return expressions, and source excerpts. Class cards also expose member attributes and every defined method.

```bash
python harness/code/build.py
python harness/code/build.py --check
python harness/visualization/export.py
python harness/code/diff.py --before /tmp/code_before.json --after doc/code/code_tree.json
```

Tree grouping is navigational. It does not create new Python APIs or substitute a generated explanation for the actual implementation.

## Domain-owned source and Code Tree

Historical work-plan numbers belong in `harness/github_work_plan`, not executable package names or Atlas functional domains. The original `saebase/wp12/` implementations now live in `saebase/interventions/`, `saebase/circuits/`, `saebase/benchmarks/`, `saebase/api/`, and `saebase/release/`. Atlas rebuilds its source-backed tree from current paths. Historical records retain their original identities and evidence.
