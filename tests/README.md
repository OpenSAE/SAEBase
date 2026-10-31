# Product and integration tests

Keep new product tests under this top-level directory when a component does not already own a local suite. Prefer `tests/workplan/<id>/`, `tests/<dataset>/<model>/<task>/<sub_name>/` or `tests/experiments/<study>/`, with justified extensions.

Independent Harness governance regression remains in `harness/github/work_test/`; isolated Agent exams remain in `harness/tests/`. See [architecture rules](../harness/core/BOUNDARIES.en.md).
