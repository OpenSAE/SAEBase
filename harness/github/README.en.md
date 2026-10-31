# GitHub engineering workspace

This folder groups [Work Plans](work_plan/README.md), [independent tests](work_test/), [development documentation](development_doc/README.md), and [assets](assets/). GitHub-native settings remain under `.github/`.

Before modifying files read the [Harness overview](../README.en.md), [repository architecture](../core/BOUNDARIES.en.md), [task contracts](../contracts/README.md), and the assigned Work Plan. Identify the existing owner first. Shared components must not depend on a single experiment.

Prefer `scripts/workplan/<id>.sh` or `scripts/<dataset>/<model>/<task>/<sub_name>/`, and equivalent `tests/` layouts. Study-based grouping is also acceptable. These layouts are flexible with documented ownership. A new root module, framework or changed ownership requires separate approval. Include new/moved paths, dependencies, tests and evidence in the PR.

## Work Plan standard

All new development plans follow the [authoring standard](work_plan/WRITING_STANDARD.en.md), including real dependency resolution, owned file paths, exact commands and independent acceptance.
