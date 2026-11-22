# SAEBase contributor and agent guidance

## Purpose

This guide explains how to develop SAEBase and verify changes through its public research workflows.

## Scope

Use this guide when maintaining SAEBase source code, configuration, documentation or tests. Start with the README and use the entry point that researchers normally call.

## Develop and verify changes

- Locate the user-facing command, the component that owns the behavior and the reader of its output. Use focused regression tests and representative inputs.
- Keep model and SAE identities, data splits, experiment provenance, metrics and numerical settings explicit. Use synthetic fixtures for offline tests and obtain task-specific approval for model downloads, training, data transfer and publishing.
- Organize shared runtime and public Python interfaces under `base/`. Put dataset readers in `datasets/`, model adapters in `models/` and independent research tasks in `tasks/`. Ownership-matched configuration lives under `conf/`.
- Run examples from `scripts/examples/`. Keep synthetic fixtures under `tests/fixtures/` and software tests under `tests/`. Update relevant documentation when a public command, parameter, dependency or result format changes.
- Run `python -m scripts.documentation conf --write` after changes to public configuration. Run `python -m pytest -q tests/test_configuration.py` to verify the reference.
- Use Ruff 0.15.22 and the repository's `scripts/quality.py` workflow. Keep review scope limited to the changes under discussion.

## Local maintenance

SAEBase installs and runs independently on Python 3.11 or 3.12. ResearchHarness is optional and provides shared maintenance guidance and resource bindings where configured. Use the README and [project documentation](doc/README.md) for standalone development. Save research runs under `experiment/` and use their manifests to retrieve configurations and results.
