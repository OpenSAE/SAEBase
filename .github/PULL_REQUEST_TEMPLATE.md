## Summary

What capability or bug does this change address?

## Validation

- [ ] Offline regression tests passed
- [ ] Public API compatibility checked
- [ ] Model/SAE/hook provenance documented where relevant
- [ ] Checkpoint-dependent or GPU-only verification status stated
- [ ] Third-party source attribution reviewed

## Code quality

- [ ] New Python modules pass Ruff lint and formatting; changed adopted modules pass type checks.
- [ ] No unrelated formatting or changes to scientific metrics and cached schema behavior.

See [Code Quality](harness/github/development_doc/code-quality.md) for local commands. Unmodified legacy modules are not required to pass repository-wide lint.

## Architecture and file ownership

- [ ] All newly added and renamed files have a clearly documented owner.
- [ ] Existing reusable implementations and registries were considered before adding a component.
- [ ] Scripts/tests follow Work Plan or experiment-oriented grouping, or the exception is justified.
- [ ] Shared components do not depend on a particular experiment.
- [ ] Configuration, packaging, docs, links and CI references were checked.

## Work Plan / Agent execution readiness

- [ ] Files are grouped by existing component ownership, without a new study framework.
- [ ] Exact branch fetch and checkout works with a single-branch origin refspec.
- [ ] Dependency resolver, transitive version pins, pip check and minimal smoke were verified.
- [ ] Generated metadata and virtualenv artifacts are excluded from source changes.
- [ ] A small pilot, resource budget, failure receipt and independent acceptance gate are defined.
