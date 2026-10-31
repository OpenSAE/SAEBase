# Contributor Guide

SAEBase welcomes reproducible tests, model/SAE compatibility work,
documentation and research infrastructure improvements. The best first
contribution solves one concrete problem with a reproducible check.

## Pick an issue

Browse [SAEBase Issues](https://github.com/OpenSAE/SAEBase/issues).
Use the repository's GitHub forms to report a bug, propose a feature
or request a model/SAE checkpoint integration.

For a first pull request, good entry points are:

| Area | Where to start | How to verify |
| --- | --- | --- |
| Documentation or tutorial | `harness/github/development_doc/`, `saebase/examples/README.md` | `python saebase/tools/check_documentation.py --check` |
| Beginner offline examples | `saebase/examples/quickstart/`, `tests/` | Run the example and its targeted test |
| Model/SAE metadata checks | `saebase/sae/` and `saebase/extensions/` | Test hook, dtype, shape and revision mismatch cases |
| Cache and experiment integrity | `saebase/store/`, `saebase/benchmarks/` | Use synthetic fixtures; assert corrupted identities are rejected |

Document new behavior with a focused issue. For sizeable API or scientific
protocol changes, discuss the proposed interface first. SAR/RSS paper
estimators stay in their respective source repositories.

## Create a working environment

~~~bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e ".[dev,quality]"
python saebase/examples/quickstart/first_experiment.py --dry-run
python saebase/tools/check_documentation.py --check
python -m pytest -q
~~~

The core/offline tests run without GPU downloads. Optional real model
testing uses the separately installed `analysis` dependencies.

Run the relevant narrower tests while editing, then the complete offline
suite before opening the PR. New Python files must satisfy Ruff lint and
formatting; already established modules are being migrated incrementally.
See [Code Quality](code-quality.md).

## Submit a focused pull request

1. State the problem, relevant issue and actual changes.
2. Include reproduction steps and a deterministic test or before/after
   documentation path.
3. Say exactly which checks you executed and which you could not execute.
4. Keep public API, persisted artifact schemas and research estimators
   compatible unless an intentional migration is separately reviewed.
5. Include exact upstream model/SAE sources, license and revision where used.
6. Link real checkpoint evidence only when those tests actually ran.

Use the [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md).
The maintainer's [review workflow](maintainer-workflow.md) defines how
changes and critical paths are reviewed. Code ownership is recorded in
[CODEOWNERS](../../../.github/CODEOWNERS); GitHub branch protection is
a distinct administrative setting, not something the file itself enforces.

## Documentation changes

Update existing task pages rather than copying explanations into several
READMEs. Keep the main [README](../../../README.md) and
[Chinese README](../../../README_cn.md) as concise capability and navigation
entry points. Add task-level details under `harness/github/development_doc/` and runnable
checks under `saebase/examples/`.

For changed local links, run:

~~~bash
python saebase/tools/check_documentation.py --check
~~~

The checker covers internal relative Markdown links. It does not
automatically validate external websites.

## What counts as verified

Registration is metadata. Synthetic execution verifies a software
contract. A real model/SAE pair claim requires exact model and checkpoint
provenance, site/hook compatibility and actual acceptance evidence.
Scientific conclusions need the original evaluation protocol and data.

[Contributing rules](../../../.github/CONTRIBUTING.md) ·
[CI Lanes](ci-maintenance.md) ·
[Security Policy](../../../.github/SECURITY.md)
