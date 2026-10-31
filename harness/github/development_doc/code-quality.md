# Code Quality and Coverage

SAEBase contains established research code. We are adopting quality checks
incrementally rather than introducing a repository-wide formatting change that
could obscure scientific or API modifications.

## Install and run

~~~bash
python -m pip install -e ".[dev,quality]"
ruff check saebase/seed.py saebase/data/artifact_layout.py saebase/data/prompt_schema.py saebase/benchmarks/registry.py
mypy --config-file pyproject.toml saebase/seed.py saebase/data/artifact_layout.py saebase/data/prompt_schema.py saebase/benchmarks/registry.py
python -m pytest -q
pre-commit install
pre-commit run --all-files
~~~

The quality dependencies are optional. They are not required to use SAEBase.

## What CI enforces

1. **Ruff lint** runs E4, E7, E9 and F checks on these initial adopted modules:
   `saebase/seed.py`, `saebase/data/artifact_layout.py`,
   `saebase/data/prompt_schema.py` and
   `saebase/benchmarks/registry.py`.
2. **Mypy** checks the same modules, requires annotations and follows imported
   modules silently; this avoids misrepresenting unadopted code as type-checked.
3. **New Python files** added in a PR must pass both Ruff lint and
   `ruff format --check`. The check uses the PR base commit; on
   pushes it uses the parent commit. Local pre-commit checks staged additions.
   Modified pre-existing files are not reformatted automatically.
4. **Coverage** is collected from the complete offline pytest suite on
   Python 3.11 using `--cov=saebase --cov-branch`. CI uploads
   `coverage.xml` and `coverage.json` as artifacts and prints
   the measured total. The default suite runs on Python 3.11 with two
   `pytest-xdist` workers. Python 3.12 does not run in routine PR CI.
5. **Coverage threshold** is deliberately not imposed until an observed
   baseline and a justified improvement target exist. No modules are omitted
   solely to make the percentage look better.

Ruff, mypy and measured coverage are not substitutes for real model and SAE
checkpoint acceptance, scientific protocol validation or reproducibility.

## Extending the quality baseline

- Select an actively maintained module and inspect existing diagnostics.
- Fix issues in a focused PR without changing experiment logic or public API.
- Add the module to the Ruff and mypy lists in CI, pre-commit and this guide.
- Re-run CPU tests, optional real-model acceptance and packaging as applicable.

Do not use a repository-wide `ruff format .` or blanket ignores. Write
narrowly scoped, documented suppressions only when necessary.

## Measure coverage locally

~~~bash
python -m pytest -q --cov=saebase --cov-branch \
  --cov-report=term --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json
~~~

Always compare reports generated with the same collection of tests and coverage
scope. The initial report is a baseline, not a certification of correctness.
