# Code quality and testing

## Purpose

Description of the method used for local formats, type checks and test results.

## Scope

For Python development, inspection and test results reports.

## Local development

Install as required `python -m pip install -e ".[dev,quality]"`.Checks the current range of documents and functions involved. When the functionality is developed, the actual use of the selected results is selected and then selected or supplemented on the basis of which the results are obtained through the public entry point, preserved and reread, and the information in it is used to complete the follow-up. Failures and partial completions confirm that reasons, impacts and disposal recommendations can be used as they are needed and are prioritized for re-use for testing..

`pre-commit install` Enables the submission check. Modules that have been included in the type check are only checked when they change; new Python files are selected and checked by `scripts/quality.py` against the temporary storage range. Ruff uses `0.15.22`.

## Tools and scope of inspection

Ruff and mypy currently cover `base/application/seed.py`, `datasets/prompt_collection/artifacts.py`, `datasets/prompt_collection/artifacts.py` and `tasks/benchmark/metrics/registry.py`. When these documents are modified, the inspection is performed and the other documents are processed on the basis of specific changes.

The new document check will format the selected new document and then perform the format and grammar check. The format changes of the existing document are made to the current task range.

## Project-level testing

Full offline testing, coverage and packing checks are used to publish tasks that are prepared or clearly require a full-scale inspection.

The results of software tools, the operation of real models and the scientific findings retain their practical basis for use in the corresponding development and scientific missions.
