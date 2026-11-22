# Repository Guide

## Purpose

Description of source code, configuration, examples, information and physical location of the maintenance portal.

## Scope

For positioning components and selecting the corresponding use or development instructions.

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`scripts/examples/`](../../../../../scripts/examples) | Two full runable presentations | [First experiment.](first-experiment.md) · [External component](extensions.md) |
| [`tests/fixtures/`](../../../../../tests/fixtures) | Synthesis components shared with offline inspection and demonstration | [Testing and development](../../../../../.github/CONTRIBUTING.md) |
| [`saebase/`](../../../../../base) | Python APIs for loading, measuring, editing and reporting | [API guide](python-api.md) |
| [`models/`](../../../../../models) and [`conf/`](../../../../../conf) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`saebase/research_lab/`](../../../../../base/research_lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../../../../../base/research_lab/README.md) |
| [`doc/code/documentation/source/assets/`](../assets/) | Branding and workflow figures used in documentation | [Workflow diagram](../assets/figures/saebase_workflow.svg) |
| [`doc/code/documentation/source/development/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`scripts/`](../../../../../.github/CONTRIBUTING.md) | Software release, distribution, document and code check access | [Script guide](../../../../../.github/CONTRIBUTING.md) |
| [Project maintenance portal](../../../../../AGENTS.md) | Choose to maintain online or offline, locate shared skills and in-house resources | [Offline manual](../../../../../AGENTS.md) |

## Reproducibility

Current experiments bind the model, dataset, task, source commit, configuration and recorded outputs to a single run identity. Use the declared configurations and task APIs rather than an older engineering plan as execution authority.

## Select development checks

Choose examples and tests for the changed behavior using [Code Quality](code-quality.md). Documentation changes use content review and affected links; full offline and package checks serve release preparation or a task with repository-wide impact.

New users can begin with [Quick Start](../../../../../README.md#-quick-start), then choose a task from the [Examples Index](first-experiment.md).
