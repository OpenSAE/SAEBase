# Repository Guide

## Purpose and Main Functions

Shows where to find source code, configurations, examples, reference material, and maintenance entry points.

## Scope

Use this guide to locate components and choose the relevant usage or development documentation.

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`scripts/examples/`](../../../../../scripts/examples) | Two complete runnable demonstrations | [First experiment](first-experiment.md) · [External components](extensions.md) |
| [`tests/fixtures/`](../../../../../tests/fixtures) | Synthetic components shared by offline checks and demonstrations | [Testing and development](../../../../../.github/CONTRIBUTING.md) |
| [`saebase/`](../../../../../base) | Python APIs for loading, measuring, editing and reporting | [API guide](python-api.md) |
| [`models/`](../../../../../models) and [`conf/`](../../../../../conf) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`saebase/research_lab/`](../../../../../base/research_lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../../../../../base/research_lab/README.md) |
| [`doc/code/documentation/source/assets/`](../assets/) | Branding and workflow figures used in documentation | [Workflow diagram](../assets/figures/saebase_workflow.svg) |
| [`doc/code/documentation/source/development/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`scripts/`](../../../../../.github/CONTRIBUTING.md) | Entry points for software release, distribution, documentation, and code checks | [Script guide](../../../../../.github/CONTRIBUTING.md) |
| [Project maintenance entry point](../../../../../AGENTS.md) | Select online or offline maintenance and locate shared skills and local resources | [Offline manual](../../../../../.researchharness/offline/operations.md) |

## Reproducibility

Current experiments bind the model, dataset, task, source commit, configuration and recorded outputs to a single run identity. Use the declared configurations and task APIs rather than an older engineering plan as execution authority.

## Select development checks

Choose examples and tests for the changed behavior using [Code Quality](code-quality.md). Documentation changes use content review and affected links; full offline and package checks serve release preparation or a task with repository-wide impact.

New users can begin with [Quick Start](../../../../../README.md#-quick-start), then choose a task from the [Examples Index](first-experiment.md).
