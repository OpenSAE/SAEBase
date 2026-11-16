# Repository Guide

## 目的和主要功能

说明源码、配置、例子、资料和维护入口的实际位置。

## 适用范围

用于定位组件和选择相应使用或开发说明。

SAEBase separates runnable examples, research code, published evidence and engineering utilities so you can start with a task rather than a development milestone.

## Find what you need

| Location | Purpose | Start here |
| --- | --- | --- |
| [`scripts/examples/`](../../../scripts/examples) | 两个完整可运行演示 | [首次实验](first-experiment.md) · [外部组件](extensions.md) |
| [`tests/fixtures/`](../../../tests/fixtures) | 离线检查与演示共用的合成组件 | [测试与开发](../../../.github/CONTRIBUTING.md) |
| [`saebase/`](../../../saebase) | Python APIs for loading, measuring, editing and reporting | [API guide](python-api.md) |
| [`models/`](../../../models) and [`conf/`](../../../conf) | Model adapters, SAE profiles and experiment configurations | [Model support](model-support.md) |
| [`saebase/research_lab/`](../../../saebase/research_lab/README.md) | Browser-based exploration of recorded measurements | [Research Lab](../../../saebase/research_lab/README.md) |
| [`doc/code/assets/`](../assets/) | Branding and workflow figures used in documentation | [Workflow diagram](../assets/figures/saebase_workflow.svg) |
| [`doc/code/development/`](index.md) | Tutorials, API details, scientific protocols and verified reports | [Documentation](index.md) |
| [`scripts/`](../../../.github/CONTRIBUTING.md) | 软件发布、分发、文档和代码检查入口 | [Script guide](../../../.github/CONTRIBUTING.md) |
| [中央接入](../../../.agents/skills/researchharness/SKILL.md) | 用户选择的中央管理与正式计划 | [工作模式](development-modes.md) |

## Reproducibility

Current experiments bind the model, dataset, task, source commit, configuration and recorded outputs to a single run identity. Use the declared configurations and task APIs rather than an older engineering plan as execution authority.

## Select development checks

Choose examples and tests for the changed behavior using [Code Quality](code-quality.md). Documentation changes use content review and affected links; full offline and package checks serve release preparation or a task with repository-wide impact.

New users can begin with [Quick Start](../../../README.md#-quick-start), then choose a task from the [Examples Index](../../../doc/code/development/first-experiment.md).
