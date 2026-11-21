# SAEBase developer documentation

## Purpose and main functions

Find the instructions for installing, running or maintaining SAEBase, from the public entry point to the responsible files and results.

## Scope

Use this index to select an operating guide or reference for the current project. Recorded results apply to the run conditions in their reports.

## Install and run an example

- The project README covers installation and the public workflow. Its quick start shows how to inspect or run an example. [Quick Start](../../../../../README.md#-quick-start)
- The setup guide is for a clean Python environment. It shows both a small local experiment and a real Pythia checkpoint check. [Getting Started](getting-started.md)
- The first experiment guide explains the input, feature edit and report from a runnable local example. Use it to identify the HTML report, CSV summary and JSON results. [First Experiment](first-experiment.md)
- The troubleshooting guide is for setup or execution errors. Use it to identify where a dependency or model load failed. [Troubleshooting](troubleshooting.md)

## Use project interfaces

| Work | What the guide covers |
| --- | --- |
| Call SAEBase from Python | The guide shows how to pass a model, prompt and task to an existing interface. The reference describes public functions and their inputs and outputs. [Python API guide](python-api.md) · [API reference](API.md) |
| Select a model or SAE | The model guide explains available resources and loading conditions. The tested matrix shows the models and layers that have recorded compatibility and evaluation evidence. [Model support](model-support.md) · [Verified model and SAE matrix](verified-model-matrix.md) |
| Capture activations | The store guide explains how to save and reuse activations, including which input and model identity must agree. [Activation Store](activation-store.md) |
| Measure an intervention | The intervention guide explains how to specify a feature edit and compare the result with the original model output. [Feature Interventions](feature-interventions.md) |
| Benchmark and audit | The benchmark guide explains protocol inputs and metrics. The reliability guide explains grouped resampling, such as keeping related prompt variants together when estimating uncertainty. [SAE Benchmarking](benchmarking.md) · [Reliability](reliability.md) |
| Run controlled comparisons | Controlled input and localization guides explain input construction and where to measure model behavior, for example under a fixed prompt condition. [Controlled Inputs](controlled-inputs.md) · [Localization](localization-benchmark.md) |
| Use a research method or extension | The integration guide describes supported research workflows. The extensions guide covers installed providers, such as a new model or task implementation. [Research Integrations](research-integrations.md) · [Extensions](extensions.md) |
| Explore recorded measurements | The Research Lab guide explains how to use its browser interface to inspect saved results. [Research Lab](../../../../../base/research_lab/README.md) |

## Maintain and distribute the project

The repository guide maps existing directories to their purpose. Use it to find the source, test or script that owns a change. The release process explains how to prepare and check a distribution. [Repository Guide](repository-guide.md) · [Release Process](release-process.md)
