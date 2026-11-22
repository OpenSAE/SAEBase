# SAEBase Documentation

## Purpose and Main Functions

Find operating instructions, parameters, research evidence, and run records.

## Scope

Use this guide to read and maintain the project. For installation and standard commands, see the [project README](../README.md). It describes the current features, commands, and workflows available from the project entry point.

## Directories and Documentation

| Top-level directory | Purpose and documentation |
| --- | --- |
| [`.agents/`](agents.md) | Project workflow and integration settings for collaborative maintenance, including how assistants obtain shared skills. |
| [`.github/`](agents.md) | Repository workflows and collaboration settings. |
| [`.researchharness/`](agents.md) | Project integration materials. |
| [`assets/`](../assets/history_experiment/sar_density/README.md) | Original materials and analyses from past experiments, including native activation density and threshold comparisons for SAR. |
| [`conf/`](conf.md) | Configuration files, defaults, and override rules, including how to choose a model's execution device. |
| [Workspace data](datasets.md) | Data and split selection, including how IOI questions are selected. |
| [`datasets/`](datasets.md) | Datasets and inputs. |
| [`doc/`](README.md) | Project documentation index. |
| [`experiment/`](experiment.md) | Comparisons and limitations based on actual runs, including the resources used by the BBH capture precheck. |
| [`models/`](models.md) | Model and weight selection, including the default SAE and analysis layer for Llama 3.1 8B. |
| [`base/`](base.md) | Shared runtime, environment, and result input and output. |
| [`scripts/`](../scripts/README.md) | Runnable engineering tools, including scripts for data preparation and export. |
| [`tasks/`](tasks.md) | Task selection and outputs, including how interventions select candidates from capture results. |
| [`tests/`](../tests/README.md) | Checks for actual behavior, including regression coverage of official entry points. |
| [`tmp/`](experiment.md) | Temporary local files. See the experiment reports for official runs and their results. |

[Research concepts and terminology](knowledge/README.md) (terms and research concepts used in this documentation) · [Runtime environment](environment/README.md) · [Source structure](code/README.md) (official entry points and responsible modules)

The [project maintenance index](maintenance.md) organizes execution records, readable recipes, and historical originals by topic.

The respective tools manage Git metadata, installation-generated files, and tool caches. The published repository retains documents in their original locations. Use the corresponding entries to locate the original run records in local experiment directories.
