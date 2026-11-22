# Statement of work on the data set

## Purpose

Description of the role of data set realization, access and product.

## Scope

Read and develop as applicable to `datasets/`. Maintenance is carried out in accordance with the mandate of project root AGENTS.md.

- `datasets/<name>/` Save data generation, preparation, reading, Token coding and division, `datasets/plugins.py` Access to Unified Registry.
- The source code is imported through `saebase.datasets` Naming Space; `base/dataset_access/` maintains compatible import and retains the normal use of the Hugging Face `datasets` package for third parties.
- Configure the corresponding presets for `conf/datasets/catalog.yaml`. Natural data access to the `__init__.py` statement of preparation, selection and support tasks in the directory to which it belongs, which are repeated from public sources; complex preparatory logic is kept in the data set. Data products are recorded as actual sources, model identities, versions and classifications; Prompt data are organized according to the existing structure of `prompts.jsonl`.
- When developing the preparation, reading or data check function, the output is determined from the target Task process of physical access to the sample: the sample, division and source can be read by the task; the specific data source, the input location and the preparatory action that needs to be performed when the preparation fails. The data description and preparation results are maintained using the existing sample reading portal, and the target Task actual consumption is used for the inspection.
- Use `maintain-dataset` skills to add, modify or remove data sets and their options.
