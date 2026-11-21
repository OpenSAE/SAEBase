# Install and Source Layout

## Purpose

Describes the current SAEBase distribution package's correspondence to the source code and the necessary checks for installation outside the warehouse.

## Scope

For the construction, installation, maintenance and allocation of Python packages; scientific missions are carried out according to their own resource conditions.

## Include

Root models, tsks and conf are installed as existing structures. Data achieves keeping at the root level dabasets, places the namespace of saebase. datasets, and the source code uses the same physics. Third party Hugging Face datasets remains independent.

The installation package contains the current scientific realization, configuration and necessary resources. It does not contain weights, directories of the operation of the machine or an archive of historical measurements. Web of Research Lab relies on an optional installation through Lab. The research core does not rely on ResearchHarness.

## Current Command

Provides seven commands: saebase, saebase-prepare-prompts, saebase-check-env, saebase-intervene, saebase-benchmark, saebase-audit, saebase-grand-truth. Multimodels read compared configurations with SAE by evaluating commands. Software releases are processed by the current independent release process.

## Scope of inspection

Installs the wheel generated outside the source directory to read models, data, task registration and configuration resources and to implement the clearly selected offline example. Sdist needs to be able to rebuild the same realization. Check and build to run according to the user-defined range; the existence of the source code does not mean that the installation or the scientific acceptance has passed.

The component presets `conf/<component>/catalog.yaml` to maintain the selection syntax by the name configuration register of Hydra; read `conf.load_group` directly. Cleans up the old construction output of the package before construction, so that the deleted or moved source code is accurately reflected in the new package. The distribution check confirms the actual resource and installation call, and the configuration is measured according to the current organizational pattern.
