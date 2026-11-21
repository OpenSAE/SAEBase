# Model job description

## Purpose

Description of responsibilities for model realization, registration and SAE resources.

## Scope

Read and develop as applicable to `models/`. Maintenance is carried out in accordance with the mandate of project root AGENTS.md.

- `models/<family>/<series>/specs.py` Save model identity and load statement, public adapter constructs. `models/plugins.py` takes the statement from the specified series and accesss the Unified Registry; complements the statement directly with the new model in the series.
- Public build and delay loading in a common module at `models/`; model presets `conf/models/catalog.yaml`; public default values are saved once, with each name only stating its own differences.
- SAE is the configured resource of the model, `model.default_saes` declares the default SAE; `base/interfaces/sae_profiles.py` is responsible for its source, model, layer, activation position and checkpoint correspondence.
- Configure and plugin testing uses light-weighted methods, real weight downloads and large models load as specified by the user's mandate.
- When developing loads, compatible checks or capability queries, you will first specify how the target task will select and use models and SAEs. Returns the actual available capacity and matching resources; identifies the selected model, SAE, layer or dependent specific problems and treatments if you lack the conditions. Record keeping is sufficient to re-select or continue loading the identity and basis, using the results directly by the caller.
- Use `maintain-model` skills to add, modify or remove models and their options.
