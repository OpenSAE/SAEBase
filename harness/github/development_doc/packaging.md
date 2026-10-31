# Packaging and Compatibility

SAEBase is an installable Python library with one primary public namespace,
`saebase`. The existing research model, configuration, and script names
remain available as compatibility imports.

## Supported installation

~~~bash
python -m pip install -e ".[analysis]"
python -m pip install -e ".[dev]"
python -m build
python saebase/tools/verify_distribution.py dist
python saebase/tools/check_installed_wheel.py dist
python saebase/tools/check_sdist_roundtrip.py dist
~~~

Python 3.11 and 3.12 are supported. The final two commands validate both the
installed wheel outside the checkout and rebuilding a new wheel from the sdist.
Neither of these commands downloads LLM or SAE checkpoint weights.

## Namespaces

| Interface | Current policy |
| --- | --- |
| `saebase.api`, `saebase.store`, `saebase.interventions` | Public research interfaces remain unchanged |
| `saebase.benchmarks` and `saebase.reliability` | Versioned experiments and result contracts remain unchanged |
| `saebase.cli` | Canonical location of all nine installed command entrypoints |
| `saebase.model_plugins` and `saebase.tasks.plugins` | Canonical built-in registration; each executes once |
| `main`, `models.plugins`, `tasks.plugins` | Backward-compatible imports and launch paths retained |
| `models.*`, `saebase.tools.*`, `tasks.*`, `conf.*` | Legacy implementation/configuration packages retained for compatibility |
| `saebase.runtime`, `saebase.datasets` | Unified runtime and prompt dataset modules are packaged into installed wheels without shadowing Hugging Face `datasets`. Historical `saebase` commands remain unchanged. |

Do not rename old configuration roots, cache schema IDs, CLI flags or existing
task/model registry identifiers. Further consolidation of `models`,
`scripts`, `tasks` and `conf` should only happen alongside
tests for all consumers. Moving directly to a complete `src/` layout in
one PR would make debugging failures and scientific protocol drift harder.

The exact **command names and flags** are unchanged; only the setuptools
entrypoint targets are now inside `saebase.cli`.

## What belongs in installed packages

The wheel and source archive contain Python implementation code, Hydra YAML
configs, compatibility shims and the small machine-readable configs required
by CLI or model registration. They do not redistribute model/SAE weights.

**Stage D GPU run evidence is not an install-time dependency.**
The approximately 200 immutable Stage D evidence files stay in the public
Git repository and the [dedicated checksum-verified Evidence Release](https://github.com/OpenSAE/SAEBase/releases/tag/stage-d-evidence-20261016).
They are excluded from newly built wheel/sdist distributions. The original
`v0.2.0` published distribution is not changed, and its historical
validation remains separate from current development-branch evidence.

For Stage D evidence verification, work from a Git checkout or download the
dedicated evidence release instead of assuming those files are bundled with pip.

## Distribution acceptance

The CI checks each new distribution for:

- the nine namespaced command entrypoints and old import compatibility;
- the required runtime modules and at least 100 Hydra YAML files;
- no Stage D evidence archive files, experiment caches or model weights in
  pip distributions;
- functional CLI invocations and imports after installing the wheel into a
  temporary location outside the repository;
- successful wheel rebuilding from the distributed sdist.

These are package integrity and compatibility checks, not model or benchmark
scientific validations.
