# SAEBase

**Shared infrastructure for reproducible sparse autoencoder experiments.**

SAEBase provides a model registry, SAE checkpoint profiles, structured prompt
artifacts, activation capture, and SAE encoding/reconstruction diagnostics.
The Phase B API adds backend-aware activation reads, feature interventions,
SAE provenance, and explicit token positions. The original **SAE activation-health**
workflow remains available, starting with Pythia and a public SAE.

## Install and quickstart

SAEBase 0.1.0 is a standalone research package. Python 3.11–3.12 is the
validated target for the standard model/SAE stack. Install from this source
checkout (checkpoints and datasets download only when requested):

```bash
python -m pip install -e ".[analysis]"
saebase --list
saebase-check-env --mode analysis
```

The lighter `pip install -e .` provides the offline public API, registries,
configuration and CLI. `pip install -e ".[dev]"` adds packaging and testing
tools, and `pip install -e ".[generation]"` adds the experimental GPU
backends. Their hardware and package compatibility require separate checks.

From any working directory, `saebase --list` and
`saebase-prepare-prompts --help` work using the installed package. The
commands below also work from this repository root.

**1. Prepare deterministic SimpleCopying prompts** (downloads public model weights):

```bash
saebase-prepare-prompts \
  --model-key pythia70m --behavior SimpleCopying \
  --device cpu --n-generate 300 --min-kept 20 --seed 0 \
  --data-dir ./data
```

The step filters prompts with Pythia logits and writes:

```text
data/SimpleCopying/pythia70m/v1/
  prompts.jsonl
  manifest.json
```

**2. Evaluate SAE activations** (downloads the published SAE checkpoint):

```bash
saebase \
  --task sae --model pythia70m --sae pythia70m_resid_sm \
  --prompt-source SimpleCopying --layer 3 --n-prompts 5 \
  --data-dir ./data --device cpu \
  --run-dir ./experiment/pythia70m-smoke
```

The run saves `sae_activation_summary.json` and Hydra configuration files
inside `experiment/pythia70m-smoke`. A failed prompt is recorded with an
explicit error; successful rows contain token count, L0, activation norm,
reconstruction MSE, and fraction of variance explained.

For a combined acceptance run, use `bash scripts/phase_a_smoke.sh` (or
`DEVICE=cuda bash scripts/phase_a_smoke.sh`). This verifies the artifacts and
actual activation-health output after downloading the published checkpoints.

CUDA users can change both commands to `--device cuda`.
Model/SAE downloads require access to their respective published repositories.
The two commands use the same explicitly selected prompt artifact directory.

### Alternative: inspect the registries

```bash
python -m saebase.sae.verify_registry --list
python main.py --list
```

## Public Python API (Phase B)

Run the full **offline demonstration**, using a small deterministic model and SAE:

```bash
python examples/05_public_api_offline.py
```

For a real published Pythia model and SAE (downloads checkpoints):

```bash
python examples/06_public_api_pythia.py
```

Both entrypoints use `saebase.api`. The API separates model capabilities,
activation locations, SAE checkpoint identity, feature-code editing, and
baseline-versus-patched logit readouts. Identity replay checks that patching the
original activation reproduces the unmodified model output.

The built-in prompt package is now `saebase.data`; Hugging Face retains the
`datasets` package name. Existing Phase A CLI commands and artifact layouts are
maintained. See [Phase B API and migration guide](docs/PHASE_B.md).

## Research integrations (Phase C)

SAR and RSS use the shared capture/patch API while retaining their **original**
estimators in separate paper repositories. Two small CPU-only examples:

```bash
python examples/07_sar_via_saebase.py --paper-dir /path/to/SAR-repo
python examples/08_rss_via_saebase.py --paper-dir /path/to/RSS-repo
```

These examples use deterministic synthetic models by default; the `--real`
option downloads a public Pythia/SAE checkpoint. An additional registered
Gemma-3 270M/SAE pairing has a separate opt-in acceptance script:

```bash
python examples/09_second_model_smoke.py --model gemma270m
```

[Phase C integration, reference checks and backend status](docs/PHASE_C.md).

## Design

```text
conf/       Hydra configuration for model, SAE, prompt source, and evaluation
models/     Model registration and loader wrappers
saebase/data/  Prompt builders, compatibility matrix, JSONL and manifest I/O
saebase/   Public API, model/SAE loading, task runner, activation diagnostics, results
tasks/     Registered implementations shipped with this package (sae)
scripts/   Prompt preparation and environment checks
tests/     Offline contracts and small model-independent integration tests
```

The `saebase/sae/registry.py` SAE profiles specify checkpoint identifiers,
allowed models, layer and hook locations, and loader types. Registry entries
are configurations; successful weight loading is evaluated separately.

## Supported and archived functionality

| Capability | Current status |
| --- | --- |
| SAE/model registry and compatibility checks | Implemented; tested offline |
| Public capture / feature-edit / patch API | Implemented; both adapter styles tested offline |
| Feature-site metadata and prompt namespace | Implemented; source namespace migrated |
| SimpleCopying prompt preparation and versioned artifacts | Implemented; tested with a deterministic mock model |
| SAE activation health and JSON results | Implemented; tested with a deterministic mock model/SAE |
| Public Pythia-70M + pretrained SAE weights | Runnable configuration; requires real-weight integration run |
| Additional research protocols in `conf/evaluation/` | Configuration references; task implementations not bundled |
| vLLM / NNsight high-performance research backend | Optional; integration and external task dependencies remain to be validated |

Only `task=sae` / `evaluation=activation_health` is registered as an executable
Phase A task. Advanced generation dependencies may be installed into the same
environment with `python -m pip install -r requirements-generation.txt` when
needed for later validation.

The uploaded SAEBase source ZIP remains the reference for its original internal
research protocols. This standalone README covers the packaged public capabilities.

## Testing

```bash
python -m pytest -q
```

Hydra-based tests run when `hydra-core` is installed. The test suite includes
prompt → manifest → activation → SAE → JSON output using small deterministic
objects that require neither model downloads nor CUDA.

See [`docs/PHASE_A.md`](docs/PHASE_A.md) for Phase A's implementation boundary,
and [`docs/PHASE_B.md`](docs/PHASE_B.md) for the public API, namespace migration,
behavioral intervention semantics and real-model integration checks.

## Contributing and releasing

- [API reference](docs/API.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
- [Provenance and third-party assets](docs/PROVENANCE.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Changelog](CHANGELOG.md) · [Apache-2.0 license](LICENSE)

The continuous integration workflow runs offline tests and builds a wheel and
source distribution. Live model, SAE and GPU backend validation is recorded
separately.
