# Extend SAEBase with Models, SAEs and Benchmarks

SAEBase can use external model adapters, SAE checkpoints and metrics without
modifying its built-in registry. Third-party Python extensions are strictly
opt-in: importing SAEBase never executes an installed extension package.

The extension API is included in **SAEBase v0.3.0**. Install the published release or use a current source checkout. Extensions are explicitly discovered and never executed merely by importing SAEBase.

## A working offline example

From the repository checkout, run:

~~~bash
python examples/extensions/external_providers.py
~~~

This tiny synthetic example registers a model, SAE and external metric,
checks hook and dimensions, probes encoding/decoding and identity patch,
then executes an M3 benchmark using the original result and report format.
It needs no GPU or checkpoint downloads. **It is a software demonstration,
not a verified pretrained SAE evaluation.**

## Register external providers

An external package provides an `ExtensionBundle` containing Model, SAE
and/or Metric definitions. For example:

~~~python
from pathlib import Path

from saebase.api import ActivationSite
from saebase.benchmarks import MetricSpec
from saebase.extensions import (
    ExtensionBundle, ExternalModel, ExternalSAE, ExternalMetric, HookSpec,
)

SITE = ActivationSite(layer=12, hook_name="blocks.12.hook_resid_post")


def make_extension():
    return ExtensionBundle(
        name="ext.my_lab.bundle",
        models=(
            ExternalModel(
                key="ext.my_lab.model",
                source="https://huggingface.co/my-lab/my-model",
                revision="IMMUTABLE_MODEL_REVISION",
                hooks=(HookSpec(SITE, 4096),),
                factory=load_my_model,  # receives a device string
            ),
        ),
        saes=(
            ExternalSAE(
                key="ext.my_lab.sae",
                model_key="ext.my_lab.model",
                source="https://huggingface.co/my-lab/my-sae",
                revision="IMMUTABLE_SAE_REVISION",
                site=SITE, d_in=4096, d_sae=32768,
                architecture="topk", normalization="folded",
                dtype="bfloat16",
                factory=load_my_sae,  # receives a device string
            ),
        ),
        metrics=(
            ExternalMetric(
                spec=MetricSpec(
                    "ext.my_lab.metric", "1", ("external_pair",),
                    "dataset", protocol_id="ext.my_lab.protocol.v1",
                ),
                source="https://github.com/my-lab/original-estimator",
                revision="IMMUTABLE_ESTIMATOR_REVISION",
                source_file=Path(__file__).resolve(),
                evaluator=compute_my_metric,
            ),
        ),
    )
~~~

`load_my_model`, `load_my_sae` and `compute_my_metric`
represent functions from the *external* package. This example is a
contract, not a new SAEBase model/SAE implementation.

Each external model must expose an existing TransformerLens or SAEBase
HF-hooks compatible backend. Each SAE must implement `encode`,
`decode` and expose `cfg.d_in` and `cfg.d_sae`.
An SAE's target model, exact layer, hook name and `d_in` must match
a registered Model HookSpec. Normalization and dtype must be explicitly
declared; `unspecified` is refused. Custom normalization remains
the provider's responsibility, not a silent SAEBase conversion.

For multi-layer SAE families, register distinct SAE keys per hook and layer.
A checkpoint trained for another Base/Instruct model or hook is not
automatically transferable merely because tensor widths are equal.

## Explicit discovery through Python packaging

In the third-party package's `pyproject.toml`, declare:

~~~toml
[project.entry-points."opensae.saebase.extensions.v1"]
my_lab = "my_lab.saebase_plugin:make_extension"
~~~

After installing and trusting that package:

~~~python
from saebase.extensions import discover_extensions, load_external_pair

discover_extensions(["my_lab"])
pair = load_external_pair(
    model_key="ext.my_lab.model",
    sae_key="ext.my_lab.sae",
    layer=12,
    device="cuda",
)
print(pair.verification)  # loaded_unverified
~~~

The `discover_extensions` call is explicit. An installed extension
is executable Python, not a sandboxed configuration file.

Benchmarks can use the same provider via the original CLI with a new
optional flag:

~~~bash
saebase-benchmark --extension my_lab --config benchmark.json --dry-run
saebase-benchmark --extension my_lab --config benchmark.json --real
~~~

Dry-run validates the registered metric, hook, layer and budget without
downloading a checkpoint. The real mode explicitly executes provider
factories and runs the requested benchmark. Passing the `--real`
flag does not, by itself, prove real pretrained weights were tested.

External metric evaluators return an `ExternalObservation` with
a finite result, positive observation count and optional measurement
references. This initial API supports required inputs
`external_pair`, `m1_activations` and `m1_codes`.
When an input uses M1 activations or SAE codes, the existing
identity-checked cache is reused.

For every external benchmark, its frozen config must also include
the model and SAE source/revision/site records. Generate their exact
values with `external_pair_sources(MODEL_KEY, SAE_KEY)` and place
them under `sources` next to the metric source path. SAEBase includes
these records in the deterministic run identity. Any change to the
registered checkpoint source, revision, hook, width, dtype or
normalization invalidates the benchmark before it can silently resume
an old result.

For every external metric, the frozen benchmark config must include
`sources["ext.my_lab.metric"]` pointing to the exact original local
estimator source file. SAEBase records its SHA256 and the declared
revision and verifies those identities on resume and report-only
execution. External metric definitions and original SAR, RSS,
Input Localization, and SAEBench implementations remain separate.

## What verification actually means

- **Registered:** factory definitions and metadata exist, not necessarily
  loaded checkpoint weights.
- **Loaded, unverified:** model and SAE objects satisfy structural
  compatibility. SAEBase records hook, model/SAE revisions, dimensions,
  declared normalization and dtype.
- **Probe passed:** when actual token IDs are supplied to
  `load_external_pair(..., probe_tokens=...)`, SAEBase checks
  activation capture, SAE encoding/decoding and identity patch.
  This is one-site software acceptance, not a scientific result.
- **Real checkpoint and scientific benchmarking:** requires a separately
  documented genuine source and weights, frozen experimental protocol,
  representative evaluation inputs and archived metrics. Registration
  never grants this level of verification.

Third-party weights are not redistributed by SAEBase. The user of an
external extension remains responsible for trusting its executable
code and respecting the upstream model/SAE licenses.
