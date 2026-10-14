# SAEBase Phase B — Public Activation and SAE APIs

Phase B adds a common read, edit and replay interface for model activations and
SAE features. It runs on a registered pretrained SAE or a user-supplied SAE.
The core `saebase/sae/` implementation from Phase A remains available.

## API at a glance

```python
from saebase.api import (
    ActivationSite, ModelAdapter, SAEIdentity, capture_features,
    coalition_replacements, evaluate_edit, load_registered,
)

# Option 1: load a bundled public checkpoint (downloads weights).
bundle = load_registered(
    model_key="pythia70m", sae_key="pythia70m_resid_sm",
    layer=3, device="cpu",
)
site = bundle.site
snapshot = capture_features(
    bundle.model, bundle.sae, site=site, prompt="Alice Emma Alice"
)
selected = snapshot.codes.abs().topk(4).indices.tolist()
replacements = coalition_replacements(
    snapshot.codes, selected_indices=selected, retained_indices=selected[:2]
)
result = evaluate_edit(
    bundle.model, bundle.sae, site=site, prompt="Alice Emma Alice",
    replacements=replacements,
)
print(bundle.identity.to_dict())
print(result.logit_delta.shape)
```

**Offline example (synthetic, runnable without weights):**

```bash
python examples/05_public_api_offline.py
```

**Real checkpoint integration smoke (requires model downloads):**

```bash
python examples/06_public_api_pythia.py
```

## Core contracts

| Object / function | What it means |
| --- | --- |
| `ModelAdapter(model)` | Exposes backend capabilities: `forward`, `capture`, `patch`, `generate` |
| `ActivationSite(layer, hook_name, position=-1)` | Exact activation location; `-1` last token, `-2` second-last |
| `capture_features(...)` | Captures an original activation, encodes it, and decodes the SAE code |
| `edit_features(...)` | Changes explicit SAE code indices and returns the patched activation |
| `evaluate_edit(...)` | Runs model baseline and patched replay; returns logits and differences |
| `SAEIdentity` | Records model/SAE identifier, hook, dimensions, loader and known revisions |
| `load_registered(...)` | Calls the existing SAEBase model/SAE registry and constructs an adapter |

The public API operates on **one prompt and one token position per call**. `tokens`
must have `[1, length]` shape and integral IDs. A `prompt` is tokenized by the
loaded model when provided instead of `tokens`. SAE latents are one-dimensional
vectors `[d_sae]` after selecting a token.

Both TransformerLens `run_with_cache` / `run_with_hooks` and the SAEBase HF
residual wrapper `get_resid` / `run_with_patch` can implement this contract.
The HF wrapper supports **block output** sites named `blocks.{layer}.hook_resid_post`.
Models with forward capability alone can expose logits through `forward_logits()`;
models that additionally support capture or patch advertise each ability separately.

The logits readout position is **independent** of the activation patch position,
so patching the second-to-last token and reading the final token is supported.
Readout returns a logit vector `[vocab]`. Other behavioral metrics belong to
calling research projects.

## Feature intervention semantics

For the observed activation `h`, encode `z = SAE.encode(h)`. After replacing
selected latent values, the patched activation is

\[
h_{\mathrm{patched}}=h+\operatorname{decode}(z')-\operatorname{decode}(z).
\]

The unchanged SAE code values and the residual `h - decode(z)` are retained.
`decode()` is called twice through the actual model API, including its bias and
normalization, rather than constructing a replacement from decoder weights.
For a linear decoder, this is the usual sum of edited feature contributions.
For a nonlinear decoder it is exactly the difference of actual decoded outputs.

`coalition_replacements(z, selected_indices, retained_indices)` zeros only
selected SAE feature IDs outside the retained group. All indices are actual
SAE feature indices, not offsets in the selected subset.

**Reliability:** `evaluate_edit(..., verify_identity=True)` first replays the
captured, unchanged activation through the patch hook. That replay must reproduce
the original forward logits (within configured numerical tolerance). Tests also
check that deleting one selected feature preserves all other code values,
that the residual survives editing, and that unsupported hooks are rejected.

This is a **model intervention**, not a claim of causal attribution to a feature
without the experiment's additional controls. For `torch.bfloat16` / `float16`
weights, numerical tolerances should be checked on the target backend.

## SAE provenance

`SAEIdentity.from_loaded()` uses the metadata returned by `load_model_and_sae()`
and the actual loaded SAE dimensions. You may pass exact `revision` and
`model_revision` values if known. The API leaves them as `null` otherwise.
`normalization` remains `unspecified` unless supplied. A custom SAE can use
`SAEIdentity.for_custom(...)` with a stable checkpoint reference and hook.

The identity records the **trained-on model**, **evaluation host model**, and
**transfer kind** where provided. A host model outside the profile's allowed
models fails validation. `load_registered()` also checks actual versus
configured hook and backend support.

## Package namespace migration

Phase B moves repository-owned prompt code:

```text
Phase A: datasets/prompt_store.py
Phase B: saebase/data/prompt_store.py
```

The canonical import is now:

```python
from saebase.data.prompt_store import load_prompts
from saebase.data.artifact_layout import artifact_dir
```

All bundled code, examples and tests use the new namespace. The **top-level
`datasets` name is reserved for the Hugging Face package**, so the old
`sys.path` / `sys.modules` import manipulation was removed from
`models/hf_compat.py`. Consumer scripts importing `datasets.<SAEBase-module>`
should change to `saebase.data.<SAEBase-module>`.

The physical default artifact location remains `./data` at repository root.
Registry entries, prompt JSON formats, model/SAE IDs, CLI options and the
existing Phase A evaluation metric definitions are preserved.

## Verification and evidence

- Existing Phase A tests remain in the suite.
- New synthetic tests exercise both backend styles, site and token validation,
  missing capabilities, SAE metadata, biased-decoder residual preservation,
  identity replay, and downstream changes.
- The offline example is executed as an additional end-to-end smoke test.
- The actual Pythia-70M pretrained model/SAE integration remains an external
  acceptance test pending a Python environment with the model dependencies
  and published checkpoint access.

Standard check:

```bash
python -m pytest -q
python examples/05_public_api_offline.py
```

Integration check with published weights:

```bash
python examples/06_public_api_pythia.py
bash scripts/phase_a_smoke.sh
```

## Phase boundary

Phase B focuses on shared model capability, provenance, positions and
activation/SAE interventions. It does not alter SAR or RSS estimators,
archive-specific missing research tasks, model training or GPU generation
integration. Model-specific NNsight / vLLM intervention capabilities are kept
separate for later verification.
