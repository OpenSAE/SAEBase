# WP12 P1 — Model, Activation, Hook, SAE and Trace Backends

This is **working implementation source**, not a final 13/13 acceptance claim.
P1 is intentionally developed with targeted offline tests and one optional
real-checkpoint smoke. Independent parity certification across all combinations
is deferred to P8. No bulk Stage D or prior scientific experiments are rerun.

## What this PR adds

- **TransformerLens version-aware boot**: `models/transformerlens_backend.py` uses
  `HookedTransformer` when available (legacy 3.x) and falls back to
  `TransformerBridge.boot_transformers` on TL4. Compatibility mode registers
  old hook aliases. `no_processing=True` preserves raw Hugging Face weights;
  otherwise the requested legacy weight-processing mode is explicitly applied.
  The loaded model records `_saebase_transformerlens_engine` and
  `_saebase_weight_processing`. Existing plugin model keys and SAE profiles
  remain unchanged.
- **Hooks and patching**: `ModelAdapter.available_hooks(layer=...)` lists
  available hook names when the backend can enumerate them. `capture` checks
  the selected hook; `patch_logits` rejects callbacks that never fire, including
  legacy-vs-bridge calling differences. An unknown hook is an error, not a
  silently ignored patch.
- **Component inspection**: `capture_block_components` obtains
  `resid_pre`, `resid_mid`, `resid_post` and returns two real additive
  differences with explicit shape and sum checks. These are residual-state
  deltas, *not* direct logit attribution or verified causal effects.
- **NNsight local**: `NNsightLocal` operates on a real local NNsight wrapper
  with a typed `TraceLocation`. It offers local tensor capture, native
  invocation-order multi-input traces, tensor output editing and a gradient
  smoke using a specified downstream scalar sum. No NDIF or GPU-serving claim.
- **Gemma Scope 2 asset index**: `registered_scope_assets()` exposes all
  registered official model/SAE/layer/width/L0 identities without loading
  weights. `discover_official_scope_assets()` optionally requests a real HF
  file manifest and pins its 40-character revision SHA. It never constructs
  nonexistent width×L0 combinations. The CPU Research Lab exposes
  `GET /api/sae-assets?model_key=gemma270m` as **registered metadata only**.
- **Real bounded smoke**: `scripts/wp12_p1_smoke.py` checks three public
  inputs with the same registered model, SAE, hooks and no-op/feature edits,
  writing a JSON artifact. This test is opt-in; it is not executed by the
  documentation and does not claim TL4 numerical parity in advance.

## Python examples

Load an actual SAE on an explicitly supported site:

```python
from saebase.api import load_registered, capture_features, capture_block_components

pair = load_registered(sae_key="pythia70m_resid_sm", layer=3, device="cuda")
ids = pair.model.tokens(prompt="The capital of France is")
print(pair.identity.to_dict(), pair.model.available_hooks(layer=3))
snapshot = capture_features(pair.model, pair.sae, tokens=ids, site=pair.site)
print(snapshot.codes.shape, snapshot.activation.shape)

# Only if pre/mid/post hooks are all present; otherwise fail explicitly.
parts = capture_block_components(pair.model, tokens=ids, layer=3)
print(parts.attention_delta.norm(), parts.mlp_delta.norm())
```

List installed registry coverage offline, or explicitly query official HF files:

```python
from saebase.sae.catalog import registered_scope_assets, discover_official_scope_assets

print(registered_scope_assets(model_key="gemma270m"))
# Requires network/authorized HF access; metadata only, no weight download.
print(discover_official_scope_assets("google/gemma-scope-2-270m-pt"))
```

Use a **locally loaded** NNsight model without activating remote NDIF:

```python
import torch
from saebase.api import NNsightLocal, TraceLocation

module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
trace = NNsightLocal.from_torch(module)
result = trace.capture(torch.zeros(1, 3), TraceLocation("[0]"))
print(result.shape)
```

NNsight paths are relative to the wrapped root (so `[0]` denotes the first module of a Sequential model). The exact module path depends on the wrapper's tree and
model class. Pick it from the native model (do not guess architecture names).
For gradients, choose the loss output at a module reached later in the forward
pass. Native NNsight enforces trace order. `TraceLocation` does not accept
private attributes, Python expressions or arbitrary method calls.

## Environment and targeted checks

Each backend should be tested in a **separate virtual environment**; do not
blindly install TL4, legacy TL3, vLLM and every training dependency together.

```bash
python -m pip install -e ".[dev,tl4]"
python -m pip check
python -m pytest -q tests/test_wp12_p1_backends.py
```

For the local NNsight worker, use `python -m pip install -e ".[dev,nnsight-local]"`.
The legacy GPU-00 environment should remain pinned to the originally reported
TL3.9 / SAELens6.54 protocol; do not replace the model weights or retroactively
reinterpret its measurements.

The *optional* real smoke downloads only the chosen public weights:

```bash
python scripts/wp12_p1_smoke.py --sae-key pythia70m_resid_sm --layer 3 \
  --device cuda --output /workspace/wp12_p1_small_smoke.json
```

If a model needs authorisation, the worker stops; never bypass gates. Cache
keys must include exact model/SAE/token/site and weight-processing identities;
old Stage D and GPU-00 artifacts are not reused as TL4 evidence.

## P1 mapping to the roadmap

| Capability | Implemented or reusable route | Limit to retain |
| --- | --- | --- |
| F06 | Version-aware TL3/TL4 loader | TL4 real-weight parity pending |
| F07 | Existing checksummed activation and code store | New TL4 processing identity requires smoke |
| F08 | Existing capture/patch + missing-hook/firing checks | Cross-version real parity pending |
| F09 | Scoped hooks via original TL API | Fireable hook names depend on backend |
| F10 | Residual block component differences | Not a full direct-logit-attribution feature |
| F11 | Existing SAELens native pretrained checkpoint loader | No new native benchmark yet |
| F14 | Existing model/SAE profile shape and site checks | P1 cross-library matrix not exhaustive |
| F16 | Local NNsight intermediate and gradient API | Native real-model invocation pending |
| F17 | Local NNsight tensor edit API | Does not imply SAE-code edit equivalence |
| F18 | Native NNsight multi-input invoke wrapper | Local trace only |
| F41 | Registered official Scope asset listing | Manifest metadata, not weights |
| F42 | Registered layer/hook mapping | Full site-family mapping not claimed |
| F45 | Registered and real HF-manifest width/L0 variants | New variants are unverified until loaded |

**Remaining development objective:** one end-to-end local model + SAE +
feature + patch through TL4, a native NNsight mini-test and a scoped Gemma
asset variant selection. If runtime packages or hardware are unavailable, the
route stays `IMPLEMENTED_UNTESTED` instead of claiming 13 verified capabilities.

## Minimal change and scientific integrity

Keep P0 legacy results and SHA256 unchanged, retain the existing P0 draft
branch/PR, and do not modify SAR, RSS, Input Localization or old benchmarks.
P2–P7 may proceed once the new code passes routine basic tests. The exhaustive
50-feature numerical and external-user acceptance belongs in **P8**.
