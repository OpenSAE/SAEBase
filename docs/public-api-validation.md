# Public API Validation

This page preserves detailed technical contracts and verification context. For current product-facing coverage, see [Validation and Reproducibility](validation.md) and [Documentation Home](index.md).

**Source baseline:** `SAEBase_Phase_A_Ready.zip`, kept as an unchanged source artifact.

## Work completed

- Added `saebase.api`: `ModelAdapter`, explicit backend capability reporting,
  token/site selection, capture, baseline/patch logits and replay consistency.
- Added SAE feature capture/edit API that retains original activation content,
  including the reconstruction residual; supports custom code values and subsets.
- Added `SAEIdentity` with checkpoint, layer/hook, original/evaluation model,
  SAE dimensions, loader, actual loaded model name, and explicitly unknown revisions.
- Added registered-checkpoint factory `load_registered()`, including hook and
  model/SAE identity checks.
- Moved built-in prompt handling from top-level `datasets` to `saebase.data`
  to resolve the Python package collision with Hugging Face `datasets`.
  Migrated all bundled Python call sites and retained the root `./data` artifact path.
- Simplified `models/hf_compat.py`: imports are direct, with no global path or
  import-cache modification.
- Added a self-contained synthetic example and an opt-in actual Pythia smoke
  entrypoint. Updated README and added API/migration documentation.

## Executed locally (2026-10-13 source series)

```text
python -m pytest -q
61 passed, 1 skipped
```

The skipped case is the existing Hydra configuration composition test (the
current execution environment does not have hydra-core installed). No automated
unit test failure remains.

The new API tests validate TransformerLens-style caching/patching and
HFHooks-style reading/patching, identity replays, custom SAE decode bias,
residual preservation, subset isolation, real HFHooksModel wrapper with a tiny
synthetic `torch.nn.Module`, output-logit changes, capability errors,
model/site and metadata drift, and Python namespace separation.

Further checks:

- Python parsing of all 171 modules succeeded.
- `python main.py --help` works without vLLM or pretrained weights.
- `python -m saebase.sae.verify_registry --list` works.
- `python examples/quickstart/public_api_offline.py` executed and emitted valid JSON;
  the toy logits differ after editing a selected SAE feature.
- Default prompt artifacts continue to resolve at `<repo>/data`.

## Earlier external integration status (historical)

The following depend on packages and checkpoint downloads not available in
this environment:

```bash
python -m pip install -r requirements.txt
python examples/interventions/pythia_feature_edit.py
bash scripts/phase_a_smoke.sh
```

External checkpoint compatibility and the real Pythia-70M model outputs are
**not claimed as verified**. The synthetic model numbers are test fixtures,
not research evidence. The existing vLLM and NNsight research paths remain
outside public API.

## Compatibility

core setup CLI, model/SAE registry content, activation-health estimator and
prompt JSON schema remain unchanged. Source imports for the bundled prompt
code now use `saebase.data.*` instead of `datasets.*`. The latter top-level
name is reserved for the Hugging Face package. Repository-relative default
paths and the core setup CLI options are preserved.
