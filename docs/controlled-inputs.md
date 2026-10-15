# Controlled Input Designs

SAEBase can prepare a complete binary input experiment, collect its model activations once, and reuse the same cached SAE codes under different **input-state mappings**. This supports [Input Localization](https://github.com/OpenSAE/SAE-Input-Localization) without moving its estimators into SAEBase.

## Offline example

```bash
python examples/16_controlled_input_design.py
```

The example defines three binary factors, evaluates all eight input states with a small synthetic model, encodes one SAE, and reuses the identical activation/code cache for an XOR state permutation. It writes a checked design manifest alongside the existing activation store.

## Python API

```python
from saebase.data import BinaryFactor, ControlledInputDesign

design = ControlledInputDesign.from_renderer(
    factors=(
        BinaryFactor("first", ("1", "2")),
        BinaryFactor("second", ("3", "4")),
        BinaryFactor("third", ("5", "6")),
    ),
    name="three-factor-experiment",
    render=lambda v: f'{v["first"]} {v["second"]} {v["third"]}',
)

# State mask 0 -> "1 3 5"; mask 1 -> "2 3 5"; mask 2 -> "1 4 5".
records = design.sample_records(model_adapter)
```

An input state mask uses its bit `k` as the level index of factor `k`, starting with the least significant bit. `sample_records()` rejects repeated model token sequences and resolves one explicit token position per state. It emits existing `SampleRecord` objects, so SAEBase's activation and code stores do not need a new file format.

With a prepared `ActivationWriter`, use:

```python
from saebase.data import collect_controlled_input_activations
from saebase.store import encode_from_store
from saebase.integrations import localization_grid_from_code_store, evaluate_localization

captured = collect_controlled_input_activations(
    model_adapter, design, writer, site=site
)
activation_path = writer.path

# After closing the writer:
code_path, _ = encode_from_store(activation_path, sae, sae_code_identity)
grid = localization_grid_from_code_store(
    code_path, manifest=captured.manifest_path, site=site
)
result = evaluate_localization(grid)  # requires the original sae_localize package
```

The estimator remains an optional independent dependency:

```bash
python -m pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git
```

The direct capture path also accepts `collect_localization_grid(model, sae, site=site, design=design)` without caching.

## Reindex the same observations

```python
new_to_old = {
    i: ((i & 1) ^ ((i >> 1) & 1)) | (i & ~1)
    for i in range(8)
}
mixed = design.reindex(new_to_old, name="xor-map")
```

`reindex` requires a bijection. It **changes which controlled state is associated with each existing input sample**, but preserves the original input's `sample_id`, token IDs, and prompt group. Capturing the reindexed design with the same store reuses the original activations. Its manifest and `design_key` are different, because the scientific input-state correspondence changed.

## Stored manifest

Each design is saved to `<activation-store>/controlled_inputs/<design_key>.json`. Its `saebase.controlled_inputs.v1` sidecar contains factor labels, explicit bitmask order, sample IDs, selected token positions, token-site checksums, the activation identity key and a manifest checksum. It does **not** persist raw input text. The existing `SampleRecord`, activation shard and code shard formats remain unchanged.

`load_controlled_input_manifest()` validates the sidecar; `localization_grid_from_code_store(..., manifest=...)` also checks that its model activation store, original token IDs and selected positions agree. A manifest from another model cache cannot be silently reused.

## Scope and interpretation

The complete `2**n_factors` input grid is uniform over all declared factor combinations. Currently the public builder supports binary factors, one model activation position per state, and at most 12 factors per experiment; this size bound controls accidental exhaustive enumeration, not Localize's computational complexity.

The input rendering function defines the actual experimental manipulations. SAEBase checks coverage, uniqueness, tokenization, positions and cache identity; **it cannot infer causal independence or semantic correctness from prompt strings**. Reindexing is not a new model forward pass and is not SAE feature ablation. The Localize estimator retains its own exact-curve and finite-sample interpretation.

[Activation Store](activation-store.md) · [Localize Benchmarking](localization-benchmark.md) · [Research Integrations](research-integrations.md) · [Examples](../examples/README.md)
