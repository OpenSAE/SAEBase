"""End-to-end controlled Boolean input design and cache provenance tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.data import (
    BinaryFactor, ControlledInputDesign, ControlledInputManifest,
    collect_controlled_input_activations, load_controlled_input_manifest,
)
from saebase.integrations import (
    collect_localization_grid, localization_grid_from_code_store,
)
from saebase.store import (
    ActivationIdentity, ActivationWriter, CodeIdentity, ShardedReader,
    encode_from_store, sha256_state_dict,
)


def _design():
    factors = (
        BinaryFactor("left", ("1", "2")),
        BinaryFactor("middle", ("3", "4")),
        BinaryFactor("right", ("5", "6")),
    )
    return ControlledInputDesign.from_renderer(
        factors=factors, name="toy-factorial",
        render=lambda x: f'{x["left"]} {x["middle"]} {x["right"]}'
    )


def _components():
    adapter = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    activation_identity = ActivationIdentity(
        model_key="test-model", model_state_sha256=sha256_state_dict(adapter.model),
        tokenizer_sha256=hashlib.sha256(b"test-tokenizer").hexdigest(),
        hook_name=site.hook_name, layer=site.layer,
    )
    sae_identity = CodeIdentity("test-sae", sha256_state_dict(sae),
                               sae.cfg.d_in, sae.cfg.d_sae)
    return adapter, sae, site, activation_identity, sae_identity


def test_binary_state_order_and_fingerprints():
    design = _design()
    assert design.n_factors == 3 and design.n_states == 8
    assert design.states[0].prompt == "1 3 5"
    assert design.states[1].prompt == "2 3 5"
    assert design.states[2].prompt == "1 4 5"
    assert design.states[4].prompt == "1 3 6"
    adapter, _, _, _, _ = _components()
    records = design.sample_records(adapter)
    assert all(r.prompt_group_id == design.prompt_group_id for r in records)
    assert all(r.selection.mode == "indices" and r.selection.indices == (2,) for r in records)
    assert len({r.sample_id for r in records}) == 8
    assert design.key == _design().key


def test_reindex_is_bijection_and_preserves_source_identity():
    design = _design()
    xor = {i: ((i & 1) ^ ((i >> 1) & 1)) | (i & ~1) for i in range(8)}
    mapped = design.reindex(xor)
    assert mapped.key != design.key
    assert mapped.prompt_group_id == design.prompt_group_id
    assert mapped.sample_ids_by_state[3] == design.sample_ids_by_state[xor[3]]
    assert set(mapped.sample_ids_by_state.values()) == set(design.sample_ids_by_state.values())
    with pytest.raises(ValueError, match="permutation"):
        design.reindex({i: 0 for i in range(8)})
    with pytest.raises(ValueError, match="cover"):
        design.reindex({i: i for i in range(7)})


def test_design_rejects_tokenizer_state_collisions_before_capture():
    class CollidingTokens(ResearchToyModel):
        def to_tokens(self, prompt):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    model = ModelAdapter(CollidingTokens().eval())
    with pytest.raises(ValueError, match="identical token IDs"):
        collect_localization_grid(
            model, ResearchToySAE().eval(),
            site=ActivationSite(0, "blocks.0.hook_resid_post"), design=_design()
        )


def test_invalid_factor_states_and_positions():
    with pytest.raises(ValueError, match="distinct"):
        BinaryFactor("x", ("same", "same"))
    with pytest.raises(ValueError, match="unique"):
        ControlledInputDesign.from_renderer(
            factors=(BinaryFactor("x"), BinaryFactor("x")),
            render=lambda x: str(x),
        )
    factors = (BinaryFactor("x"), BinaryFactor("y"))
    with pytest.raises(ValueError, match="cover"):
        ControlledInputDesign.from_prompts(
            factors=factors, prompts_by_state={0: "a", 1: "b", 2: "c"}
        )
    with pytest.raises(ValueError, match="distinct prompts"):
        ControlledInputDesign.from_prompts(
            factors=factors, prompts_by_state={0: "a", 1: "a", 2: "b", 3: "c"}
        )
    with pytest.raises(ValueError, match="token position"):
        ControlledInputDesign.from_renderer(
            factors=factors, render=lambda x: "1 2 3 " + x["x"] + x["y"],
            position=100,
        ).sample_records(_components()[0])


def test_direct_design_localize_capture_preserves_bitmask_order():
    model, sae, site, _, _ = _components()
    design = _design()
    direct = collect_localization_grid(model, sae, site=site, design=design)
    assert direct.design_key == design.key
    assert direct.state_sample_ids == tuple(s.sample_id for s in design.states)
    assert direct.codes.shape == (8, 7)
    with pytest.raises(ValueError, match="not both"):
        collect_localization_grid(
            model, sae, site=site, design=design, n_factors=3
        )


def test_factorial_cache_and_manifest_reuse(tmp_path: Path):
    model, sae, site, activation_id, sae_id = _components()
    original = _design()
    mapping = {i: ((i & 1) ^ ((i >> 1) & 1)) | (i & ~1) for i in range(8)}
    transformed = original.reindex(mapping, name="xor-map")
    with ActivationWriter(tmp_path, activation_id, width=sae.cfg.d_in) as writer:
        first = collect_controlled_input_activations(model, original, writer, site=site)
        second = collect_controlled_input_activations(model, transformed, writer, site=site)
        assert first.stats.collected == 8 and first.stats.reused == 0
        assert second.stats.collected == 0 and second.stats.reused == 8
        assert first.manifest_path != second.manifest_path
        source_path = writer.path
    code_path, code_stats = encode_from_store(source_path, sae, sae_id)
    assert code_stats.collected == 8
    original_grid = localization_grid_from_code_store(code_path, manifest=first.manifest_path, site=site)
    mapped_grid = localization_grid_from_code_store(code_path, manifest=second.manifest_path, site=site)
    assert original_grid.design_key == original.key
    assert mapped_grid.design_key == transformed.key
    assert original_grid.state_sample_ids == tuple(s.sample_id for s in original.states)
    np.testing.assert_allclose(mapped_grid.codes, original_grid.codes[list(mapping.values())], rtol=0, atol=0)
    assert sorted(original_grid.state_sample_ids) == sorted(mapped_grid.state_sample_ids)
    live_grid = collect_localization_grid(model, sae, site=site, design=original)
    np.testing.assert_allclose(live_grid.codes, original_grid.codes, rtol=0, atol=0)

    with ActivationWriter(tmp_path, activation_id, width=sae.cfg.d_in) as writer:
        replay = collect_controlled_input_activations(model, original, writer, site=site)
        assert replay.stats.collected == 0 and replay.stats.reused == 8
    _, code_replay = encode_from_store(source_path, sae, sae_id)
    assert code_replay.collected == 0 and code_replay.reused == 8
    assert len(list(ShardedReader(code_path).iter_samples())) == 8


def test_manifest_rejects_modified_source_and_model_identity(tmp_path: Path):
    model, sae, site, activation_id, sae_id = _components()
    with ActivationWriter(tmp_path, activation_id, width=sae.cfg.d_in) as writer:
        capture = collect_controlled_input_activations(model, _design(), writer, site=site)
        source_path = writer.path
    code_path, _ = encode_from_store(source_path, sae, sae_id)
    path = capture.manifest_path
    data = json.loads(path.read_text())
    data["states"][0]["sample_id"] = "fabricated"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_controlled_input_manifest(path)
    with pytest.raises(ValueError, match="checksum"):
        localization_grid_from_code_store(code_path, manifest=path, site=site)

    good = capture.manifest.to_dict()
    good["activation_identity_key"] = "b" * 64
    good.pop("manifest_sha256")
    from saebase.store import fingerprint
    good["manifest_sha256"] = fingerprint(good)
    different = ControlledInputManifest.from_dict(good)
    with pytest.raises(ValueError, match="different activation cache"):
        localization_grid_from_code_store(code_path, manifest=different, site=site)


def test_manifest_position_integrity(tmp_path: Path):
    model, sae, site, activation_id, sae_id = _components()
    with ActivationWriter(tmp_path, activation_id, width=sae.cfg.d_in) as writer:
        capture = collect_controlled_input_activations(model, _design(), writer, site=site)
        source_path = writer.path
    code_path, _ = encode_from_store(source_path, sae, sae_id)
    altered = capture.manifest.to_dict()
    altered["states"][0]["position"] = 1
    altered.pop("manifest_sha256")
    from saebase.store import fingerprint
    altered["manifest_sha256"] = fingerprint(altered)
    with pytest.raises(ValueError, match="position mismatch"):
        localization_grid_from_code_store(
            code_path, manifest=ControlledInputManifest.from_dict(altered), site=site
        )
