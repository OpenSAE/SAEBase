"""M1 contracts: identities, selection, streaming, atomic resume, CSR, and SAE reuse."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.store import (ActivationIdentity, ActivationWriter, CacheIntegrityError,
                           CodeIdentity, SampleRecord, TokenSelection, ShardedReader,
                           capture_selected, collect_activations, encode_from_store,
                           resolve_positions, sha256_state_dict)


def identity(**changes):
    return replace(ActivationIdentity(model_key="toy", model_state_sha256="a" * 64,
                                      tokenizer_sha256="b" * 64,
                                      hook_name="blocks.0.hook_resid_post", layer=0), **changes)


def sample(name="one", tokens=(1, 2, 3), *, selection=None, mask=None, offsets=None,
           split="evaluation"):
    return SampleRecord(name, tokens, attention_mask=mask,
                        selection=selection or TokenSelection(), token_offsets=offsets,
                        prompt_group_id="g1", split=split)


def site():
    return ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")


def live_identity(model):
    return identity(model_state_sha256=sha256_state_dict(model))


def codes(key="sae-a", digest="c"):
    return CodeIdentity(sae_key=key, sae_state_sha256=digest * 64,
                        d_in=4, d_sae=7)


@pytest.mark.parametrize("field,value", [
    ("model_state_sha256", "c" * 64), ("tokenizer_sha256", "d" * 64),
    ("chat_template_sha256", "e" * 64), ("hook_name", "blocks.0.hook_mlp_out"),
    ("layer", 1), ("dtype", "float16"),
    ("preprocessing", {"bos": False}), ("execution", {"backend": "tl"}),
])
def test_identity_provenance_invalidation(field, value):
    if field == "layer":
        altered = replace(identity(), layer=1, hook_name="blocks.1.hook_resid_post")
    else:
        altered = replace(identity(), **{field: value})
    assert altered.key != identity().key
    assert len(identity().key) == 64
    assert codes().key != codes("different").key


def test_identity_rejects_mutable_or_invalid_provenance():
    with pytest.raises(ValueError, match="SHA-256"):
        identity(model_state_sha256="main")
    with pytest.raises(ValueError, match="layer"):
        identity(hook_name="blocks.2.hook_resid_post")
    with pytest.raises(ValueError, match="dtype"):
        identity(dtype="float64")
    with pytest.raises(ValueError, match="inference"):
        CodeIdentity("x", "a" * 64, 4, 7, inference_mode="batch_topk")


def test_selection_last_mask_indices_span_and_all():
    tokens = (101, 5, 6, 99)
    assert resolve_positions(sample(tokens=tokens, mask=(1, 1, 1, 0))) == (2,)
    assert resolve_positions(sample(tokens=tokens, mask=(1, 1, 0, 0),
                                    selection=TokenSelection("all_valid"))) == (0, 1)
    assert resolve_positions(sample(tokens=tokens, selection=TokenSelection("indices", (2, 1)))) == (2, 1)
    assert resolve_positions(sample(tokens=tokens, selection=TokenSelection("span", char_span=(3, 8)),
                                    offsets=((0, 0), (0, 4), (4, 7), (0, 0)))) == (1, 2)
    with pytest.raises(ValueError, match="token_offsets"):
        sample(tokens=tokens, selection=TokenSelection("span", char_span=(0, 1)))
    with pytest.raises(ValueError, match="valid"):
        sample(tokens=tokens, mask=(1, 1, 0, 0), selection=TokenSelection("indices", (2,)))
    with pytest.raises(ValueError, match="unique"):
        TokenSelection("indices", (1, 1))


def test_dense_roundtrip_resumability_and_consistent_metadata(tmp_path):
    r1 = sample("a", (3, 4, 5), selection=TokenSelection("all_valid"))
    r2 = sample("b", (1, 2, 4))
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([[11., 12., 13., 14.]], dtype=np.float32)
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as writer:
        assert writer.append(r1, a)
        assert writer.append(r2, b)
        assert writer.append(r1, a) is False
        writer.flush()
        path = writer.path
        assert writer.committed_samples == 2
    store = ShardedReader(path)
    assert store.sample_count == 2 and store.row_count == 4
    assert tuple(store.get("a").positions) == (0, 1, 2)
    np.testing.assert_array_equal(store.get("a").values, a)
    np.testing.assert_array_equal(store.get("b").values, b)
    assert len((path / "samples.jsonl").read_text().splitlines()) == 2
    with ActivationWriter(tmp_path, identity(), width=4) as writer:
        assert writer.has(r1) and writer.has(r2)
        assert not writer.append(r1, a)
        assert writer.committed_samples == 2
    assert len(ShardedReader(path).manifest["shards"]) == 1
    with pytest.raises(CacheIntegrityError, match="changed|differs"):
        with ActivationWriter(tmp_path, identity(), width=4) as writer:
            writer.has(replace(r1, split="test"))
    with pytest.raises(CacheIntegrityError, match="match existing"):
        ShardedReader(path, expected_identity=identity(dtype="float16").to_dict())
    with pytest.raises(FileExistsError):
        ActivationWriter(tmp_path, identity(), width=4, resume=False)


def test_small_shards_streaming_and_window_budget(tmp_path):
    rows = [sample(f"row-{i:04d}", (i, i+1)) for i in range(112)]
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as writer:
        for i, record in enumerate(rows):
            assert writer.append(record, np.full((1, 4), i, dtype=np.float32))
            assert writer._pending_bytes <= 128
        path = writer.path
    reader = ShardedReader(path)
    assert len(reader.manifest["shards"]) >= 10
    assert reader.sample_count == len(rows)
    samples = sum((batch for batch in reader.iter_batches(max_rows=7)), [])
    assert len(samples) == 112
    np.testing.assert_allclose(reader.get("row-0111").values, np.full((1, 4), 111))
    assert reader.get("row-0000").values.dtype == np.float32
    assert len(reader._mmap) <= 2  # Bounded number of mapped shards.


def test_finite_shape_dtype_errors(tmp_path):
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as w:
        with pytest.raises(ValueError, match="shape"):
            w.append(sample(), np.ones((2, 4), np.float32))
        with pytest.raises(ValueError, match="finite"):
            w.append(sample(), np.array([[0, 1, np.nan, 2]], np.float32))
        with pytest.raises(ValueError, match="shape"):
            w.append(sample(), np.ones((1, 3)))


@pytest.mark.parametrize("dtype,atol", [("float32", 0), ("float16", .003), ("bfloat16", .03)])
def test_numeric_storage_precisions_and_error_bounds(tmp_path, dtype, atol):
    values = np.array([[1.111, -2.222, .14159, 3.333]], dtype=np.float32)
    with ActivationWriter(tmp_path, identity(dtype=dtype), width=4,
                          shard_target_bytes=256) as writer:
        writer.append(sample(), values)
        assert writer.max_observed_quantization_error <= atol
        path = writer.path
    np.testing.assert_allclose(ShardedReader(path).get("one").values, values, atol=atol, rtol=0)
    if dtype != "float32":
        with pytest.raises(ValueError, match="conversion error"):
            with ActivationWriter(tmp_path / "limited", identity(dtype=dtype), width=4,
                                  max_quantization_error=0.0) as w:
                w.append(sample(), values)


def test_manifest_is_commit_barrier_and_orphans_ignored(tmp_path, monkeypatch):
    import saebase.store.shards as mod
    writer = ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=256)
    writer.append(sample(), np.array([[1, 2, 3, 4]], dtype=np.float32))
    real = mod._atomic_json
    def crash(path, value):
        if Path(path).name == "manifest.json" and value["shards"]:
            raise OSError("simulated crash after writing shard files")
        return real(path, value)
    with monkeypatch.context() as m:
        m.setattr(mod, "_atomic_json", crash)
        with pytest.raises(OSError, match="simulated crash"):
            writer.flush()
    writer.abort()
    assert ShardedReader(writer.path).sample_count == 0
    assert list((writer.path / "shards").glob("*.npy"))  # Unreferenced shard exists.
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=256) as resumed:
        assert resumed.append(sample(), np.array([[1, 2, 3, 4]], dtype=np.float32))
    reader = ShardedReader(writer.path)
    assert reader.sample_count == 1
    assert reader.manifest["shards"][0]["index"] > 0  # old orphan not overwritten


def test_corruption_detection_and_explicit_repair(tmp_path):
    r1 = sample("a", (1, 2, 3)); r2 = sample("b", (2, 3, 4))
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as w:
        w.append(r1, np.ones((1, 4), dtype=np.float32))
        w.flush()
        w.append(r2, np.ones((1, 4), dtype=np.float32) * 2)
        path = w.path
    reader = ShardedReader(path)
    corrupt = path / 'shards' / 'shard-00000.npy'
    raw = bytearray(corrupt.read_bytes())
    raw[-4:] = b'BADD'
    corrupt.write_bytes(raw)
    with pytest.raises(CacheIntegrityError, match="corrupted shard"):
        ShardedReader(path)
    with pytest.raises(CacheIntegrityError):
        ActivationWriter(tmp_path, identity(), width=4)
    with ActivationWriter(tmp_path, identity(), width=4, repair_corruption=True) as w:
        assert w.has(r2)
        assert not w.has(r1)
        w.append(r1, np.ones((1, 4), dtype=np.float32))
    after = ShardedReader(path)
    assert after.sample_count == 2
    assert len(after.manifest['repairs']) == 1
    np.testing.assert_allclose(after.get('a').values, np.ones((1, 4)))
    # A corrupted sidecar is also detected.
    sidecar = path / 'shards' / after.manifest['shards'][0]['sidecar']
    sidecar.write_text('BROKEN')
    with pytest.raises(CacheIntegrityError, match="sidecar"):
        ShardedReader(path)


def test_capture_multi_site_matches_existing_api_and_skips_repeats(tmp_path):
    class CountingModel(ResearchToyModel):
        def __init__(self):
            super().__init__()
            self.cache_calls = 0
        def run_with_cache(self, *args, **kwargs):
            self.cache_calls += 1
            return super().run_with_cache(*args, **kwargs)
    model = CountingModel().eval()
    adapter = ModelAdapter(model)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    actual = capture_selected(adapter, tokens=ids, site=site(), positions=(0, 3, 1))
    assert actual.shape == (3, 4) and model.cache_calls == 1
    for i, pos in enumerate((0, 3, 1)):
        np.testing.assert_allclose(actual[i], adapter.capture(tokens=ids, site=replace(site(), position=pos)))
    records = [sample('a', (1, 2, 3), selection=TokenSelection('all_valid')),
               sample('b', (1, 2, 3, 4))]
    with ActivationWriter(tmp_path, live_identity(model), width=4, shard_target_bytes=256) as w:
        assert collect_activations(adapter, records, w, site=site()).collected == 2
        path = w.path
    before = model.cache_calls
    with ActivationWriter(tmp_path, live_identity(model), width=4) as w:
        stats = collect_activations(adapter, records, w, site=site())
        assert stats.collected == 0 and stats.reused == 2
    assert model.cache_calls == before
    reader = ShardedReader(path)
    assert reader.get('a').positions == (0, 1, 2)
    assert reader.get('b').positions == (3,)


def test_hf_hooks_capture_multiple_positions_in_one_backend_call():
    class HFStyle(ResearchToyModel):
        _saebase_backend = 'hf_hooks'
        def __init__(self):
            super().__init__()
            self.calls = 0
        def get_resid(self, ids, layer, offsets):
            self.calls += 1
            assert layer == 0
            h = self._hidden(ids)
            return h[0, [ids.shape[1] - 1 - k for k in offsets]]
        def run_with_patch(self, *a, **k):
            return self(ids)
    adapter = ModelAdapter(HFStyle().eval())
    ids = torch.tensor([[1, 2, 3, 4]])
    out = capture_selected(adapter, tokens=ids, site=site(), positions=(2, 0))
    assert out.shape == (2, 4)
    assert adapter.model.calls == 1
    np.testing.assert_allclose(out, adapter.model._hidden(ids)[0, [2, 0]].detach())


def test_second_sae_uses_cached_rows_without_model_forward(tmp_path):
    class CountingModel(ResearchToyModel):
        def __init__(self):
            super().__init__()
            self.capture_calls = 0
        def run_with_cache(self, *a, **k):
            self.capture_calls += 1
            return super().run_with_cache(*a, **k)
    model = CountingModel().eval()
    adapter = ModelAdapter(model)
    records = [sample('one', (1, 2, 3)), sample('two', (4, 5, 6)),
               sample('three', (3, 3, 7), selection=TokenSelection('indices', (0, 2)))]
    with ActivationWriter(tmp_path, live_identity(model), width=4, shard_target_bytes=256) as w:
        collect_activations(adapter, records, w, site=site())
        raw_path = w.path
    forwards = model.capture_calls
    sae1 = ResearchToySAE().eval()
    sae2 = ResearchToySAE().eval()
    with torch.no_grad():
        sae2.bias.add_(0.5)
    code_path1, stats1 = encode_from_store(raw_path, sae1, replace(codes('A', 'c'), sae_state_sha256=sha256_state_dict(sae1)), layout='csr', max_rows=3,
                                           shard_target_bytes=256)
    code_path2, stats2 = encode_from_store(raw_path, sae2, replace(codes('B', 'd'), sae_state_sha256=sha256_state_dict(sae2)), layout='dense', max_rows=3,
                                           shard_target_bytes=256)
    assert model.capture_calls == forwards
    assert stats1.collected == 3 and stats2.collected == 3
    assert code_path1 != code_path2
    assert ShardedReader(code_path1).sample_count == ShardedReader(code_path2).sample_count == 3
    for r in records:
        cached = ShardedReader(raw_path).get(r.sample_id)
        expected = sae1.encode(torch.as_tensor(np.array(cached.values))).detach().numpy()
        np.testing.assert_array_equal(ShardedReader(code_path1).get(r.sample_id).values, expected)
        np.testing.assert_array_equal(ShardedReader(code_path2).get(r.sample_id).values, expected)
    new_path, stats = encode_from_store(raw_path, sae1, replace(codes('A', 'c'), sae_state_sha256=sha256_state_dict(sae1)), layout='csr',
                                        max_rows=3, shard_target_bytes=256)
    assert new_path == code_path1 and stats.collected == 0 and stats.reused == 3


def test_csr_exact_zeros_and_all_features_preserved(tmp_path):
    from saebase.store import CodeWriter
    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(sample(), np.ones((1, 4), dtype=np.float32))
        raw_path = w.path
    # d_sae=7; intentionally include a feature far beyond typical top-k.
    code = np.array([[0, 0, -3, 0, 1e-5, 0, 4]], dtype=np.float32)
    with CodeWriter(raw_path, codes(), layout='csr', shard_target_bytes=128) as w:
        w.append(sample(), code)
        path = w.path
    np.testing.assert_array_equal(ShardedReader(path).get('one').values, code)
    assert ShardedReader(path).manifest['layout'] == 'csr'
    assert len(list(path.glob('shards/*-indptr.npy'))) == 1


def test_mask_right_padding_supported_and_left_padding_rejected(tmp_path):
    adapter = ModelAdapter(ResearchToyModel().eval())
    good = sample('a', (1,2,3,0,0), mask=(1,1,1,0,0))
    with ActivationWriter(tmp_path, live_identity(adapter.model), width=4) as w:
        collect_activations(adapter, [good], w, site=site())
        np.testing.assert_allclose(ShardedReader(w.path).get('a').values[0],
                                   adapter.capture(tokens=torch.tensor([[1,2,3,0,0]]),
                                                   site=replace(site(), position=2)))
    with ActivationWriter(tmp_path, live_identity(adapter.model), width=4) as w:
        bad = sample('b', (0,0,1,2,3), mask=(0,0,1,1,1))
        with pytest.raises(ValueError, match="right-side padding"):
            collect_activations(adapter, [bad], w, site=site())


def test_memory_map_corruption_is_checked_even_when_initial_verification_deferred(tmp_path):
    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(sample(), np.ones((1,4), dtype=np.float32))
        path=w.path
    lazy = ShardedReader(path, verify_shards=False)
    p = path / 'shards/shard-00000.npy'
    p.write_bytes(p.read_bytes()+b'X')
    with pytest.raises(CacheIntegrityError, match="corrupted"):
        lazy.get('one')


def test_dense_and_sparse_code_layouts_can_coexist(tmp_path):
    from saebase.store import CodeWriter
    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(sample(), np.ones((1, 4), dtype=np.float32))
        raw_path = w.path
    dense_codes = np.array([[1, 0, 0, 3, 0, 0, 0]], dtype=np.float32)
    with CodeWriter(raw_path, codes(), layout="dense") as writer:
        writer.append(sample(), dense_codes)
        dense_path = writer.path
    with CodeWriter(raw_path, codes(), layout="csr") as writer:
        writer.append(sample(), dense_codes)
        sparse_path = writer.path
    assert dense_path != sparse_path
    np.testing.assert_array_equal(ShardedReader(dense_path).get("one").values,
                                  ShardedReader(sparse_path).get("one").values)


def test_frozen_m0_reference_fixture_is_actual_verified_public_checkpoint_data():
    reference = json.loads((Path(__file__).parent / "fixtures/m0_reference_public.json").read_text())
    assert reference["schema"] == "saebase.m0.reference.v1"
    assert reference["real_checkpoint_validated"] is True
    assert reference["sample_count"] == 3
    assert reference["model_state_sha256"] == "dc595ff7ad9263a3ef34ce8975beadfe76b7271916c63a7a67b8a195d7e50a12"
    assert reference["sae_state_sha256"] == "cde8b7c669860ec0dbd4b8ede465ff08868b317fc80010dac2703bc553c6585f"


def test_collect_refuses_incorrect_model_weight_fingerprint(tmp_path):
    adapter = ModelAdapter(ResearchToyModel().eval())
    with ActivationWriter(tmp_path, identity(), width=4) as writer:
        with pytest.raises(ValueError, match="model state SHA-256"):
            collect_activations(adapter, [sample()], writer, site=site())


def test_encoding_refuses_different_checkpoint_using_existing_identity(tmp_path):
    adapter = ModelAdapter(ResearchToyModel().eval())
    with ActivationWriter(tmp_path, live_identity(adapter.model), width=4) as writer:
        collect_activations(adapter, [sample()], writer, site=site())
        path = writer.path
    sae = ResearchToySAE().eval()
    with pytest.raises(ValueError, match="SAE state SHA-256"):
        encode_from_store(path, sae, codes(), shard_target_bytes=256)


def test_code_cache_detects_recaptured_activation_content_drift(tmp_path):
    r = sample()
    original = np.array([[1, 2, 3, 4]], dtype=np.float32)
    sae = ResearchToySAE().eval()
    codeid = replace(codes(), sae_state_sha256=sha256_state_dict(sae))
    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(r, original)
        raw_path = w.path
    codepath, _ = encode_from_store(raw_path, sae, codeid, shard_target_bytes=256)
    assert ShardedReader(codepath).sample_count == 1
    # Corrupt the original activation file and explicitly repair the shard.
    target = raw_path / "shards/shard-00000.npy"
    with target.open("ab") as f:
        f.write(b"broken")
    with ActivationWriter(tmp_path, identity(), width=4, repair_corruption=True) as w:
        w.append(r, original + 1)
    with pytest.raises(CacheIntegrityError, match="differs|changed"):
        encode_from_store(raw_path, sae, codeid, shard_target_bytes=256)


def test_m0_and_m1_state_hashes_agree_for_model_and_sae():
    from scripts.m0_reference_smoke import _sha_weights
    model = ResearchToyModel()
    sae = ResearchToySAE()
    assert sha256_state_dict(model) == _sha_weights(model)
    assert sha256_state_dict(sae) == _sha_weights(sae)
