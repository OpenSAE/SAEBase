"""Exercise product behavior and explicitly requested real-checkpoint numerical regressions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from saebase.api import (
    ActivationSite,
    ModelAdapter,
    load_registered,  # noqa: E402
)
from saebase.store import (  # noqa: E402
    ActivationIdentity,
    ActivationWriter,
    CacheIntegrityError,
    CodeIdentity,
    SampleRecord,
    ShardedReader,
    TokenSelection,
    capture_selected,
    collect_activations,
    encode_from_store,
    fingerprint,
    resolve_positions,
    sha256_state_dict,
)
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE


def identity(**changes):
    return replace(
        ActivationIdentity(
            model_key="toy",
            model_state_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            hook_name="blocks.0.hook_resid_post",
            layer=0,
        ),
        **changes,
    )


def sample(
    name="one", tokens=(1, 2, 3), *, selection=None, mask=None, offsets=None, split="evaluation"
):
    return SampleRecord(
        name,
        tokens,
        attention_mask=mask,
        selection=selection or TokenSelection(),
        token_offsets=offsets,
        prompt_group_id="g1",
        split=split,
    )


def site():
    return ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post")


def live_identity(model):
    return identity(model_state_sha256=sha256_state_dict(model))


def codes(key="sae-a", digest="c"):
    return CodeIdentity(sae_key=key, sae_state_sha256=digest * 64, d_in=4, d_sae=7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_state_sha256", "c" * 64),
        ("tokenizer_sha256", "d" * 64),
        ("chat_template_sha256", "e" * 64),
        ("hook_name", "blocks.0.hook_mlp_out"),
        ("layer", 1),
        ("dtype", "float16"),
        ("preprocessing", {"bos": False}),
        ("execution", {"backend": "tl"}),
    ],
)
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
    assert resolve_positions(
        sample(tokens=tokens, mask=(1, 1, 0, 0), selection=TokenSelection("all_valid"))
    ) == (0, 1)
    assert resolve_positions(
        sample(tokens=tokens, selection=TokenSelection("indices", (2, 1)))
    ) == (2, 1)
    assert resolve_positions(
        sample(
            tokens=tokens,
            selection=TokenSelection("span", char_span=(3, 8)),
            offsets=((0, 0), (0, 4), (4, 7), (0, 0)),
        )
    ) == (1, 2)
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
    b = np.array([[11.0, 12.0, 13.0, 14.0]], dtype=np.float32)
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
    rows = [sample(f"row-{i:04d}", (i, i + 1)) for i in range(112)]
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


@pytest.mark.parametrize("dtype,atol", [("float32", 0), ("float16", 0.003), ("bfloat16", 0.03)])
def test_numeric_storage_precisions_and_error_bounds(tmp_path, dtype, atol):
    values = np.array([[1.111, -2.222, 0.14159, 3.333]], dtype=np.float32)
    with ActivationWriter(
        tmp_path, identity(dtype=dtype), width=4, shard_target_bytes=256
    ) as writer:
        writer.append(sample(), values)
        assert writer.max_observed_quantization_error <= atol
        path = writer.path
    np.testing.assert_allclose(ShardedReader(path).get("one").values, values, atol=atol, rtol=0)
    if dtype != "float32":
        with pytest.raises(ValueError, match="conversion error"):
            with ActivationWriter(
                tmp_path / "limited", identity(dtype=dtype), width=4, max_quantization_error=0.0
            ) as w:
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
    r1 = sample("a", (1, 2, 3))
    r2 = sample("b", (2, 3, 4))
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as w:
        w.append(r1, np.ones((1, 4), dtype=np.float32))
        w.flush()
        w.append(r2, np.ones((1, 4), dtype=np.float32) * 2)
        path = w.path
    ShardedReader(path)
    corrupt = path / "shards" / "shard-00000.npy"
    raw = bytearray(corrupt.read_bytes())
    raw[-4:] = b"BADD"
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
    assert len(after.manifest["repairs"]) == 1
    np.testing.assert_allclose(after.get("a").values, np.ones((1, 4)))
    # A corrupted sidecar is also detected.
    sidecar = path / "shards" / after.manifest["shards"][0]["sidecar"]
    sidecar.write_text("BROKEN")
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
        np.testing.assert_allclose(
            actual[i], adapter.capture(tokens=ids, site=replace(site(), position=pos))
        )
    records = [
        sample("a", (1, 2, 3), selection=TokenSelection("all_valid")),
        sample("b", (1, 2, 3, 4)),
    ]
    with ActivationWriter(tmp_path, live_identity(model), width=4, shard_target_bytes=256) as w:
        assert collect_activations(adapter, records, w, site=site()).collected == 2
        path = w.path
    before = model.cache_calls
    with ActivationWriter(tmp_path, live_identity(model), width=4) as w:
        stats = collect_activations(adapter, records, w, site=site())
        assert stats.collected == 0 and stats.reused == 2
    assert model.cache_calls == before
    reader = ShardedReader(path)
    assert reader.get("a").positions == (0, 1, 2)
    assert reader.get("b").positions == (3,)


def test_hf_hooks_capture_multiple_positions_in_one_backend_call():
    class HFStyle(ResearchToyModel):
        _saebase_backend = "hf_hooks"

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
    records = [
        sample("one", (1, 2, 3)),
        sample("two", (4, 5, 6)),
        sample("three", (3, 3, 7), selection=TokenSelection("indices", (0, 2))),
    ]
    with ActivationWriter(tmp_path, live_identity(model), width=4, shard_target_bytes=256) as w:
        collect_activations(adapter, records, w, site=site())
        raw_path = w.path
    forwards = model.capture_calls
    sae1 = ResearchToySAE().eval()
    sae2 = ResearchToySAE().eval()
    with torch.no_grad():
        sae2.bias.add_(0.5)
    code_path1, stats1 = encode_from_store(
        raw_path,
        sae1,
        replace(codes("A", "c"), sae_state_sha256=sha256_state_dict(sae1)),
        layout="csr",
        max_rows=3,
        shard_target_bytes=256,
    )
    code_path2, stats2 = encode_from_store(
        raw_path,
        sae2,
        replace(codes("B", "d"), sae_state_sha256=sha256_state_dict(sae2)),
        layout="dense",
        max_rows=3,
        shard_target_bytes=256,
    )
    assert model.capture_calls == forwards
    assert stats1.collected == 3 and stats2.collected == 3
    assert code_path1 != code_path2
    assert ShardedReader(code_path1).sample_count == ShardedReader(code_path2).sample_count == 3
    for r in records:
        cached = ShardedReader(raw_path).get(r.sample_id)
        expected = sae1.encode(torch.as_tensor(np.array(cached.values))).detach().numpy()
        np.testing.assert_array_equal(ShardedReader(code_path1).get(r.sample_id).values, expected)
        np.testing.assert_array_equal(ShardedReader(code_path2).get(r.sample_id).values, expected)
    new_path, stats = encode_from_store(
        raw_path,
        sae1,
        replace(codes("A", "c"), sae_state_sha256=sha256_state_dict(sae1)),
        layout="csr",
        max_rows=3,
        shard_target_bytes=256,
    )
    assert new_path == code_path1 and stats.collected == 0 and stats.reused == 3


def test_csr_exact_zeros_and_all_features_preserved(tmp_path):
    from saebase.store import CodeWriter

    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(sample(), np.ones((1, 4), dtype=np.float32))
        raw_path = w.path
    # d_sae=7; intentionally include a feature far beyond typical top-k.
    code = np.array([[0, 0, -3, 0, 1e-5, 0, 4]], dtype=np.float32)
    with CodeWriter(raw_path, codes(), layout="csr", shard_target_bytes=128) as w:
        w.append(sample(), code)
        path = w.path
    np.testing.assert_array_equal(ShardedReader(path).get("one").values, code)
    assert ShardedReader(path).manifest["layout"] == "csr"
    assert len(list(path.glob("shards/*-indptr.npy"))) == 1


def test_mask_right_padding_supported_and_left_padding_rejected(tmp_path):
    adapter = ModelAdapter(ResearchToyModel().eval())
    good = sample("a", (1, 2, 3, 0, 0), mask=(1, 1, 1, 0, 0))
    with ActivationWriter(tmp_path, live_identity(adapter.model), width=4) as w:
        collect_activations(adapter, [good], w, site=site())
        np.testing.assert_allclose(
            ShardedReader(w.path).get("a").values[0],
            adapter.capture(
                tokens=torch.tensor([[1, 2, 3, 0, 0]]), site=replace(site(), position=2)
            ),
        )
    with ActivationWriter(tmp_path, live_identity(adapter.model), width=4) as w:
        bad = sample("b", (0, 0, 1, 2, 3), mask=(0, 0, 1, 1, 1))
        with pytest.raises(ValueError, match="right-side padding"):
            collect_activations(adapter, [bad], w, site=site())


def test_memory_map_corruption_is_checked_even_when_initial_verification_deferred(tmp_path):
    with ActivationWriter(tmp_path, identity(), width=4) as w:
        w.append(sample(), np.ones((1, 4), dtype=np.float32))
        path = w.path
    lazy = ShardedReader(path, verify_shards=False)
    p = path / "shards/shard-00000.npy"
    p.write_bytes(p.read_bytes() + b"X")
    with pytest.raises(CacheIntegrityError, match="corrupted"):
        lazy.get("one")


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
    np.testing.assert_array_equal(
        ShardedReader(dense_path).get("one").values, ShardedReader(sparse_path).get("one").values
    )


def test_frozen_m0_reference_fixture_is_actual_verified_public_checkpoint_data():
    reference = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/m0_reference_public.json").read_text()
    )
    assert reference["schema"] == "saebase.m0.reference.v1"
    assert reference["real_checkpoint_validated"] is True
    assert reference["sample_count"] == 3
    assert (
        reference["model_state_sha256"]
        == "dc595ff7ad9263a3ef34ce8975beadfe76b7271916c63a7a67b8a195d7e50a12"
    )
    assert (
        reference["sae_state_sha256"]
        == "cde8b7c669860ec0dbd4b8ede465ff08868b317fc80010dac2703bc553c6585f"
    )


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


def test_state_hash_matches_fixed_serialization_reference():
    module = torch.nn.Module()
    module.register_buffer("probe", torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert (
        sha256_state_dict(module)
        == "1b49ad52928e70d9c6f14a64741f3aa8c133b5888ad1015717536dd0769c7e66"
    )


def test_state_hash_is_strict_for_inplace_and_data_edits():
    model = ResearchToyModel()
    first = sha256_state_dict(model)
    assert sha256_state_dict(model) == first
    with torch.no_grad():
        parameter = next(model.parameters())
        parameter.add_(1)
    changed = sha256_state_dict(model)
    assert changed != first
    with torch.no_grad():
        parameter.data.add_(1)
    assert sha256_state_dict(model) != changed


def test_many_shards_use_incremental_metadata_index(tmp_path, monkeypatch):
    import saebase.store.shards as mod

    original = mod.ShardedReader
    calls = []

    class CountingReader(original):
        def __init__(self, *a, **k):
            calls.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(mod, "ShardedReader", CountingReader)
    with ActivationWriter(tmp_path, identity(), width=4, shard_target_bytes=128) as w:
        for idx in range(190):
            w.append(sample(f"item-{idx:05d}", (idx, idx + 1)), np.full((1, 4), idx, np.float32))
        path = w.path
    assert len(calls) == 1, "writer should not reindex every prior shard after each flush"
    assert original(path).sample_count == 190
    assert len(original(path).manifest["shards"]) >= 20


def run_real_checkpoint(reference: Path, cache_root: Path, output: Path) -> dict:
    reference_record = json.loads(reference.read_text(encoding="utf-8"))
    if (
        not reference_record.get("real_checkpoint_validated")
        or reference_record.get("status") != "passed"
    ):
        raise ValueError("Input must be a successful real-checkpoint reference")
    bundle = load_registered(
        model_key="pythia70m", sae_key="pythia70m_resid_sm", layer=3, device="cpu"
    )
    model_sha = sha256_state_dict(bundle.model.model)
    sae_sha = sha256_state_dict(bundle.sae)
    if (
        model_sha != reference_record["model_state_sha256"]
        or sae_sha != reference_record["sae_state_sha256"]
    ):
        raise RuntimeError(
            "Loaded model or SAE does not match actual reference weight fingerprints"
        )
    site = ActivationSite(layer=3, hook_name="blocks.3.hook_resid_post")
    if bundle.site.layer != site.layer or bundle.site.hook_name != site.hook_name:
        raise RuntimeError("Loaded checkpoint is incompatible with reference site")
    tokenizer = bundle.model.model.tokenizer
    token_data = {
        "vocab": tokenizer.get_vocab(),
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    tokenizer_sha = fingerprint(token_data)
    cache_id = ActivationIdentity(
        model_key="pythia70m",
        model_state_sha256=model_sha,
        tokenizer_sha256=tokenizer_sha,
        hook_name=site.hook_name,
        layer=site.layer,
        # Keep this provenance label stable: it contributes to saved cache identities.
        preprocessing={"source": "M0 SimpleCopying token IDs", "selection": "last_valid"},
        execution={"model_backend": bundle.model.backend, "run_mode": "eval", "device": "cpu"},
    )
    records = [
        SampleRecord(
            sample_id=row["sample_id"],
            token_ids=tuple(row["token_ids"]),
            selection=TokenSelection("last_valid"),
            split="m0-reference",
            prompt_group_id=row["sample_id"],
        )
        for row in reference_record["samples"]
    ]
    if not records:
        raise RuntimeError("Checkpoint reference contains no sample rows")
    # Include two explicitly selected positions to validate multi-position capture.
    first = records[0]
    records.append(
        SampleRecord(
            sample_id="m1-multi-position",
            token_ids=first.token_ids,
            selection=TokenSelection("indices", (0, len(first.token_ids) - 1)),
            split="m1-extra",
            prompt_group_id=first.sample_id,
        )
    )
    with ActivationWriter(cache_root, cache_id, width=512, shard_target_bytes=4096) as writer:
        stats = collect_activations(bundle.model, records, writer, site=site)
        raw_path = writer.path
    if stats.collected != len(records):
        raise RuntimeError("Activation cache run did not collect the requested activations")
    raw = ShardedReader(raw_path)
    max_probe_err = 0.0
    for reference_row in reference_record["samples"]:
        stored = raw.get(reference_row["sample_id"])
        assert stored.positions == (len(reference_row["token_ids"]) - 1,)
        probe = np.asarray(stored.values[0, : len(reference_row["activation_probe"])])
        err = float(np.max(np.abs(probe - np.array(reference_row["activation_probe"]))))
        max_probe_err = max(max_probe_err, err)
        if not np.allclose(probe, reference_row["activation_probe"], atol=1e-4, rtol=1e-4):
            raise RuntimeError("ActivationStore does not reproduce reference activation probe")
    full_tokens = torch.tensor([list(first.token_ids)], dtype=torch.long)
    multi = raw.get("m1-multi-position")
    assert len(multi.positions) == 2
    for row, position in zip(multi.values, multi.positions):
        expected = bundle.model.capture(
            tokens=full_tokens, site=ActivationSite(site.layer, site.hook_name, position)
        )
        if not np.allclose(row, expected.detach().cpu().numpy(), atol=1e-5, rtol=1e-5):
            raise RuntimeError("Multi-position capture differs from the reference adapter")

    # All requested samples already in the store: no model forward is required.
    def no_model_calls(*args, **kwargs):
        raise AssertionError("resuming an existing activation cache must not run the model")

    original_capture = bundle.model.model.run_with_cache
    bundle.model.model.run_with_cache = no_model_calls
    try:
        with ActivationWriter(cache_root, cache_id, width=512) as writer:
            resumed = collect_activations(bundle.model, records, writer, site=site)
            if resumed.collected != 0 or resumed.reused != len(records):
                raise RuntimeError("activation resume did not reuse every committed sample")
        cident = CodeIdentity(
            sae_key="pythia70m_resid_sm",
            sae_state_sha256=sae_sha,
            d_in=512,
            d_sae=32768,
            inference_mode="per_token",
        )
        code_path, encoded = encode_from_store(
            raw_path, bundle.sae, cident, layout="csr", max_rows=16, shard_target_bytes=256 * 1024
        )
        code_path2, repeat = encode_from_store(
            raw_path, bundle.sae, cident, layout="csr", max_rows=16, shard_target_bytes=256 * 1024
        )
    finally:
        bundle.model.model.run_with_cache = original_capture
    if (
        code_path != code_path2
        or encoded.collected != len(records)
        or repeat.reused != len(records)
    ):
        raise RuntimeError("SAE code cache did not resume independently from model activations")
    encoded_cache = ShardedReader(code_path)
    nonzero = []
    for reference_row in reference_record["samples"]:
        actual = encoded_cache.get(reference_row["sample_id"]).values[0]
        nz = int(np.count_nonzero(actual))
        nonzero.append(nz)
        if (
            nz != int(reference_row["codes_l0"])
            or int(np.argmax(np.abs(actual))) != reference_row["selected_feature"]
        ):
            raise RuntimeError(
                "SAE code cache disagrees with real checkpoint reference feature activations"
            )
    report = {
        "schema": "saebase.m1.real_cache.v1",
        "real_checkpoint_validated": True,
        "m0_reference_verified": True,
        "status": "passed",
        "model_state_sha256": model_sha,
        "sae_state_sha256": sae_sha,
        "activation_key": cache_id.key,
        "sae_code_key": cident.key,
        "sample_count": raw.sample_count,
        "row_count": raw.row_count,
        "code_sample_count": encoded_cache.sample_count,
        "activation_shards": len(raw.manifest["shards"]),
        "code_shards": len(encoded_cache.manifest["shards"]),
        "max_m0_activation_probe_abs_error": max_probe_err,
        "model_forward_calls_during_resume_and_sae_encoding": 0,
        "resumed_samples": resumed.reused,
        "encoded_samples": encoded.collected,
        "repeat_code_cache_reused": repeat.reused,
        "code_l0_for_m0_samples": nonzero,
        "cache_root": "Local activation/code cache artifact (not committed to source)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "[activation/code cache] Real activation cache, checkpoint reference, multi-position capture and SAE reuse passed"
    )
    print(json.dumps(report, indent=2))
    return report


def real_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", "--m0-reference", dest="reference", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_real_checkpoint(args.reference, args.cache_root, args.output)


if __name__ == "__main__":
    raise SystemExit(real_main())
