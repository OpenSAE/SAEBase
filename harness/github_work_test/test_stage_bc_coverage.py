"""Stage B/C 15-model / 48-checkpoint offline coverage and adapter contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from saebase.sae.coverage import load_coverage
from saebase.sae.registry import get_profile
from saebase.sae.sparsify_adapter import SparsifyDenseAdapter
from saebase.sae.loading import _phase_a_sae_lens_source

CATALOG = Path(__file__).resolve().parents[2] / "conf" / "coverage" / "stage_bc.json"


def test_multilayer_matrix_and_independent_architecture_extras():
    catalog = load_coverage(CATALOG)
    summary = catalog.plan()
    assert summary["kind"] == "registry_plan_not_real_checkpoint_evidence"
    assert summary["core_models"] == 15
    assert summary["extra_architectures"] == 3
    assert summary["planned_checkpoints"] == 48
    assert summary["planned_cpu_sae_probes"] == 21
    assert summary["planned_cpu_model_pair_probes"] == 3
    assert len(catalog.entries) == 18
    assert all(e.depth_bands() for e in catalog.entries)
    assert all(
        s["verification"] == "registered_only" for e in summary["entries"] for s in e["checkpoints"]
    )


def test_official_qwen_base_not_confused_with_posttrained_chat():
    from models.qwen3_1_7b_base.qwen3_1_7b_base import SPEC

    p = get_profile("qwen3_1_7b_base_resid_w32k")
    assert SPEC.model_name == "Qwen/Qwen3-1.7B-Base"
    assert p.model_key == "qwen3_1_7b_base"
    assert p.loader == "qwen_resid_topk"
    assert p.sae_id(14) == "layer14.sae.pt"
    assert p.repo_id == "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    assert get_profile("qwen3_1_7b_ak_resid_l14_65k").model_key == "qwen3_1_7b"


def test_mistral_source_site_and_qwen3_9b_bands():
    m = load_coverage(CATALOG).find("mistral7b_instruct_tc_l16_131k")
    assert m.selected_layers == (16,) and m.cpu_probe_layers == ()
    source = m.source(16)
    assert source["hook_name"] == "blocks.16.hook_mlp_out"
    assert source["repo_id"] == "tylercosgrove/mistral-7b-sparse-autoencoder-layer16"
    assert source["verification"] == "registered_only"
    q = load_coverage(CATALOG).find("qwen35_9b_resid_w64k")
    assert [v["layer"] for v in q.depth_bands().values()] == [8, 16, 21, 27]


def test_gemma_4b_and_r1_native_hf_converters():
    a = get_profile("gemma3_4b_pt_scope_resid")
    b = get_profile("gemma3_4b_it_scope_resid")
    assert _phase_a_sae_lens_source(a, 17) == (
        "google/gemma-scope-2-4b-pt",
        "resid_post/layer_17_width_65k_l0_medium",
        "gemma_3_sae_huggingface_loader",
    )
    assert _phase_a_sae_lens_source(b, 17)[0] == "google/gemma-scope-2-4b-it"
    r = get_profile("r1_distill_llama8b_llamascope_l23")
    assert _phase_a_sae_lens_source(r, 23)[1] == ("400M-Slimpajama-400M-OpenR1-Math-220k/L23R")


def test_sparsify_and_batchtopk_routing():
    assert get_profile("r1_distill_qwen15b_resid_l20").loader == "sparsify_topk"
    assert get_profile("qwen3_1_7b_ak_resid_l14_65k").loader == "dictionary_learning"


class NativeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_in = 3
        self.num_latents = 5
        self.cfg = SimpleNamespace(
            k=2,
            transcode=False,
            skip_connection=False,
            activation="relu",
        )
        self.encoder = nn.Linear(3, 5, bias=True)
        self.W_dec = nn.Parameter(torch.randn(5, 3))
        self.b_dec = nn.Parameter(torch.randn(3))

    def encode(self, x):
        raw = torch.relu(self.encoder(x - self.b_dec))
        vals, indices = torch.topk(raw, k=2, dim=-1)
        return vals, indices, raw


def test_native_sparsify_dense_bridge_uses_same_original_topk_values():
    torch.manual_seed(33)
    native = NativeToy()
    bridge = SparsifyDenseAdapter(native, hook_name="blocks.20.hook_resid_post")
    x = torch.randn(4, 3)
    codes = bridge.encode(x)
    vals, ind, _ = native.encode(x)
    expected = torch.zeros_like(codes).scatter_(1, ind, vals)
    assert torch.allclose(codes, expected, rtol=0, atol=0)
    assert torch.allclose(bridge.decode(codes), expected @ native.W_dec + native.b_dec)
    assert bridge.cfg.d_in == 3 and bridge.cfg.d_sae == 5
    with pytest.raises(ValueError, match="input"):
        bridge.encode(torch.ones((2, 4)))
    with pytest.raises(ValueError, match="decoder"):
        bridge.decode(torch.ones((2, 4)))
    edited = codes.detach().clone()
    edited[0, 4] = -4.0
    assert torch.allclose(bridge.decode(edited)[0], edited[0] @ native.W_dec + native.b_dec)


def test_catalog_rejects_incomplete_or_incorrect_real_source_mapping(tmp_path):
    data = __import__("json").loads(CATALOG.read_text())
    data["targets"][0]["selected_layers"] = [5, 5]
    path = tmp_path / "bad.json"
    path.write_text(__import__("json").dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_coverage(path)
    data["targets"][0]["selected_layers"] = [200]
    path.write_text(__import__("json").dumps(data))
    with pytest.raises(ValueError, match="layers"):
        load_coverage(path)
    data["targets"][0]["selected_layers"] = [5]
    data["targets"][0]["cpu_probe_layers"] = [9]
    path.write_text(__import__("json").dumps(data))
    with pytest.raises(ValueError, match="subset"):
        load_coverage(path)


def test_registry_bootstrap_has_new_models_and_sae_profiles():
    from saebase.registry import list_registered_model_keys, list_registered_sae_keys

    models = list_registered_model_keys()
    saes = list_registered_sae_keys()
    for key in ("mistral7b_instruct", "qwen3_1_7b_base"):
        assert key in models
    for key in ("mistral7b_instruct_tc_l16_131k", "qwen3_1_7b_base_resid_w32k"):
        assert key in saes


def test_matryoshka_real_source_paths_are_concrete():
    from saebase.tools.phase_bc_source_metadata import probe_file_path

    catalog = load_coverage(CATALOG)
    a = catalog.find("gemma2_2b_it_matryoshka_l12_32k")
    b = catalog.find("qwen35_4b_matryoshka_l15_65k")
    assert probe_file_path(a, 12) == ("standard/blocks.12.hook_resid_post/sae_weights.safetensors")
    assert probe_file_path(b, 15) == ("qwen-3.5-4b/btk-mat-layer-15-k-100/sae_weights.safetensors")
