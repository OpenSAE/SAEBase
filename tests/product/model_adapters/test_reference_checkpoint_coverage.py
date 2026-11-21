"""Phase A metadata contracts; no downloads and no claims about live weights."""

from __future__ import annotations

import pytest

from saebase.api.sae_profiles import get_profile
from tasks.precheck.sae_checks import TARGETS

EXPECTED = {
    "gemma270m": ("gemma270m", "gemma270m_scope_resid", "sae_lens", 5),
    "gemma1b": ("gemma1b", "gemma1b_scope_resid", "sae_lens", 13),
    "gemma2_2b": ("gemma2_2b_base", "gemma2_2b_pt_l25_canonical", "sae_lens", 25),
    "qwen35_2b": ("qwen35_2b", "qwen35_2b_resid_w32k", "qwen_resid_topk", 12),
    "qwen3_8b": ("qwen3_8b", "qwen3_8b_resid_w64k", "qwen_resid_topk", 18),
    "llama31_8b": ("llama31_8b", "llama31_8b_lxr_8x", "sae_lens", 16),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_phase_a_profiles_and_host_model_specs(name):
    model_key, sae_key, loader, layer = EXPECTED[name]
    expected_key, expected_layer, width = TARGETS[name]
    assert (expected_key, expected_layer) == (sae_key, layer)
    assert width > 0
    profile = get_profile(sae_key)
    assert profile.model_key == model_key
    assert profile.loader == loader
    assert profile.repo_id and "/" in profile.repo_id
    assert profile.hook_name(layer) == f"blocks.{layer}.hook_resid_post"
    assert profile.sae_id(layer)
    from models.plugins import get_model_adapter

    module = get_model_adapter(model_key)
    assert module.spec.key == model_key
    assert module.spec.model_name
    assert module.spec.family


def test_qwen35_2b_public_checkpoint_metadata():
    profile = get_profile("qwen35_2b_resid_w32k")
    assert profile.sae_release == "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
    assert profile.sae_id(12) == "layer12.sae.pt"
    assert profile.base.model_name == "Qwen/Qwen3.5-2B-Base"
    assert profile.layers == (6, 12, 15, 20)


def test_gemma_scope_2_release_aliases():
    assert get_profile("gemma270m_scope_resid").sae_release == "gemma-scope-2-270m-pt-resid_post"
    assert get_profile("gemma1b_scope_resid").sae_release == "gemma-scope-2-1b-pt-resid_post"


def test_phase_a_requires_real_weights_for_evidence():
    from tasks.precheck.sae_checks import verify_target

    with pytest.raises(KeyError):
        verify_target("imaginary", device="cpu")


@pytest.mark.parametrize(
    ("profile_key", "layer", "expected_repo", "expected_path", "converter"),
    [
        (
            "gemma270m_scope_resid",
            5,
            "google/gemma-scope-2-270m-pt",
            "resid_post/layer_5_width_65k_l0_medium",
            "gemma_3_sae_huggingface_loader",
        ),
        (
            "gemma1b_scope_resid",
            13,
            "google/gemma-scope-2-1b-pt",
            "resid_post/layer_13_width_16k_l0_medium",
            "gemma_3_sae_huggingface_loader",
        ),
        (
            "llama31_8b_lxr_8x",
            16,
            "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
            "Llama3_1-8B-Base-L16R-8x",
            "llama_scope_sae_huggingface_loader",
        ),
    ],
)
def test_exact_native_sae_source_paths(profile_key, layer, expected_repo, expected_path, converter):
    from saebase.api.sae_loading import _phase_a_sae_lens_source

    assert _phase_a_sae_lens_source(get_profile(profile_key), layer) == (
        expected_repo,
        expected_path,
        converter,
    )


def test_pair_measurement_preserves_identity_and_observes_feature_change():
    import torch

    from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
    from tasks.precheck.sae_checks import measure_pair
    from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE

    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post", position=-1)
    prompt = "1 2 3 4"
    snap = capture_features(model, sae, site=site, prompt=prompt)
    baseline = model.forward_logits(tokens=snap.tokens).detach().float()
    selected = int(snap.codes.abs().argmax().item())
    edited = edit_features(sae, snap, replacements={selected: 0.0})
    observed = (
        model.patch_logits(tokens=snap.tokens, site=site, activation=edited.activation)
        .detach()
        .float()
    )
    rows = measure_pair(model, sae, site, [prompt])
    assert rows[0]["identity_patch_max_abs_logit_error"] == 0
    assert rows[0]["ablated_feature_id"] == selected
    assert rows[0]["feature_ablation_logit_delta_l2"] == pytest.approx(
        float(torch.linalg.vector_norm(baseline - observed))
    )


def test_pair_cli_preserves_missing_asset_reason_and_recovery(tmp_path, monkeypatch):
    import json

    from tasks.precheck import sae_checks

    def missing(*args, **kwargs):
        raise FileNotFoundError("selected local SAE checkpoint is missing")

    monkeypatch.setattr(sae_checks, "verify_target", missing)
    assert (
        sae_checks.main(["--model", "qwen35_2b", "--device", "cpu", "--output", str(tmp_path)]) == 1
    )
    result = json.loads((tmp_path / "qwen35_2b.json").read_text())
    assert result["status"] == "blocked"
    assert "selected local SAE checkpoint" in result["reason"]
    assert result["repair"] and result["stage"] == "model_sae_pair:qwen35_2b"
    assert result["real_weights_validated"] is False
