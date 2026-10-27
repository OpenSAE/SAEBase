"""Phase A metadata contracts; no downloads and no claims about live weights."""
from __future__ import annotations
import importlib

import pytest

from saebase.sae.registry import get_profile
from saebase.tools.phase_a_verify import TARGETS

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
    module = importlib.import_module(f"models.{model_key}.{model_key}")
    assert module.SPEC.key == model_key
    assert module.SPEC.model_name
    assert module.SPEC.family

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
    from saebase.tools.phase_a_verify import verify_target
    with pytest.raises(KeyError):
        verify_target("imaginary",device="cpu")

@pytest.mark.parametrize(("profile_key","layer","expected_repo","expected_path","converter"), [
    ("gemma270m_scope_resid",5,"google/gemma-scope-2-270m-pt",
     "resid_post/layer_5_width_65k_l0_medium","gemma_3_sae_huggingface_loader"),
    ("gemma1b_scope_resid",13,"google/gemma-scope-2-1b-pt",
     "resid_post/layer_13_width_16k_l0_medium","gemma_3_sae_huggingface_loader"),
    ("llama31_8b_lxr_8x",16,"OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
     "Llama3_1-8B-Base-L16R-8x","llama_scope_sae_huggingface_loader"),
])
def test_exact_native_sae_source_paths(profile_key,layer,expected_repo,expected_path,converter):
    from saebase.sae.loading import _phase_a_sae_lens_source
    assert _phase_a_sae_lens_source(get_profile(profile_key), layer) == (
        expected_repo, expected_path, converter
    )
