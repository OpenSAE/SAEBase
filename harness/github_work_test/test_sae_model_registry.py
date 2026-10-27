"""Update SAE model registry tests for P0 profiles."""

from __future__ import annotations

from models.gemma1b.gemma1b import SPEC as G1
from models.gemma270m.gemma270m import SPEC as G270
from models.pythia410m.pythia410m import SPEC as P410
from models.pythia70m.pythia70m import SPEC as P70
from omegaconf import OmegaConf

from models.r1_distill_qwen15b.model_task_dataset_compatibility import (
    assert_compatible as assert_r1_compatible,
)
from saebase.plugin_bootstrap import ensure_plugins_loaded
from saebase.registry import (
    list_registered_model_keys,
    list_registered_sae_keys,
    list_registered_task_keys,
)
from saebase.sae.registry import (
    DEFAULT_SAE_FOR_MODEL,
    MODEL_KEYS,
    SAE_KEYS,
    get_profile,
)


def test_model_keys_match_specs():
    assert "pythia70m" in MODEL_KEYS
    assert "gemma1b" in MODEL_KEYS
    assert "gemma1b_it" in MODEL_KEYS
    assert "r1_distill_qwen15b" in MODEL_KEYS
    assert "r1_distill_llama8b" in MODEL_KEYS
    assert "gemma2_2b_base" in MODEL_KEYS
    assert "gemma2_2b_it" in MODEL_KEYS
    assert "gemma3_4b_pt" in MODEL_KEYS
    assert "llama31_8b" in MODEL_KEYS
    assert "llama31_8b_instruct" in MODEL_KEYS
    assert "qwen3_8b" in MODEL_KEYS
    assert "gemma2_9b_pt" in MODEL_KEYS
    assert "qwen35_4b" in MODEL_KEYS
    assert "qwen35_9b" in MODEL_KEYS
    assert "gemma3_12b_it" in MODEL_KEYS
    assert "gemma3_27b_pt" in MODEL_KEYS
    assert "qwen35_27b" in MODEL_KEYS
    assert get_profile("pythia70m").model_name == P70.model_name
    assert get_profile("pythia410m").model_name == P410.model_name
    assert get_profile("gemma270m").model_name == G270.model_name
    assert get_profile("gemma1b").model_name == G1.model_name
    assert get_profile("r1_distill_qwen15b").model_name == (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    assert get_profile("r1_distill_llama8b").model_name == (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )


def test_default_sae_profiles():
    assert DEFAULT_SAE_FOR_MODEL["pythia70m"] == "pythia70m_resid_sm"
    assert get_profile("pythia410m").hook_kind == "hook_mlp_out"
    assert get_profile("gemma1b").sae_id(13) == "layer_13_width_16k_l0_medium"
    assert "gemma1b_pt_l25_transfer_big" in SAE_KEYS
    assert "gemma2_2b_pt_l25_canonical" in SAE_KEYS
    assert get_profile("gemma2_2b_pt_l25_canonical").transfer_kind == "base_to_instruct"
    assert 25 in get_profile("gemma1b_pt_l25_transfer_big").layers
    assert get_profile("gemma3_4b_pt").layers == (9, 17, 22, 29)
    assert get_profile("gemma3_4b_it_scope_resid").sae_id(17) == (
        "layer_17_width_65k_l0_medium"
    )
    assert DEFAULT_SAE_FOR_MODEL["llama31_8b"] == "llama31_8b_lxr_8x"
    assert DEFAULT_SAE_FOR_MODEL["llama31_8b_instruct"] == "llama31_8b_instruct_andyrdt_l19_k64"
    assert get_profile("llama31_8b_instruct_andyrdt_l19_k64").loader == "dictionary_learning"
    assert get_profile("llama31_8b_instruct_andyrdt_l19_k64").layers == (19,)
    assert get_profile("r1_distill_qwen15b_resid_l20").layers == (20,)
    assert get_profile("r1_distill_qwen15b").sae_id(20) == "resid_post_layer20"
    assert get_profile("r1_distill_qwen15b_resid_l20").loader == "sparsify_topk"
    assert get_profile("r1_distill_qwen15b_resid_l20").sae_folder(20) == (
        "resid_post_layer20"
    )
    assert DEFAULT_SAE_FOR_MODEL["r1_distill_qwen15b"] == "r1_distill_qwen15b_resid_l20"
    assert DEFAULT_SAE_FOR_MODEL["r1_distill_llama8b"] == (
        "r1_distill_llama8b_llamascope_l23"
    )
    assert get_profile("r1_distill_llama8b_llamascope_l23").layers == (23,)
    assert get_profile("r1_distill_llama8b_llamascope_l23").sae_id(23) == (
        "l23r_400m_slimpajama_400m_openr1_math"
    )
    assert get_profile("r1_distill_llama8b_llamascope_l23").loader == "sae_lens"
    assert get_profile("r1_distill_llama8b_llamascope_l23").sae_release == (
        "llama_scope_r1_distill"
    )
    assert get_profile("qwen3_8b_resid_w64k").loader == "qwen_resid_topk"
    assert get_profile("gemma2_9b_pt_scope_resid").sae_id(9) == (
        "layer_9/width_16k/average_l0_51"
    )
    assert get_profile("gemma2_9b_it_scope_resid").layers == (9, 20, 31)
    assert get_profile("qwen35_27b").sae_id(54) == "layer54.sae.pt"
    assert DEFAULT_SAE_FOR_MODEL["gemma3_27b_it"] == "gemma3_27b_it_scope_resid"


def test_hydra_compose_standalone_sae():
    import pytest
    pytest.importorskip("hydra")
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from saebase.cfg.config_validate import validate_hydra_cfg

    ensure_plugins_loaded()
    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf)):
        cfg = compose(config_name="config", overrides=[
            "task=sae", "evaluation=activation_health", "model=pythia70m",
            "sae=pythia70m_resid_sm", "prompt_source=simple_copying",
            "sae.layer=3", "paths.data_dir=/tmp/saebase-prepared",
        ])
    validate_hydra_cfg(cfg)
    assert cfg.task.name == "sae"
    assert cfg.prompt_source.name == "SimpleCopying"
    assert cfg.sae.layer == 3
    assert cfg.paths.data_dir == "/tmp/saebase-prepared"


def test_config_rejects_unbundled_research_task():
    import pytest
    from saebase.cfg.config_validate import validate_hydra_cfg
    cfg = OmegaConf.create({
        "prompt_source": {"name": "MMLUPro"},
        "model": {"name": "r1_distill_qwen15b"},
        "task": {"name": "discovery", "evaluation": "discovery"},
        "sae": {"name": "r1_distill_qwen15b_resid_l20"},
        "evaluation": {"name": "discovery"},
        "paths": {},
        "seed": 0,
    })
    with pytest.raises(ValueError, match="Unknown task"):
        validate_hydra_cfg(cfg)


def test_plugins_register_sae_stack():
    ensure_plugins_loaded()
    for k in MODEL_KEYS:
        assert k in list_registered_model_keys()
    for k in SAE_KEYS:
        assert k in list_registered_sae_keys()
    assert "none" in list_registered_sae_keys()
    assert "sae" in list_registered_task_keys()
    assert list_registered_task_keys() == sorted({"sae", "capture", "intervene", "generate", "circuits", "explain", "benchmark", "train"})


def test_r1_accepts_discovery_on_mmlu_pro():
    ensure_plugins_loaded()
    cfg = OmegaConf.create(
        {
            "task": {"name": "discovery"},
            "prompt_source": {"name": "MMLUPro"},
            "model": {"name": "r1_distill_qwen15b"},
        }
    )
    assert_r1_compatible(cfg)
    from saebase.registry import get_prompt_source_plugin

    get_prompt_source_plugin("MMLUPro").assert_compatible(cfg)


def test_unbundled_discovery_task_fails_registration():
    import pytest
    from saebase.registry import get_task_plugin
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_plugin("discovery")
