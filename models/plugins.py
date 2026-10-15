"""Register HookedTransformer model plugins and SAE profile plugins."""

from __future__ import annotations

import importlib

from omegaconf import DictConfig

from saebase.registry import register_model, register_sae
from saebase.sae.registry import DEFAULT_SAE_FOR_MODEL, SAE_KEYS, get_profile

_HOOKED_KEYS = (
    "gpt2small",
    "pythia70m",
    "pythia410m",
    "gemma270m",
    "gemma1b",
    "gemma1b_it",
    "r1_distill_qwen15b",
    "r1_distill_llama8b",
    "gemma2_2b_base",
    "gemma2_2b_it",
    "qwen25_3b",
    "qwen25_3b_it",
    "gemma3_4b_pt",
    "gemma3_4b_it",
    "llama31_8b",
    "llama31_8b_instruct",
    "qwen3_8b",
    "qwen3_1_7b",
    "qwen35_4b",
    "gemma2_9b_pt",
    "gemma2_9b_it",
    "gemma3_12b_pt",
    "gemma3_12b_it",
    "qwen35_9b",
    "qwen35_2b",
    "gemma3_27b_pt",
    "gemma3_27b_it",
    "qwen35_27b",
)


def _register_builtin_models() -> None:
    def _register_hooked(key: str) -> None:
        compat = importlib.import_module(
            f"models.{key}.model_task_dataset_compatibility"
        )
        mod = importlib.import_module(f"models.{key}.{key}")
        register_model(key, compat.assert_compatible, mod.construct_lazy)

    for hooked_key in _HOOKED_KEYS:
        _register_hooked(hooked_key)

    def _make_sae_assert(sae_key: str):
        def assert_compatible(cfg: DictConfig) -> None:
            profile = get_profile(sae_key)
            model_key = str(cfg.model.name)
            if model_key not in profile.allowed_model_keys():
                raise ValueError(
                    f"sae.name={sae_key!r} allows models "
                    f"{sorted(profile.allowed_model_keys())}, got {model_key!r}"
                )
            layer = cfg.sae.get("layer")
            if layer is not None and int(layer) not in profile.layers:
                raise ValueError(
                    f"sae.layer={layer} not in {sae_key} layers {list(profile.layers)}"
                )

        return assert_compatible

    for sae_key in SAE_KEYS:
        register_sae(sae_key, _make_sae_assert(sae_key), profile_key=sae_key)

    def _assert_none_sae(cfg: DictConfig) -> None:
        # Placeholder SAE key; task=sae rejects sae=none at config validation.
        return None

    register_sae("none", _assert_none_sae, profile_key="none")

    # Convenience: register bare model_key as SAE alias to its default profile.
    for model_key, sae_key in DEFAULT_SAE_FOR_MODEL.items():
        if model_key not in SAE_KEYS:
            register_sae(model_key, _make_sae_assert(sae_key), profile_key=sae_key)


_register_builtin_models()
