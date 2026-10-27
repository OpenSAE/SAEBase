"""Preflight checks for the Hydra ``sae`` evaluation task."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.datasets.artifact_layout import canonicalize_behavior
from saebase.datasets.prompt_schema import assert_available
from saebase.datasets.prompt_store import load_manifest
from saebase.registry import (
    get_model_plugin,
    get_prompt_source_plugin,
    get_sae_plugin,
    get_task_plugin,
)
from saebase.sae.contracts import assert_manifest_matches
from saebase.sae.registry import get_profile


def _prompt_source_name(cfg: DictConfig) -> str:
    if "prompt_source" in cfg and "name" in cfg.prompt_source:
        return str(cfg.prompt_source.name)
    return str(cfg.dataset.name)


def run_preflight(cfg: DictConfig) -> dict:
    """Validate registry gates + prompt artifact identity."""
    get_task_plugin(str(cfg.task.name))
    ps_name = _prompt_source_name(cfg)
    get_prompt_source_plugin(ps_name).assert_compatible(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)
    sae_plugin = get_sae_plugin(str(cfg.sae.name))
    sae_plugin.assert_compatible(cfg)

    behavior = canonicalize_behavior(ps_name)
    model_key = str(cfg.model.name)
    sae_key = sae_plugin.profile_key
    assert_available(behavior, model_key)

    profile = get_profile(sae_key)
    if model_key not in profile.allowed_model_keys():
        raise RuntimeError(
            f"sae.name={sae_key!r} allows models {sorted(profile.allowed_model_keys())}, "
            f"but model.name={model_key!r}"
        )

    require = True
    if "evaluation" in cfg and "require_prepared_prompts" in cfg.evaluation:
        require = bool(cfg.evaluation.require_prepared_prompts)
    elif "sae" in cfg and "require_prepared_prompts" in cfg.sae:
        require = bool(cfg.sae.require_prepared_prompts)

    if not require:
        raise ValueError(
            "activation_health in Phase A requires prepared prompts and a manifest; "
            "run saebase/tools/sae_prepare_prompts.py first"
        )

    data_root = cfg.paths.get("data_dir") if "paths" in cfg else None
    manifest = load_manifest(
        behavior,
        model_key,
        data_root=str(data_root) if data_root is not None else None,
    )
    assert_manifest_matches(manifest, behavior=behavior, model_key=model_key)

    layer = None
    if "sae" in cfg:
        layer = cfg.sae.get("layer")
    if layer is None and "evaluation" in cfg:
        layer = cfg.evaluation.get("layer")
    if layer is None:
        layer = profile.layers[len(profile.layers) // 2]
    else:
        layer = int(layer)
    if layer not in profile.layers:
        raise ValueError(
            f"layer={layer} not in selected layers for {sae_key}: {list(profile.layers)}"
        )

    if model_key == "pythia410m" and profile.hook_kind != "hook_mlp_out":
        raise RuntimeError("pythia410m must use hook_mlp_out SAE sites")

    return {
        "behavior": behavior,
        "model_key": model_key,
        "sae_key": profile.key,
        "layer": layer,
        "manifest": manifest,
        "profile": profile,
        "hook_name": profile.hook_name(layer),
        "evaluation": str(cfg.task.get("evaluation", "activation_health")),
    }
