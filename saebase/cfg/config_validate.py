"""Hydra config validation for the tasks included with SAEBase Phase A."""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from saebase.registry import get_model_plugin, get_prompt_source_plugin, get_sae_plugin, get_task_plugin


def validate_common_cfg(cfg: DictConfig) -> None:
    for group in ("prompt_source", "model", "sae", "task", "paths", "evaluation"):
        if group not in cfg:
            raise ValueError(f"Missing config group: {group}")
    for group in ("prompt_source", "model", "sae", "task"):
        if not OmegaConf.is_config(cfg[group]) or "name" not in cfg[group]:
            raise ValueError(f"cfg.{group} must be a config node with 'name'")
    if "seed" not in cfg:
        raise ValueError("Missing cfg.seed")
    get_prompt_source_plugin(str(cfg.prompt_source.name))
    get_model_plugin(str(cfg.model.name))
    if str(cfg.task.name) != "sae":  # legacy standalone CLI only
        get_task_plugin(str(cfg.task.name))
    get_sae_plugin(str(cfg.sae.name))


def validate_hydra_cfg(cfg: DictConfig) -> None:
    validate_common_cfg(cfg)
    if str(cfg.task.name) != "sae":
        raise ValueError("Phase A task registry supports task=sae only")
    if str(cfg.task.get("evaluation")) != "activation_health":
        raise ValueError("task=sae requires task.evaluation=activation_health")
    if str(cfg.evaluation.get("name")) != "activation_health":
        raise ValueError("task=sae requires evaluation=activation_health")
    if str(cfg.sae.name) == "none":
        raise ValueError("task=sae requires an actual SAE profile")
    get_sae_plugin(str(cfg.sae.name)).assert_compatible(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)
    get_prompt_source_plugin(str(cfg.prompt_source.name)).assert_compatible(cfg)
