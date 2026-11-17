"""Validate common selections and delegate method-specific parameters to their task.

No checkpoint is loaded and no training parameter is required for unrelated tasks.
"""

from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
from saebase.registry import TaskModule, get_model_plugin, get_task_plugin, selected_models


def validate_hydra_cfg(cfg: DictConfig) -> TaskModule:
    from models.loading_policy import resolve_config_devices

    required = ("model", "dataset", "task", "paths", "experiment", "seed")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")
    for group in ("task",):
        if not OmegaConf.is_config(cfg[group]) or "name" not in cfg[group]:
            raise ValueError(f"cfg.{group} must contain a component name")
    from saebase.extensions.registry import external_model
    from tasks.sae_runtime import load_requested_extensions

    load_requested_extensions(cfg)

    plugin = get_task_plugin(str(cfg.task.name))
    task = plugin.task_class()
    models = selected_models(cfg)
    if len(models) > 1 and not task.supports_multiple_models:
        raise ValueError(f"Task {cfg.task.name} does not support multiple models")
    for model in models:
        if external_model(str(model.name)) is None:
            get_model_plugin(str(model.name))
    if (
        not OmegaConf.is_dict(cfg.dataset)
        or not isinstance(cfg.dataset.get("name"), str)
        or not cfg.dataset.name
    ):
        raise ValueError("cfg.dataset must contain a component name")
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if cfg.get("runtime", {}).get("numerics", "native") not in {"native", "vllm"}:
        raise ValueError("runtime.numerics must be native or vllm")
    threads = cfg.get("runtime", {}).get("cpu_threads")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("runtime.cpu_threads must be a positive integer")
    if "sample_count" in cfg.dataset:
        from saebase.datasets.selection import validate_selection

        validate_selection(cfg.dataset)
    effective = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolve_config_devices(effective)
    task.validate_cfg(effective)
    return task
