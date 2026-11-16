"""Ordered loading policies preserving existing precision and weight processing."""

from __future__ import annotations

import torch


def execution_context_window(config) -> int | None:
    """Use at most 4096 tokens, bounded by the model's native text configuration."""
    from collections.abc import Mapping

    def value(node, key):
        return node.get(key) if isinstance(node, Mapping) else getattr(node, key, None)

    text = value(config, "text_config")
    if text is not None:
        config = text
    for key in ("max_position_embeddings", "n_positions", "n_ctx"):
        limit = value(config, key)
        if type(limit) is int and limit > 0:
            return min(limit, 4096)
    return None


def resolve_device(device: str | None = None) -> str:
    """Prefer available CUDA for automatic placement; preserve explicit placement."""
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return str(torch.device(device))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid execution device: {device!r}") from exc


def resolve_config_devices(cfg) -> str:
    """Bind Task and Model placement before validation, records or resource loading."""
    from omegaconf import OmegaConf, open_dict

    model = cfg.get("model")
    models = model.get("members", [model]) if OmegaConf.is_dict(model) else []
    if not isinstance(models, (list, tuple)) and not OmegaConf.is_list(models):
        models = []
    nodes = [node for node in [cfg.get("task"), *models] if OmegaConf.is_dict(node)]
    explicit = {
        resolve_device(node.device)
        for node in nodes
        if node.get("device") not in (None, "", "auto")
    }
    if any(device.startswith("cuda:") for device in explicit):
        explicit.discard("cuda")
    if len(explicit) > 1:
        raise ValueError(f"Task and Model execution devices conflict: {sorted(explicit)}")
    chosen = next(iter(explicit)) if explicit else resolve_device()
    for node in nodes:
        with open_dict(node):
            node.device = chosen
    return chosen


# First match wins, including historical architecture aliases and mirror IDs.
# Patterns preserve the existing public loader's acceptance of model-name aliases.
FAMILY_POLICIES = (
    (("gemma-3-1b",), {"no_processing": True, "dtype": torch.float32}),
    (("qwen2.5-1.5b", "deepseek-r1-distill-qwen-1.5b"), {"dtype": torch.float32}),
    (("gemma-2-2b",), {"dtype": torch.float16}),
    (
        (
            "gemma-3-4b",
            "gemma-3-12b",
            "gemma-3-27b",
            "gemma-2-9b",
            "llama-3.1-8b",
            "deepseek-r1-distill-llama-8b",
            "qwen3",
            "qwen3.5",
            "mistral-7b",
        ),
        {"no_processing": True, "dtype": torch.bfloat16},
    ),
)


def default_load_kwargs(model_name: str) -> dict:
    name = model_name.lower()
    for patterns, options in FAMILY_POLICIES:
        if any(pattern in name for pattern in patterns):
            return dict(options)
    return {}
