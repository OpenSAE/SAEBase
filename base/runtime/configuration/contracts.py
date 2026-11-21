"""Generic cfg-level assertions (no hard-coded dataset / model / task names)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig


def assert_task_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    t = str(cfg.task.name)
    if t not in allowed:
        raise ValueError(f"{context}: unsupported task.name={t!r} (allowed: {sorted(allowed)})")


def assert_dataset_name_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    ds = str(cfg.dataset.name)
    if ds not in allowed:
        raise ValueError(f"{context}: unsupported dataset.name={ds!r}")


def assert_prompt_source_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    # Support temporary alias ``dataset`` during migration of call sites.
    if "prompt_source" in cfg and "name" in cfg.prompt_source:
        name = str(cfg.prompt_source.name)
    elif "dataset" in cfg and "name" in cfg.dataset:
        name = str(cfg.dataset.name)
    else:
        raise ValueError(f"{context}: missing prompt_source.name")
    if name not in allowed:
        raise ValueError(f"{context}: unsupported prompt_source.name={name!r}")


def assert_model_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    name = str(cfg.model.name)
    if name not in allowed:
        raise ValueError(f"{context}: unsupported model.name={name!r}")


def assert_sae_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    name = str(cfg.sae.name)
    if name not in allowed:
        raise ValueError(f"{context}: unsupported sae.name={name!r}")
