"""Per-task dataset policy for CUB_200 (see README §3.2)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from dlbase.cfg.contracts import assert_task_in

_ALLOWED: FrozenSet[str] = frozenset({"classification"})


def assert_task_supported(cfg: DictConfig) -> None:
    """Raise if ``cfg.task.name`` is not supported for this dataset."""
    assert_task_in(cfg, _ALLOWED, "CUB_200")
