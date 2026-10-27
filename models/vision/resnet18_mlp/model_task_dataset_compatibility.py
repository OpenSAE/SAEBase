"""task × dataset compatibility for ``resnet18_mlp`` (see README §3.2)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from dlbase.cfg.contracts import assert_dataset_name_in, assert_task_in

_RESNET_ALLOWED_DATASETS: FrozenSet[str] = frozenset({"CUB_200", "CUB"})
_CLASSIFICATION: FrozenSet[str] = frozenset({"classification"})


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _CLASSIFICATION, "resnet18_mlp")
    assert_dataset_name_in(cfg, _RESNET_ALLOWED_DATASETS, "resnet18_mlp")
