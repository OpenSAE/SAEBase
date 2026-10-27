"""Classification pre-training checks: registry gates + logits dry-run (README §3.2).

Prefer fixing ``dataset_task_policy.py`` / ``model_task_dataset_compatibility.py`` over expanding
this module's orchestration.
"""

from __future__ import annotations

from typing import FrozenSet, Union

import torch
from omegaconf import DictConfig

from dlbase.cfg.contracts import assert_task_in
from dlbase.registry import get_dataset_plugin, get_model_plugin

_CLASSIFICATION_ONLY: FrozenSet[str] = frozenset({"classification"})


def assert_classification_task(cfg: DictConfig, context: str) -> None:
    assert_task_in(cfg, _CLASSIFICATION_ONLY, context)


def run_final_pre_training_checks(cfg: DictConfig) -> None:
    """Dataset task policy → model compatibility → logits shape dry-run (fail-fast before DataModule)."""
    get_dataset_plugin(str(cfg.dataset.name)).assert_task_supported(cfg)
    get_model_plugin(str(cfg.model.name)).assert_compatible(cfg)
    classification_logits_shape_dry_run(cfg)


def classification_logits_shape_dry_run(cfg: DictConfig) -> None:
    """Mirror ``ClassificationModule.forward`` preprocessing; assert logits ``[B, num_classes]``."""
    if str(cfg.task.name) != "classification":
        return

    from models.build import construct_model

    def _get_output(outputs: Union[torch.Tensor, tuple]) -> torch.Tensor:
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    net = construct_model(cfg)
    fmt = str(getattr(cfg.dataset, "input_format", "image"))
    nc = int(cfg.dataset.num_classes)
    sz = int(cfg.dataset.image_size)
    in_channels = int(getattr(cfg.dataset, "in_channels", 3))
    is_transformer = "transformer" in str(cfg.model.name)

    if fmt == "panels":
        raise RuntimeError("this snapshot only supports image inputs")
    x = torch.zeros(2, in_channels, sz, sz)

    epoch = 0
    if is_transformer:
        out = net(x, epoch)
    else:
        out = net(x, epoch)

    logits = _get_output(out)
    if logits.dim() != 2 or logits.shape[-1] != nc:
        raise RuntimeError(
            f"classification pipeline check: expected logits shape [B, {nc}], got {tuple(logits.shape)}"
        )
