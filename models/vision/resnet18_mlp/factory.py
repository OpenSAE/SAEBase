"""ResNet-18 model-specific configuration and instantiation.

All ResNet head selection and legacy CUB handling belong to this concrete
model implementation, not the shared models/plugins.py registration layer.
"""
from __future__ import annotations

from omegaconf import DictConfig


def construct(cfg: DictConfig):
    from models.vision.resnet18_mlp.resnet18_mlp import Resnet18MLP

    raw = str(cfg.model.get("head_style", "auto")).lower().strip()
    head_style = raw if raw in ("auto", "linear", "mlp") else "auto"
    if head_style == "auto":
        # Backward-compatible selection for this specific pre-existing model.
        dataset = str(cfg.dataset.name)
        head_style = "linear" if dataset in ("CUB_200", "CUB") else "mlp"

    return Resnet18MLP(
        cfg.model.output_size,
        in_channels=int(cfg.model.get("in_channels", 3)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        small_input=bool(cfg.model.get("small_input", False)),
        head_style=head_style,
    )
