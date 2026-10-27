"""CUB-200 Lightning datamodule wiring (registered via ``research_datasets.plugins``)."""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import random_split

from research_datasets.vision.CUB_200 import CUB200


def _cub_transforms(image_size: int, train: bool):
    """ImageNet-style preprocessing for pretrained ResNet."""
    _ = image_size  # Keep signature stable for existing call sites.
    # 与细粒度分类参考配方对齐：Resize 512 → Crop 448
    if train:
        return T.Compose(
            [
                T.Resize((512, 512)),
                T.RandomCrop(448, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((512, 512)),
            T.CenterCrop(448),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def setup_cub_datamodule(dm: Any, cfg: DictConfig, stage: Any = None) -> None:
    """Populate ``dm`` train/val/test sets for CUB (``dataset.name`` CUB_200 or CUB alias)."""
    ds = cfg.dataset
    sz = ds.image_size
    ratio_cfg = ds.get("train_val_ratio")
    force_eval = bool(ds.get("ssi_force_eval_transforms", False))
    full_train = CUB200(
        data_path=ds.data_path,
        train=True,
        transforms=_cub_transforms(sz, train=not force_eval),
    )
    test_ds = CUB200(
        data_path=ds.data_path,
        train=False,
        transforms=_cub_transforms(sz, train=False),
    )
    dm.test_set = test_ds
    n = len(full_train)
    use_ratio = ratio_cfg is not None and len(list(ratio_cfg)) == 2
    if use_ratio:
        p0, p1 = int(ratio_cfg[0]), int(ratio_cfg[1])
        if p0 <= 0 or p1 <= 0:
            raise ValueError(f"dataset.train_val_ratio must be positive pairs, got [{p0}, {p1}]")
        n_val = int(round(n * (float(p1) / float(p0 + p1))))
        n_val = max(1, min(n_val, n - 1))
        n_train = n - n_val
        dm._cub_legacy_val_is_test = False
        gen = torch.Generator().manual_seed(int(cfg.seed))
        dm.train_set, dm.val_set = random_split(full_train, [n_train, n_val], generator=gen)
    else:
        frac = float(ds.get("val_split_fraction", 0.1))
        if frac <= 0.0:
            dm.train_set = full_train
            dm.val_set = test_ds
            dm._cub_legacy_val_is_test = True
        else:
            dm._cub_legacy_val_is_test = False
            n_val = int(round(n * frac))
            n_val = max(1, min(n_val, n - 1))
            n_train = n - n_val
            gen = torch.Generator().manual_seed(int(cfg.seed))
            dm.train_set, dm.val_set = random_split(full_train, [n_train, n_val], generator=gen)


def cub_eval_split_note(dm: Any) -> str:
    """Match legacy ``DLBaseDataModule.eval_split_note`` for CUB."""
    if dm._cub_legacy_val_is_test:
        return (
            "note: val_split_fraction=0 — Lightning val set is the official CUB test split; "
            "val and test tables evaluate the same images."
        )
    r = dm.cfg.dataset.get("train_val_ratio")
    if r is not None and len(list(r)) == 2:
        return (
            "note: CUB — train/val from official train only "
            f"(train_val_ratio={list(r)}); test = official test ({len(dm.test_set)} imgs)."
        )
    return (
        "note: train is a subset of official CUB training images; "
        "val is the held-out fraction; test is the official CUB test split."
    )
