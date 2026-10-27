"""Post-training evaluation and ``summary.log`` helpers for the classification task."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tasks.vision.classification.classification_module import _get_output

# Final best weights when undertrained rename runs; also preferred by ``resolve_best_checkpoint_path``.
BEST_CHECKPOINT_FILENAME = "best.ckpt"


@dataclass
class SplitMetrics:
    n_samples: int
    loss: float
    acc_top1: float
    acc_top5: float

    def as_dict(self, num_classes: int) -> Dict[str, Any]:
        max_k = min(5, num_classes)
        return {
            "samples": self.n_samples,
            "cross_entropy_loss": self.loss,
            "top1_acc": self.acc_top1,
            f"top{max_k}_acc": self.acc_top5,
        }


@dataclass
class MultiLabelSplitMetrics:
    n_samples: int
    loss: float
    f1_micro: float
    f1_macro: float
    mean_ap: float

    def as_dict(self, num_classes: int) -> Dict[str, Any]:
        return {
            "samples": self.n_samples,
            "bce_loss": self.loss,
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "mAP": self.mean_ap,
        }


def count_module_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def evaluate_classification_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    multi_label: bool = False,
) -> "SplitMetrics | MultiLabelSplitMetrics":
    model.eval()
    if multi_label:
        return _evaluate_multi_label(model, loader, device, num_classes)
    return _evaluate_single_label(model, loader, device, num_classes)


def _evaluate_single_label(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> SplitMetrics:
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            y = y.long()
            out = model(x)
            logits = _get_output(out)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            pred = logits.argmax(dim=-1)
            top1_correct += int((pred == y).sum().item())
            max_k = min(5, num_classes)
            if max_k >= 1:
                _, pred_k = torch.topk(logits, max_k, dim=-1, largest=True, sorted=True)
                top5_correct += int((pred_k == y.unsqueeze(-1)).any(dim=-1).sum().item())
            n_total += int(y.size(0))
    if n_total == 0:
        return SplitMetrics(0, 0.0, 0.0, 0.0)
    return SplitMetrics(
        n_samples=n_total,
        loss=total_loss / n_total,
        acc_top1=top1_correct / n_total,
        acc_top5=top5_correct / n_total,
    )


def _evaluate_multi_label(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.5,
) -> MultiLabelSplitMetrics:
    total_loss = 0.0
    n_total = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device).float()
            out = model(x)
            logits = _get_output(out)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            total_loss += float(loss.item())
            n_total += int(y.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    if n_total == 0:
        return MultiLabelSplitMetrics(0, 0.0, 0.0, 0.0, 0.0)

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    probs = logits_cat.sigmoid()
    preds = (probs > threshold).float()

    # Micro F1
    tp = (preds * labels_cat).sum()
    fp = (preds * (1 - labels_cat)).sum()
    fn = ((1 - preds) * labels_cat).sum()
    p_micro = tp / (tp + fp + 1e-8)
    r_micro = tp / (tp + fn + 1e-8)
    f1_micro = float(2 * p_micro * r_micro / (p_micro + r_micro + 1e-8))

    # Macro F1 (per-class, then average)
    f1_per_class = []
    for c in range(num_classes):
        tp_c = (preds[:, c] * labels_cat[:, c]).sum()
        fp_c = (preds[:, c] * (1 - labels_cat[:, c])).sum()
        fn_c = ((1 - preds[:, c]) * labels_cat[:, c]).sum()
        p_c = tp_c / (tp_c + fp_c + 1e-8)
        r_c = tp_c / (tp_c + fn_c + 1e-8)
        f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-8)
        f1_per_class.append(float(f1_c))
    f1_macro = sum(f1_per_class) / max(1, len(f1_per_class))

    # Mean Average Precision (per-class AP, then mean)
    ap_per_class = []
    for c in range(num_classes):
        gt = labels_cat[:, c]
        sc = probs[:, c]
        if gt.sum() == 0:
            continue
        sorted_idx = sc.argsort(descending=True)
        gt_sorted = gt[sorted_idx]
        tp_cumsum = gt_sorted.cumsum(0)
        precision_at_k = tp_cumsum / torch.arange(1, len(gt_sorted) + 1, dtype=torch.float32)
        ap = float((precision_at_k * gt_sorted).sum() / gt.sum())
        ap_per_class.append(ap)
    mean_ap = sum(ap_per_class) / max(1, len(ap_per_class))

    return MultiLabelSplitMetrics(
        n_samples=n_total,
        loss=total_loss / n_total,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        mean_ap=mean_ap,
    )


def describe_classification_input_for_summary(cfg: DictConfig) -> str:
    """
    Human-readable shape of tensors passed to the *inner* model after
    ``ClassificationModule`` preprocessing (matches training / post_train eval).
    """
    h = int(cfg.dataset.image_size)
    input_format = getattr(cfg.dataset, "input_format", "image")
    in_channels = int(getattr(cfg.dataset, "in_channels", 3))
    is_transformer = "transformer" in str(cfg.model.name)

    if is_transformer:
        if input_format == "panels":
            return (
                f"16 x {h} x {h} (panel stack [B,16,H,W]; transformer path, no panels_to_rgb_grid)"
            )
        return f"{in_channels} x {h} x {h} (image, CHW after transforms)"

    if input_format == "panels":
        gh, gw = 4 * h, 4 * h
        return f"3 x {gh} x {gw} (RGB 4×4 grid from 16×{h}×{h} panels via panels_to_rgb_grid, CHW)"
    return f"{in_channels} x {h} x {h} (image, CHW after transforms)"


def classification_eval_splits(
    m_tr: "SplitMetrics | MultiLabelSplitMetrics",
    m_va: "SplitMetrics | MultiLabelSplitMetrics",
    m_te: "SplitMetrics | MultiLabelSplitMetrics",
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """Task-level eval payload for generic summary writer in ``dlbase``."""
    return {
        "train": m_tr.as_dict(num_classes),
        "val": m_va.as_dict(num_classes),
        "test": m_te.as_dict(num_classes),
    }


def classification_eval_train_test_only(
    m_tr: "SplitMetrics | MultiLabelSplitMetrics",
    m_te: "SplitMetrics | MultiLabelSplitMetrics",
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    """当 Lightning val 与 test 为同一数据（如 CUB legacy ``val_split_fraction=0``）时，summary 宽表仅保留 train/test。"""
    return {
        "train": m_tr.as_dict(num_classes),
        "test": m_te.as_dict(num_classes),
    }


def resolve_best_checkpoint_path(cwd: str, mc_best_path: Optional[str]) -> Optional[str]:
    named = os.path.join(cwd, "checkpoints", BEST_CHECKPOINT_FILENAME)
    if os.path.isfile(named):
        return named
    if mc_best_path and os.path.isfile(mc_best_path):
        return mc_best_path
    return None
