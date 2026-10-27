from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from models.build import build_model
from dlbase.training.schedulers import build_warmup_cosine


def _get_output(outputs: Union[torch.Tensor, Tuple[Any, ...]]) -> torch.Tensor:
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _single_label_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean()


def _multi_label_f1_micro(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Micro-averaged F1 for multi-label: treats all (sample, class) pairs as one pool."""
    preds = (logits.sigmoid() > threshold).float()
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


class ClassificationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.model = build_model(cfg)
        self.is_transformer = "transformer" in cfg.model.name
        self._input_format = getattr(cfg.dataset, "input_format", "image")
        self._multi_label = bool(getattr(cfg.dataset, "multi_label", False))
        self._label_smoothing = float(
            OmegaConf.select(cfg, "train.loss.label_smoothing", default=0.0) or 0.0
        )
        self._run_loss_sum = 0.0
        self._run_acc_sum = 0.0
        self._run_count = 0

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        epoch = self.current_epoch
        if self.is_transformer:
            return self.model(x, epoch)
        if self._input_format == "panels":
            raise RuntimeError("this snapshot only supports image inputs")
        return self.model(x, epoch)

    def on_train_epoch_start(self) -> None:
        self._run_loss_sum = 0.0
        self._run_acc_sum = 0.0
        self._run_count = 0

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch[0], batch[1]
        out = self(x)
        logits = _get_output(out)

        if self._multi_label:
            y = y.float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            acc = _multi_label_f1_micro(logits, y)
        else:
            y = y.long()
            loss = F.cross_entropy(logits, y, label_smoothing=self._label_smoothing)
            acc = _single_label_acc(logits, y)

        bs = int(x.size(0))
        if stage == "train":
            lv, av = float(loss.detach()), float(acc.detach())
            self._run_loss_sum += lv * bs
            self._run_acc_sum += av * bs
            self._run_count += bs
            loss_ep = self._run_loss_sum / max(1, self._run_count)
            acc_ep = self._run_acc_sum / max(1, self._run_count)
            self.log(
                "train/loss",
                loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
            self.log(
                "train/acc",
                acc,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
            self.log(
                "train/loss_bar",
                loss_ep,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=bs,
                rank_zero_only=True,
            )
            self.log(
                "train/acc_bar",
                acc_ep,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=bs,
                rank_zero_only=True,
            )
        else:
            self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
            self.log(f"{stage}/acc", acc, prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        o = self.cfg.train.optimizer
        name = str(o.name)
        p2_lr = self.cfg.train.get("phase2_constant_lr")
        if p2_lr is not None:
            lr = float(p2_lr)
            rank_zero_info(f"Phase 2: constant LR={lr} (no scheduler).")
            if name == "AdamW":
                return torch.optim.AdamW(
                    self.parameters(),
                    lr=lr,
                    weight_decay=float(o.weight_decay),
                    betas=tuple(o.betas),
                    eps=float(o.eps),
                )
            if name == "SGD":
                return torch.optim.SGD(
                    self.parameters(),
                    lr=lr,
                    momentum=float(o.get("momentum", 0.9)),
                    weight_decay=float(o.weight_decay),
                    nesterov=bool(o.get("nesterov", False)),
                )
            raise ValueError(f"Phase 2 unsupported optimizer.name: {name}")

        if name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(o.lr),
                weight_decay=float(o.weight_decay),
                betas=tuple(o.betas),
                eps=float(o.eps),
            )
        elif name == "SGD":
            param_groups = self._sgd_param_groups(o)
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=float(o.get("momentum", 0.9)),
                weight_decay=float(o.weight_decay),
                nesterov=bool(o.get("nesterov", False)),
            )
        else:
            raise ValueError(f"Unsupported optimizer.name: {name}")

        sch_cfg = self.cfg.train.lr_scheduler
        max_epochs = int(self.cfg.train.max_epochs)
        sched_name = str(sch_cfg.get("name", "warmup_cosine"))
        if sched_name == "cosine_annealing":
            t_max = int(sch_cfg.get("t_max", max_epochs))
            eta_min = float(sch_cfg.get("eta_min", 0.0))
            rank_zero_info(
                f"LR schedule: CosineAnnealingLR T_max={t_max}, eta_min={eta_min}, max_epochs={max_epochs}"
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else:
            warmup_epochs = max(5, int(0.1 * max_epochs))
            rank_zero_info(
                f"LR schedule: warmup_cosine warmup_epochs={warmup_epochs}, max_epochs={max_epochs}"
            )
            scheduler = build_warmup_cosine(
                optimizer,
                max_epochs=max_epochs,
                warmup_epochs=warmup_epochs,
                min_lr_ratio=float(sch_cfg.min_lr_ratio),
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _sgd_param_groups(self, o: Any) -> List[Dict[str, Any]]:
        """Split backbone vs head for differentiated LR（``resnet18_mlp``：linear 用 ``resnet18.fc``，mlp 用 ``mlp.*``；``resnet50_mlp``：``resnet50.fc``）。"""
        backbone_lr = float(o.lr)
        head_lr = float(o.get("head_lr", backbone_lr))
        model_name = str(self.cfg.model.name)
        head_prefix = None
        if model_name == "resnet18_mlp":
            hm = getattr(self.model, "head_mode", "linear")
            head_prefix = "mlp." if hm == "mlp" else "resnet18.fc"
        elif model_name == "resnet50_mlp":
            head_prefix = "resnet50.fc"
        if head_prefix is not None and abs(head_lr - backbone_lr) > 1e-12:
            backbone_params: List[torch.nn.Parameter] = []
            head_params: List[torch.nn.Parameter] = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith(head_prefix):
                    head_params.append(p)
                else:
                    backbone_params.append(p)
            return [
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ]
        return [{"params": self.model.parameters(), "lr": backbone_lr, "name": "all"}]
