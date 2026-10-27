"""
Classification task: Lightning ``Trainer`` for CUB-200 (entry: ``tasks.runner.run_task``).

Run directory (``hydra.job.chdir``): ``experiment/<dataset>/<model>/<task>/<sub_name>_<seed>/``.

Hyperparameters come from ``conf/`` (``conf/train/default.yaml`` and ``conf/task/classification.yaml``).
This module does not overwrite them. Engineering stack (checkpoint / loggers / tqdm / TB / leaf cleanup)
comes from ``dlbase.training.standard_lightning`` and is toggled via ``train.facets``.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn

from datasets.datamodule import DLBaseDataModule
from dlbase.training.facets import merge_classification_task_facets, merge_train_facets
from dlbase.training.seed import apply_lightning_reproducibility
from dlbase.training.standard_lightning import (
    build_lightning_loggers,
    build_model_checkpoint_callbacks,
    early_stopping_patience_epochs,
    lightning_trainer_kwargs,
    maybe_launch_tensorboard_browser,
)
from dlbase.summary_log import build_eval_blocks_with_adapter, write_summary_log
from dlbase.experiment_paths import artifact_path
from dlbase.training.training_callbacks import (
    EpochEtaHoursCallback,
    ValLossHistoryAndPhaseLrCallback,
    tqdm_epoch_progress_bar,
    val_still_strictly_decreasing,
)
from tasks.vision.classification.classification_module import ClassificationModule
from tasks.vision.classification.pipeline_checks import run_final_pre_training_checks
from tasks.vision.classification.post_train_eval import (
    BEST_CHECKPOINT_FILENAME,
    classification_eval_splits,
    classification_eval_train_test_only,
    count_module_parameters,
    describe_classification_input_for_summary,
    evaluate_classification_loader,
    resolve_best_checkpoint_path,
)


def run_classification_training(cfg: DictConfig) -> None:
    train_f = merge_train_facets(cfg)
    task_f = merge_classification_task_facets(cfg)

    apply_lightning_reproducibility(int(cfg.seed), workers=True)
    train_start = datetime.now()

    run_final_pre_training_checks(cfg)

    dm = DLBaseDataModule(cfg)
    model = ClassificationModule(cfg)

    M = int(cfg.train.max_epochs)
    patience_epochs = early_stopping_patience_epochs(M)
    min_delta = float(cfg.train.early_stopping.min_delta)

    if train_f.lightning_loggers:
        loggers = build_lightning_loggers(cfg)
    else:
        loggers = False

    if train_f.early_stopping and cfg.train.early_stopping.enable:
        rank_zero_info(
            f"Early stopping: patience={patience_epochs} validation checks "
            f"(one check per epoch; equals max(5, int(0.1*max_epochs)), max_epochs={M})."
        )

    if train_f.model_checkpoint:
        ck_keep = bool(cfg.train.checkpoint.get("keep_intermediate", False))
        ck_every = max(1, int(cfg.train.checkpoint.get("keep_every_n_epochs", 10)))
        if ck_keep:
            rank_zero_info(
                f"Checkpoint: best → {BEST_CHECKPOINT_FILENAME} (top-1); "
                f"plus epoch_<epoch>.ckpt every {ck_every} epoch(s)."
            )
        else:
            rank_zero_info(
                f"Checkpoint: best only → {BEST_CHECKPOINT_FILENAME} (top-1; no intermediate epoch files)."
            )
    else:
        rank_zero_warn(
            "train.facets.model_checkpoint=false: no ModelCheckpoint; phase-2 reload and best.ckpt eval may be unavailable."
        )

    val_hist = ValLossHistoryAndPhaseLrCallback()
    cwd = os.getcwd()
    enable_pb = bool(train_f.tqdm_epoch_progress)

    callbacks_p1: List = []
    if train_f.tqdm_epoch_progress:
        callbacks_p1.append(tqdm_epoch_progress_bar(cfg))
    if train_f.epoch_eta_hours:
        callbacks_p1.append(EpochEtaHoursCallback(M))
    callbacks_p1.append(val_hist)
    if train_f.learning_rate_monitor:
        callbacks_p1.append(LearningRateMonitor(logging_interval="epoch"))

    mc: Any = None
    mc_extra: List = []
    if train_f.model_checkpoint:
        mc, mc_extra = build_model_checkpoint_callbacks(cfg, str(artifact_path(cwd, "checkpoints")))
        callbacks_p1.append(mc)
        callbacks_p1.extend(mc_extra)

    if train_f.early_stopping and cfg.train.early_stopping.enable:
        callbacks_p1.append(
            EarlyStopping(
                monitor=cfg.train.early_stopping.monitor,
                patience=patience_epochs,
                mode=cfg.train.early_stopping.mode,
                min_delta=min_delta,
            )
        )

    trainer1 = pl.Trainer(
        **lightning_trainer_kwargs(cfg, M, callbacks_p1, loggers, enable_progress_bar=enable_pb)
    )
    trainer1.fit(model, datamodule=dm)
    phase1_epochs = len(val_hist.val_losses)

    extend = (
        bool(task_f.phase_two_extension)
        and train_f.model_checkpoint
        and mc is not None
        and len(val_hist.val_losses) == M
        and val_hist.phase1_end_lr is not None
    )

    phase2_ran = False
    if extend:
        best_path = mc.best_model_path
        if not best_path or not os.path.isfile(best_path):
            rank_zero_warn("Phase 2 skipped: no valid checkpoint path from phase 1.")
        else:
            rank_zero_info(
                f"Phase 2: phase 1 finished all {M} epochs (early stopping did not apply). "
                f"Training {M} more epochs with LR fixed to {val_hist.phase1_end_lr} (end of phase 1)."
            )
            with open_dict(cfg):
                cfg.train.phase2_constant_lr = float(val_hist.phase1_end_lr)

            model_p2 = ClassificationModule.load_from_checkpoint(
                best_path,
                cfg=cfg,
            )
            callbacks_p2: List = []
            if train_f.tqdm_epoch_progress:
                callbacks_p2.append(tqdm_epoch_progress_bar(cfg))
            if train_f.epoch_eta_hours:
                callbacks_p2.append(EpochEtaHoursCallback(M))
            callbacks_p2.append(val_hist)
            if train_f.learning_rate_monitor:
                callbacks_p2.append(LearningRateMonitor(logging_interval="epoch"))
            if train_f.model_checkpoint:
                callbacks_p2.append(mc)
                callbacks_p2.extend(mc_extra)
            if train_f.early_stopping and cfg.train.early_stopping.enable:
                callbacks_p2.append(
                    EarlyStopping(
                        monitor=cfg.train.early_stopping.monitor,
                        patience=patience_epochs,
                        mode=cfg.train.early_stopping.mode,
                        min_delta=min_delta,
                    )
                )

            trainer2 = pl.Trainer(
                **lightning_trainer_kwargs(cfg, M, callbacks_p2, loggers, enable_progress_bar=enable_pb)
            )
            trainer2.fit(model_p2, datamodule=dm)
            phase2_ran = True

    undertrained = (
        phase2_ran
        and len(val_hist.val_losses) == 2 * M
        and val_still_strictly_decreasing(val_hist.val_losses, min_delta, window=3)
    )
    undertrained_warn = False
    if undertrained and train_f.model_checkpoint and mc is not None and mc.best_model_path and os.path.isfile(mc.best_model_path):
        dst = os.path.join(os.getcwd(), BEST_CHECKPOINT_FILENAME)
        if os.path.normpath(mc.best_model_path) != os.path.normpath(dst):
            os.replace(mc.best_model_path, dst)
        undertrained_warn = True
        rank_zero_warn(
            f"Val loss 仍在连续下降，训练可能尚未充分；已将全局最优 val loss 对应权重保存为 {BEST_CHECKPOINT_FILENAME} "
            f"（总训练上限 2×max_epochs={2 * M} 已用尽；原先的 best.ckpt 已由该全局最优覆盖）。"
        )

    train_end = datetime.now()
    best_ckpt_path_for_eval = None
    if train_f.model_checkpoint and mc is not None:
        best_ckpt_path_for_eval = resolve_best_checkpoint_path(os.getcwd(), mc.best_model_path)

    tb_url: Optional[str] = None
    if train_f.tensorboard_auto_launch_after_fit:
        tb_url = maybe_launch_tensorboard_browser(os.getcwd(), cfg)

    if train_f.summary_log and task_f.post_train_eval_and_summary:
        input_shape_desc = describe_classification_input_for_summary(cfg)
        num_classes = int(cfg.dataset.num_classes)
        is_multi_label = bool(getattr(cfg.dataset, "multi_label", False))
        best_ckpt = best_ckpt_path_for_eval
        eval_blocks = ""
        param_model = model
        if best_ckpt:
            with open_dict(cfg):
                cfg.train.phase2_constant_lr = None
            eval_model = ClassificationModule.load_from_checkpoint(best_ckpt, cfg=cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            eval_model.to(device)
            param_model = eval_model
            train_loader = dm.train_dataloader()
            test_loader = dm.test_dataloader()
            legacy_val_is_test = bool(getattr(dm, "_cub_legacy_val_is_test", False))
            if is_multi_label:
                formatters = {
                    "f1_micro": lambda x: f"{100.0 * float(x):.2f}%",
                    "f1_macro": lambda x: f"{100.0 * float(x):.2f}%",
                    "mAP": lambda x: f"{100.0 * float(x):.2f}%",
                }
                loss_name = "binary_cross_entropy"
            else:
                formatters = {
                    "top1_acc": lambda x: f"{100.0 * float(x):.2f}%",
                    f"top{min(5, num_classes)}_acc": lambda x: f"{100.0 * float(x):.2f}%",
                }
                loss_name = "cross_entropy"
            if legacy_val_is_test:
                m_tr = evaluate_classification_loader(
                    eval_model, train_loader, device, num_classes, multi_label=is_multi_label
                )
                m_te = evaluate_classification_loader(
                    eval_model, test_loader, device, num_classes, multi_label=is_multi_label
                )
                eval_blocks = build_eval_blocks_with_adapter(
                    task_name=str(cfg.task.name),
                    adapter_name="classification_eval_train_test_only",
                    adapter=classification_eval_train_test_only,
                    adapter_args=(m_tr, m_te, num_classes),
                    split_order=("train", "test"),
                    formatters=formatters,
                )
            else:
                val_loader = dm.val_dataloader()
                m_tr = evaluate_classification_loader(
                    eval_model, train_loader, device, num_classes, multi_label=is_multi_label
                )
                m_va = evaluate_classification_loader(
                    eval_model, val_loader, device, num_classes, multi_label=is_multi_label
                )
                m_te = evaluate_classification_loader(
                    eval_model, test_loader, device, num_classes, multi_label=is_multi_label
                )
                eval_blocks = build_eval_blocks_with_adapter(
                    task_name=str(cfg.task.name),
                    adapter_name="classification_eval_splits",
                    adapter=classification_eval_splits,
                    adapter_args=(m_tr, m_va, m_te, num_classes),
                    split_order=("train", "val", "test"),
                    formatters=formatters,
                )
        else:
            eval_blocks = "(evaluation skipped: no checkpoint file on disk)"
            loss_name = "binary_cross_entropy" if is_multi_label else "cross_entropy"

        run_details: Dict[str, Any] = {
            "input": input_shape_desc,
            "output": f"num_classes={num_classes} (logits dim)",
            "optimization_loss": loss_name,
            "multi_label": "yes" if is_multi_label else "no",
            "num_parameters": f"{count_module_parameters(param_model) / 1e6:.2f}M",
            "phase1_epochs_ran": f"{phase1_epochs} / planned_max {M}",
            "phase1_early_stopped": "yes" if phase1_epochs < M else "no",
            "phase2_ran": "yes" if phase2_ran else "no",
            "total_validation_epochs_recorded": len(val_hist.val_losses),
            "undertrained_warning_emitted": "yes" if undertrained_warn else "no",
        }
        if train_f.tensorboard_auto_launch_after_fit:
            run_details["tensorboard_url"] = tb_url or "(launch skipped, suppressed, or failed)"

        summary_path = str(artifact_path(os.getcwd(), "reports", "summary.log"))
        write_summary_log(
            summary_path,
            cfg,
            train_start=train_start,
            train_end=train_end,
            run_details=run_details,
            best_ckpt_path=best_ckpt or "(none)",
            eval_note=dm.eval_split_note(),
            eval_blocks=eval_blocks,
        )
        rank_zero_info(f"Wrote summary.log to {summary_path}")
    elif not train_f.summary_log or not task_f.post_train_eval_and_summary:
        rank_zero_info("summary.log skipped (train.facets.summary_log or task.facets.post_train_eval_and_summary is false).")
