"""Capture matrix orchestration over existing Dataset, Model and Runtime APIs.

The configuration declares this experiment's selections, never component availability.
Each cell uses the public execution lifecycle; one child owns one loaded model/SAE.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import merge_task_overrides, task_value
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from saebase.runtime.experiment_results import write_json as save
from conf import load_group


class CaptureMatrix:
    def __init__(
        self, config=None, *, root=None, tag="feature_ranking", mode="non_whiten", report=None
    ):
        from importlib.resources import files
        from omegaconf import OmegaConf

        self.CONFIG = (
            Path(config)
            if config
            else Path(str(files("conf") / "tasks/capture/task.yaml"))
        )
        selection = OmegaConf.load(self.CONFIG).matrix
        self.ROOT = Path(root or Path.cwd()).resolve()
        self.CAPTURE_MODE = mode
        self.MODELS = [(row.model, row.sae_key, row.layer) for row in selection.models]
        self.DATASETS = [
            (
                str(load_group("dataset", str(alias)).dataset.name),
                str(alias),
            )
            for alias in selection.datasets
        ]
        if not self.MODELS or not self.DATASETS:
            raise ValueError("Capture matrix requires nonempty model and dataset selections")
        if len({m[0] for m in self.MODELS}) != len(self.MODELS) or len(set(self.DATASETS)) != len(
            self.DATASETS
        ):
            raise ValueError("Capture matrix selections must be unique")
        from saebase.runtime.experiment_paths import capture_method_role, experiment_name, matrix_report

        self.TAG = experiment_name(tag)
        self.REPORT = (
            Path(report).resolve()
            if report
            else matrix_report(self.ROOT, "capture", self.TAG, capture_method_role(self.TAG, self.CAPTURE_MODE))
        )
        self.SEED = int(selection.get("seed", 42))
        self.MEMORY = {
            row.model: mode not in {"sar", "g_s", "I"}
            and bool(row.get("offload_unused_layers", False))
            for row in selection.models
        }
        self.TIMEOUT = int(selection.get("model_timeout_seconds", 86400))
        if selection.get("interpreters"):
            raise ValueError("Model interpreters are configured in conf/base/runtime/runtime.yaml")
        self.OVERRIDES = dict(selection.get("task_overrides", {}))
        self.DATASET_OVERRIDES = dict(selection.get("dataset_overrides", {}))
        self.MODEL_OVERRIDES = {
            row.model: dict(row.get("model_overrides", {})) for row in selection.models
        }

    def member(self, dataset, model):
        from saebase.runtime.experiment_paths import capture_method_role, member_directory

        return member_directory(
            self.ROOT, dataset, model, "capture", self.TAG,
            capture_method_role(self.TAG, self.CAPTURE_MODE),
        )

    def cell_state(self, dataset, model, cfg):
        from saebase.runtime.experiment_results import inspect_run

        return inspect_run(self.member(dataset, model), expected_cfg=cfg)

    def prepare_inputs(self, config, pair):
        """Each Dataset owns its preparation; a failed source cannot stop its peers."""
        from saebase.datasets.selection import select_dataset
        from tasks.precheck.report import error_result

        errors = {}
        for name, alias in self.DATASETS:
            try:
                cfg = config(alias, "capture")
                device = str(next(pair.sae.parameters()).device)
                try:
                    if name == "IOI":
                        pair.sae.to("cpu")
                    select_dataset(
                        cfg.dataset,
                        model_key=pair.identity.model_key,
                        model=pair.model.model,
                        loaded_model_name=pair.identity.loaded_model_name,
                    )
                finally:
                    if name == "IOI":
                        pair.sae.to(device)
            except Exception as exc:
                errors[name] = {
                    "status": error_result(
                        exc, stage="dataset_prepare", attempted=["Dataset-owned preparation"]
                    )["status"],
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "phase": "dataset_prepare",
                }
        return errors

    def run_cell(self, name, alias, key, sae, layer, config, preparation_error=None):
        from saebase.runtime.experiment_paths import prepare_run, run_session
        from saebase.runtime.experiment_results import record_run
        from saebase.runtime.train import execute_run

        record = self.REPORT / "cells" / f"{name}--{key}.json"
        base_leaf = self.member(name, key)
        cell = {
            "dataset": name,
            "model": key,
            "sae": sae,
            "layer": layer,
            "requested_samples": None,
            "tasks": {},
            "capture_scope": "all Dataset-selected inputs, final token",
        }
        cfg = None
        leaf = base_leaf
        origin = Path.cwd()
        started = time.time()
        try:
            cfg = config(alias, "capture")
            from saebase.runtime.experiment_paths import capture_method_role

            cfg.experiment.run_id = capture_method_role(self.TAG, self.CAPTURE_MODE)
            explicit_samples = task_value(cfg.task, "sample_ids")
            cell["requested_samples"] = (
                len(explicit_samples)
                if explicit_samples is not None
                else cfg.dataset.get("sample_count")
            )
            if explicit_samples is not None:
                cell["dataset_cohort_samples"] = cfg.dataset.get("sample_count")
                cell["capture_scope"] = (
                    "explicit observation subset of the Dataset cohort, final token"
                )
            state = self.cell_state(name, key, cfg)
            if state["action"] in {"skip", "annotations_pending"}:
                return state
            leaf.mkdir(parents=True, exist_ok=True)
            os.chdir(leaf)
            if preparation_error is not None:
                cell["preparation"] = preparation_error
                cell["tasks"]["precheck"] = {
                    "status": "blocked",
                    "exit_code": 2,
                    "reason": "Dataset preparation did not finish; no capture was executed",
                }
                from saebase.results import EvaluationResult, normalize_status

                outcome = EvaluationResult(
                    ok=False,
                    task="capture",
                    evaluation="capture",
                    model_key=key,
                    prompt_source=name,
                    status=normalize_status(preparation_error.get("status", "FAIL")),
                    error=preparation_error["message"],
                    summary={"preparation_error": preparation_error},
                )
                with run_session(cfg, leaf):
                    prepare_run(cfg, leaf)
                    record_run(leaf, cfg, result=outcome)
            else:
                outcome = execute_run(cfg, leaf)
            cell["tasks"]["capture"] = {
                "status": outcome.status,
                "exit_code": outcome.exit_code,
                "completion": outcome.completion,
                "stages": outcome.summary.get("stages", {}),
                "seconds": time.time() - started,
            }
            receipt = outcome.artifact_paths.get("precheck")
            if receipt and Path(receipt).is_file():
                status = json.loads(Path(receipt).read_text())["summary"]["status"]
                cell["tasks"]["precheck"] = {
                    "status": status,
                    "exit_code": 0 if status == "PASS" else 1 if status == "FAIL" else 2,
                }
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if (
                cfg is not None
                and leaf.is_dir()
                and (not (leaf / "metadata/manifest.json").exists())
            ):
                try:
                    record_run(leaf, cfg, error=error)
                except Exception as recording_error:
                    error["recording_error"] = str(recording_error)
            cell["tasks"]["capture"] = {
                "status": "failed",
                "exit_code": 1,
                "error": error,
                "seconds": time.time() - started,
            }
        finally:
            os.chdir(origin)
        save(record, cell)
        print(
            name,
            key,
            {
                task: {field: value.get(field) for field in ("status", "exit_code", "seconds")}
                for task, value in cell["tasks"].items()
            },
            flush=True,
        )
        return {"action": "attempted", "cell": cell}

    def capture_config(self, alias, key, sae, layer):
        from hydra import compose
        from models.loading_policy import resolve_config_devices

        cfg = compose(
            config_name="config",
            overrides=[
                f"model={key}",
                f"paths.root={self.ROOT}",
                f"dataset={alias}",
                "task=capture",
                "task.execute=true",
                f"task.capture.mode={self.CAPTURE_MODE}",
                f"experiment.sub_name={self.TAG}",
                f"seed={self.SEED}",
                f"task.sae_key={sae}",
                f"task.layer={layer}",
            ],
        )
        from omegaconf import OmegaConf, open_dict
        from tasks.capture.configuration import migrate_capture_task

        overrides = migrate_capture_task(self.OVERRIDES)
        if self.OVERRIDES.get("top_feature_num") is not None:
            overrides.setdefault("capture", {}).setdefault("selection", {}).setdefault(
                "candidate_count", self.OVERRIDES["top_feature_num"]
            )
        cfg.task = merge_task_overrides(cfg.task, overrides)
        from saebase.runtime.experiment_paths import capture_method_role

        if task_value(cfg.task, "mode") != self.CAPTURE_MODE:
            raise ValueError("Capture matrix mode is selected by the matrix entry point")
        cfg.experiment.run_id = capture_method_role(self.TAG, self.CAPTURE_MODE)
        cfg.dataset = OmegaConf.merge(cfg.dataset, self.DATASET_OVERRIDES)
        with open_dict(cfg.model):
            cfg.model.merge_with(self.MODEL_OVERRIDES.get(key, {}))
        resolve_config_devices(cfg)
        from models.numerics import bind_configuration

        bind_configuration(cfg)
        return cfg

    def child(self, key, *, offload_unused_layers=False):
        if self.CAPTURE_MODE in {"sar", "g_s", "I"} and offload_unused_layers:
            raise ValueError(
                "SAR requires the complete downstream model; prefix-only tail offload is unavailable"
            )
        import torch
        from hydra import initialize_config_module
        from saebase.runtime.sae_operations import resolve_sae, load_selection, loaded_pair_session

        _, sae, layer = next((m for m in self.MODELS if m[0] == key))
        self.REPORT.mkdir(parents=True, exist_ok=True)
        with initialize_config_module(version_base=None, config_module="conf"):

            def config(alias, task):
                cfg = self.capture_config(alias, key, sae, layer)
                if offload_unused_layers:
                    from omegaconf import open_dict

                    with open_dict(cfg.experiment):
                        cfg.experiment.capture_memory_policy = (
                            "unused_tail_on_cpu; forward_stops_after_capture_layer"
                        )
                return cfg

            start = time.time()
            cfg = config(self.DATASETS[0][1], "capture")
            try:
                from models.numerics import configure_task
                from saebase.runtime.environment import configure_worker

                configure_worker(cfg)
                configure_task(cfg)
                selection = resolve_sae(cfg.model, sae, layer, operation="capture")
                pair = load_selection(selection, "cuda")
                save(
                    self.REPORT / f"pair-{key}.json",
                    {
                        "status": "READY",
                        "identity": pair.identity.to_dict(),
                        "seconds": time.time() - start,
                    },
                )
            except Exception as e:
                save(
                    self.REPORT / f"pair-{key}.json",
                    {
                        "status": "BLOCKED",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "seconds": time.time() - start,
                    },
                )
                from tasks.precheck.report import error_result

                failure = {
                    "status": error_result(
                        e, stage="model_load", attempted=["load selected model/SAE"]
                    )["status"],
                    "type": type(e).__name__,
                    "message": str(e),
                    "phase": "model_load",
                }
                for name, alias in self.DATASETS:
                    self.run_cell(name, alias, key, sae, layer, config, failure)
                raise
            preparation_errors = self.prepare_inputs(config, pair)
            from contextlib import nullcontext
            from tasks.capture.operation import offload_capture_tail

            memory = offload_capture_tail(pair) if offload_unused_layers else nullcontext()
            with memory, loaded_pair_session(pair):
                for name, alias in self.DATASETS:
                    self.run_cell(
                        name, alias, key, sae, layer, config, preparation_errors.get(name)
                    )
                    torch.cuda.empty_cache()

    def summarize(self):
        from collections import Counter

        counts = {"precheck": Counter(), "capture": Counter()}
        entries = []
        failed = False
        incomplete = False
        for model, sae, layer in self.MODELS:
            pair_file = self.REPORT / f"pair-{model}.json"
            pair = json.loads(pair_file.read_text()) if pair_file.exists() else {}
            for dataset, alias in self.DATASETS:
                path = self.REPORT / "cells" / f"{dataset}--{model}.json"
                continuation = None
                try:
                    cell = (
                        json.loads(path.read_text())
                        if path.is_file()
                        else {"dataset": dataset, "model": model, "tasks": {}}
                    )
                except (OSError, ValueError, TypeError) as exc:
                    cell = {
                        "dataset": dataset,
                        "model": model,
                        "tasks": {},
                        "record_error": str(exc),
                    }
                    continuation = {
                        "action": "retry",
                        "state": "incomplete",
                        "reason": "invalid_cell_record",
                    }
                from saebase.runtime.experiment_results import inspect_run

                member = self.member(dataset, model)
                capture = cell.get("tasks", {}).get("capture", {})
                if (member / "metadata/manifest.json").is_file() or capture.get("exit_code") == 0:
                    continuation = inspect_run(member)
                    continuation.pop("path", None)
                cell.get("tasks", {}).get("capture", {}).pop("run_dir", None)
                cell.get("tasks", {}).get("precheck", {}).pop("receipt", None)
                if continuation is not None:
                    cell["continuation"] = continuation
                for task in counts:
                    result = cell["tasks"].get(task, {})
                    if task == "capture" and continuation is not None:
                        status = continuation["state"]
                    elif result.get("exit_code") == 0:
                        status = "complete"
                    elif result.get("exit_code") == 2 or result.get("status") == "blocked":
                        status = (
                            "incomplete"
                            if result.get("completion", {}).get("status") == "incomplete"
                            else "blocked"
                        )
                    elif result:
                        status = "failed"
                    else:
                        status = "not_run"
                    counts[task][status] += 1
                    failed |= status == "failed"
                    incomplete |= status in {
                        "blocked",
                        "incomplete",
                        "annotations_pending",
                        "not_run",
                    }
                if not cell.get("tasks"):
                    cell["dependency_reason"] = (
                        cell.get("preparation") or pair.get("error") or "No execution receipt"
                    )
                entries.append(cell)
        summary = {
            "schema": "saebase.dataset_model_matrix.v1",
            "cells": len(entries),
            "counts": {k: dict(v) for k, v in counts.items()},
            "records": entries,
            "exit_code": 1 if failed else 2 if incomplete else 0,
            "scope": "Dataset-owned selected cohort across original splits; every selected input captured at final token; no native benchmark accuracy claims",
        }
        save(self.REPORT / "summary.json", summary)
        print(
            json.dumps(
                {
                    "cells": len(entries),
                    "counts": summary["counts"],
                    "exit_code": summary["exit_code"],
                }
            ),
            flush=True,
        )
        return summary["exit_code"]

    def settled_cell(self, name, model):
        from hydra import initialize_config_module

        _, sae, layer = next((item for item in self.MODELS if item[0] == model))
        alias = next((alias for dataset, alias in self.DATASETS if dataset == name))
        with initialize_config_module(version_base=None, config_module="conf"):
            cfg = self.capture_config(alias, model, sae, layer)
        return self.cell_state(name, model, cfg)["action"] in {"skip", "annotations_pending"}

    def record_scope(self):
        self.REPORT.mkdir(parents=True, exist_ok=True)
        save(
            self.REPORT / "scope.json",
            {
                "models": self.MODELS,
                "datasets": self.DATASETS,
                "capture_mode": self.CAPTURE_MODE,
                "tasks": ["precheck", "capture"],
            },
        )
    def run(self):
        self.record_scope()
        for model, _, _ in self.MODELS:
            from saebase.runtime.environment import run_model

            log = self.REPORT / f"run-{model}.log"
            command = [
                "-m",
                "tasks.capture.matrix",
                "--config",
                str(self.CONFIG.resolve()),
                "--root",
                str(self.ROOT),
                "--tag",
                self.TAG,
                "--capture-mode",
                self.CAPTURE_MODE,
                "--report-dir",
                str(self.REPORT),
                "--model",
                model,
            ]
            if self.MEMORY.get(model):
                command.append("--offload-unused-layers")
            outcome = run_model(
                model,
                command,
                log,
                cwd=self.ROOT,
                timeout=self.TIMEOUT or None,
            )
            save(self.REPORT / f"process-{model}.json", outcome)
        return self.summarize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="feature_ranking")
    parser.add_argument(
        "--capture-mode", choices=["non_whiten", "whiten", "g_s", "sar", "I"], default="non_whiten"
    )
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--offload-unused-layers", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    matrix = CaptureMatrix(
        args.config, root=args.root, tag=args.tag, mode=args.capture_mode, report=args.report_dir
    )
    if args.dataset:
        matrix.DATASETS = [
            (name, alias) for name, alias in matrix.DATASETS if args.dataset in (name, alias)
        ]
        if not matrix.DATASETS:
            parser.error("Unknown selected dataset")
    if args.summarize_only:
        matrix.record_scope()
        return matrix.summarize()
    if args.model:
        import fcntl
        from saebase.runtime.environment import redirect_model

        if args.model not in {row[0] for row in matrix.MODELS}:
            parser.error("Unknown selected model")
        matrix.REPORT.mkdir(parents=True, exist_ok=True)
        redirected = redirect_model(
            args.model,
            ["-m", "tasks.capture.matrix", *(sys.argv[1:] if argv is None else argv)],
            matrix.REPORT / f"run-{args.model}.log",
            cwd=matrix.ROOT,
        )
        if redirected is not None:
            return redirected
        with (matrix.REPORT / f"{args.model}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if all(matrix.settled_cell(name, args.model) for name, _ in matrix.DATASETS):
                return 0
            matrix.child(args.model, offload_unused_layers=args.offload_unused_layers)
        return 0
    return matrix.run()


if __name__ == "__main__":
    raise SystemExit(main())
