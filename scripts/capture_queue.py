"""Wait for the selected GPU, then launch a configured capture matrix.

This experiment launcher delegates data, model loading, SAR and result handling
to tasks.capture.matrix. It never interrupts foreign GPU or tmux processes.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from saebase.runtime.experiment_results import write_json as save

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    """Linux process birth identity prevents waiting on a reused PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def gpu_snapshot(gpu):
    def query(args):
        return subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout

    lines = list(
        csv.reader(
            query(
                ["--id=" + str(gpu), "--query-gpu=uuid,memory.total,memory.used,utilization.gpu"]
            ).splitlines()
        )
    )
    if len(lines) != 1 or len(lines[0]) != 4:
        raise ValueError("Expected one GPU with UUID, memory and utilization")
    uuid, total, used, utilization = [v.strip() for v in lines[0]]
    processes = list(csv.reader(query(["--query-compute-apps=gpu_uuid,pid"]).splitlines()))
    pids = [int(row[1].strip()) for row in processes if len(row) == 2 and row[0].strip() == uuid]
    return {
        "uuid": uuid,
        "total_mib": int(total),
        "used_mib": int(used),
        "utilization": int(utilization),
        "compute_pids": pids,
    }


def is_idle(snapshot, max_used_mib, max_utilization):
    return (
        not snapshot["compute_pids"]
        and snapshot["used_mib"] <= max_used_mib
        and snapshot["utilization"] <= max_utilization
    )


def wait_for_gpu(
    args, status_path, watched, *, sample=gpu_snapshot, sleep=time.sleep, now=time.monotonic
):
    quiet_since = None
    while True:
        active = [
            pid
            for pid, birth in watched.items()
            if birth is not None and process_identity(pid) == birth
        ]
        try:
            gpu = sample(args.gpu)
            quiet = not active and is_idle(gpu, args.max_used_mib, args.max_utilization)
            if quiet:
                quiet_since = now() if quiet_since is None else quiet_since
            else:
                quiet_since = None
            elapsed = now() - quiet_since if quiet_since is not None else 0
            state = {
                "status": "WAITING_GPU",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "waiting_pids": active,
                "gpu": gpu,
                "quiet_seconds": elapsed,
                "required_quiet_seconds": args.idle_seconds,
            }
            save(status_path, state)
            if quiet and elapsed >= args.idle_seconds:
                return gpu
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            quiet_since = None
            save(
                status_path,
                {"status": "WAITING_GPU_PROBE", "error": str(exc), "waiting_pids": active},
            )
        sleep(args.poll_seconds)


def freeze_config(args, report):
    from omegaconf import OmegaConf
    import hashlib

    cfg = OmegaConf.masked_copy(OmegaConf.load(args.config), ["matrix"])
    matrix = cfg.matrix
    if not matrix.models or not matrix.datasets:
        raise ValueError("Capture matrix requires model and dataset selections")
    mode = getattr(args, "capture_mode", "sar")
    source = args.sar_source.resolve() if args.sar_source else None
    if source is not None and source.is_dir():
        source = source / "sar.py"
    if mode == "sar" and (source is None or not source.is_file()):
        raise ValueError("Supply the original trusted sar.py for SAR")
    revisions = {}
    for item in args.model_revision:
        model, separator, revision = item.partition("=")
        if (
            not separator
            or model not in {str(row.model) for row in matrix.models}
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError("--model-revision requires a selected MODEL=40-character-commit")
        revisions[model] = revision
    for model in matrix.models:
        if model.model in revisions:
            model.model_overrides = {
                **dict(model.get("model_overrides", {})),
                "revision": revisions[model.model],
            }
        if mode in {"sar", "g_s", "I"}:
            model.offload_unused_layers = False  # SAR evaluates the complete downstream model.
    matrix.model_timeout_seconds = args.model_timeout_seconds
    feature_limit = getattr(args, "filter", None)
    if feature_limit is None:
        feature_limit = args.top_feature_num if args.top_feature_num is not None else 10
    observed = getattr(args, "capture_count", None)
    if observed is None and hasattr(args, "filter"):
        observed = args.top_feature_num
    from tasks.capture.selection import settings

    settings({"task": {"mode": mode, "filter": feature_limit, "capture_count": observed}})
    if mode == "sar":
        matrix.task_overrides = {
            "mode": "sar",
            "top_feature_num": args.top_feature_num if not hasattr(args, "filter") else None,
            "filter": feature_limit,
            "capture_count": observed,
            "neuronpedia": {"enabled": False},
            "sar": {
                "source": str(source) if source else None,
                "M": args.coalitions,
                "n_fit": args.n_fit,
                "readout": args.readout,
                "require_affine_decode": True,
            },
        }
    else:
        matrix.task_overrides = {
            **dict(matrix.get("task_overrides", {})),
            "mode": mode,
            "filter": feature_limit,
            "capture_count": observed,
        }
    path = report / "effective_matrix.yaml"
    content = OmegaConf.to_yaml(cfg, resolve=True)
    if path.exists() and path.read_text() != content:
        raise ValueError("Existing batch configuration differs; select a new report/tag")
    path.write_text(content)
    launch = {
        "argv": sys.argv,
        "python": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(path),
        "sar_source": str(source) if source else None,
        "sar_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        "models": OmegaConf.to_container(matrix.models),
        "datasets": list(matrix.datasets),
        "cells": len(matrix.models) * len(matrix.datasets),
        "seed": int(matrix.get("seed", 42)),
        "readout": args.readout,
        "scientific_acceptance": False,
    }
    save(report / "launch.json", launch)
    with (report / "launch_history.jsonl").open("a") as stream:
        stream.write(json.dumps(launch, ensure_ascii=False) + "\n")
    return path, matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf/tasks/capture/task.yaml")
    parser.add_argument("--sar-source", type=Path)
    parser.add_argument(
        "--capture-mode", choices=["sar", "whiten", "non_whiten", "g_s", "I"], default="sar"
    )
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument(
        "--model-revision",
        action="append",
        default=[],
        help="Explicit preparation revision, MODEL=COMMIT",
    )
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--idle-seconds", type=float, default=300)
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=0,
        help="0 waits for the complete model batch; positive values bound a model child",
    )
    parser.add_argument(
        "--top-feature-num", type=int, default=None, help="Legacy capture-count alias"
    )
    parser.add_argument("--filter", type=lambda value: "all" if value == "all" else int(value))
    parser.add_argument(
        "--capture-count", type=lambda value: "all" if value == "all" else int(value)
    )
    parser.add_argument("--coalitions", type=int, default=384)
    parser.add_argument("--n-fit", type=int, default=256)
    parser.add_argument(
        "--readout",
        choices=["logit_difference", "answer_mean_logprob"],
        default="answer_mean_logprob",
    )
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if args.idle_seconds < 300:
        parser.error("GPU must remain continuously idle for at least 300 seconds (5 minutes)")
    if (
        args.poll_seconds <= 0
        or args.idle_seconds < 0
        or args.max_used_mib < 0
        or not 0 <= args.max_utilization <= 100
    ):
        parser.error("Invalid GPU waiting thresholds")
    if args.capture_mode == "sar":
        k = args.filter if type(args.filter) is int else args.top_feature_num or 10
        if k < 2 or args.n_fit < 1 + k + k * (k - 1) // 2 or args.coalitions - args.n_fit < 2:
            parser.error("SAR coalition budget must cover the pairwise design and held-out set")
    from tasks.capture.matrix import CaptureMatrix

    report = CaptureMatrix(
        args.config, root=ROOT, tag=args.tag, mode=args.capture_mode, report=args.report_dir
    ).REPORT
    report.mkdir(parents=True, exist_ok=True)
    with (report / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, _ = freeze_config(args, report)
        # Compose and validate every cell without loading model weights or the estimator.
        from hydra import initialize_config_module
        from tasks.capture.operation import validate_cfg

        matrix = CaptureMatrix(
            config, root=ROOT, tag=args.tag, mode=args.capture_mode, report=report
        )
        with initialize_config_module(version_base=None, config_module="conf"):
            for model, sae, layer in matrix.MODELS:
                for _, alias in matrix.DATASETS:
                    validate_cfg(matrix.capture_config(alias, model, sae, layer))
        print("Validated 120 SAR capture configurations", flush=True)
        if args.check_config:
            return 0
        watched = {pid: process_identity(pid) for pid in args.wait_pid}
        outcomes = []
        for model, _, _ in matrix.MODELS:
            from saebase.runtime.environment import run_model

            gpu = wait_for_gpu(args, report / "status.json", watched)
            command = [
                "-u",
                "-m",
                "tasks.capture.matrix",
                "--config",
                str(config),
                "--root",
                str(ROOT),
                "--tag",
                args.tag,
                "--capture-mode",
                args.capture_mode,
                "--report-dir",
                str(report),
                "--model",
                model,
            ]
            log = report / f"run-{model}.log"
            save(
                report / "status.json",
                {
                    "status": "RUNNING_MODEL",
                    "model": model,
                    "gpu": gpu,
                    "command": command,
                    "log": str(log),
                },
            )
            print(f"Starting capture model {model}; log={log}", flush=True)
            started = time.time()
            outcome = run_model(
                model,
                command,
                log,
                cwd=ROOT,
                environment={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                },
                timeout=matrix.TIMEOUT or None,
            )
            outcome.update(
                model=model, seconds=time.time() - started, log=str(log), command=command
            )
            outcomes.append(outcome)
            save(report / f"process-{model}.json", outcome)
            matrix.summarize()
        code = matrix.summarize()
        save(report / "status.json", {"status": "FINISHED", "exit_code": code, "models": outcomes})
        return code


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    raise SystemExit(main())
