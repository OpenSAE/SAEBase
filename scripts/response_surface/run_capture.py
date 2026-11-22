"""Run the recorded QA selection through the normal Capture Runtime entrypoint."""

import argparse
from pathlib import Path
import json
import os

from omegaconf import OmegaConf

from saebase.datasets.artifacts import PREPARED_ASSET, prepared_dataset_root
from saebase.runtime.experiment_results import historical_run_path


def limit(value):
    return "all" if value == "all" else int(value)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--dataset",
        default="GSM8K",
        choices=["GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"],
    )
    parser.add_argument("--mode", default="I", choices=["whiten", "non_whiten", "g_s", "sar", "I"])
    parser.add_argument("--filter", type=limit, default=10)
    parser.add_argument("--capture-count", type=limit)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--tag", default="filtered_behavior")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "assets/history_experiment/response_surface/files/plan.json").read_text())
    cfg = OmegaConf.load(
        historical_run_path(
            root / "experiment" / args.dataset / plan["model"]
            / "capture/reference_additivity/metadata/config.yaml"
        )
    )
    cfg.paths.root = str(root)
    cfg.paths.data_dir = str(root / PREPARED_ASSET)
    cfg.dataset.data_path = str(prepared_dataset_root(cfg.paths.data_dir, args.dataset))
    old_task = cfg.task
    cfg.task = OmegaConf.load(root / "conf/tasks/capture/task.yaml").task
    for key in ("sae_key", "layer", "device", "neuronpedia"):
        cfg.task[key] = old_task[key]
    cfg.task.execute = True
    cfg.task.capture.mode = args.mode
    if args.mode == "sar":
        cfg.task.capture.sar.source = str(root / "tasks/capture/scoring/reference_response_surface.py")
    cfg.task.capture.selection.candidate_policy = "legacy"
    cfg.task.capture.selection.candidate_count = args.filter
    cfg.task.capture.selection.feature_filter.min_native_nonzero_fraction = 0.5
    cfg.task.capture.selection.feature_filter.density_strict = False
    cfg.task.capture.output.feature_count = args.capture_count
    cfg.dataset.sample_count = args.questions
    cfg.experiment.sub_name = args.tag
    import torch

    torch.set_num_threads(4)
    from saebase.runtime.train import execute_run
    from saebase.runtime.configuration.config_validate import validate_hydra_cfg

    validate_hydra_cfg(cfg)
    if args.check_config:
        print(OmegaConf.to_yaml(cfg.task))
        return
    from saebase.runtime.experiment_paths import run_leaf

    directory = run_leaf(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    origin = Path.cwd()
    try:
        os.chdir(directory)
        outcome = execute_run(cfg, directory)
    finally:
        os.chdir(origin)
    print(f"{outcome.status}: {directory}")
    raise SystemExit(outcome.exit_code)


if __name__ == "__main__":
    main()
