"""Run the recorded QA selection through the normal Capture Runtime entrypoint."""

import argparse
from pathlib import Path
import json
import os

from omegaconf import OmegaConf


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
    parser.add_argument("--tag", default="capture_filtered_behavior")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "scripts/SAR/results/plan.json").read_text())
    source = next(q for q in plan["questions"] if q["dataset"] == args.dataset)
    cfg = OmegaConf.load(Path(source["source_run"]) / "metadata/config.yaml")
    old_task = cfg.task
    cfg.task = OmegaConf.load(root / "conf/task/capture.yaml").task
    for key in ("sae_key", "layer", "device", "neuronpedia"):
        cfg.task[key] = old_task[key]
    cfg.task.execute = True
    cfg.task.mode = args.mode
    if args.mode == "sar":
        cfg.task.sar.source = str(root / "scripts/SAR/legacy/sar.py")
    cfg.task.filter = args.filter
    cfg.task.capture_count = args.capture_count
    cfg.dataset.sample_count = args.questions
    cfg.experiment.sub_name = args.tag
    import torch

    torch.set_num_threads(4)
    from saebase.runtime.train import execute_run
    from saebase.runtime.cfg.config_validate import validate_hydra_cfg

    validate_hydra_cfg(cfg)
    if args.check_config:
        print(OmegaConf.to_yaml(cfg.task))
        return
    directory = (
        root / "experiment" / args.dataset / plan["model"] / "capture" / f"{args.tag}_{cfg.seed}"
    )
    directory.mkdir(parents=True, exist_ok=False)
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
