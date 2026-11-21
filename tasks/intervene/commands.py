"""CLI argument conversion for the public intervention Task lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from conf import read_configuration
from tasks.intervene.batch import parse_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    path = args.config.resolve()
    raw = read_configuration(path)
    _, _, _, _, plan, _ = parse_plan(raw)
    if args.dry_run:
        print(json.dumps({"schema": "saebase.m2.dry_run.v1", **plan}, indent=2))
        return 0
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={raw['model_key']}",
                "task=intervene",
                f"paths.root={args.output.resolve()}",
                "experiment.sub_name=feature_intervention",
                "task.execute=true",
            ],
        )
    cfg.model.device = cfg.task.device = args.device
    cfg.dataset = OmegaConf.create({"name": "explicit-token-samples", "samples": raw["samples"]})
    cfg.task.sae_key = raw["sae_key"]
    cfg.task.layer = raw["layer"]
    cfg.task.intervene.protocol = {
        k: v for k, v in raw.items() if k not in {"model_key", "sae_key", "layer", "samples"}
    }
    from saebase.runtime.experiment_paths import run_leaf
    from saebase.runtime.train import execute_run

    folder = run_leaf(cfg)
    if not args.no_resume and (folder / "metadata/manifest.json").is_file():
        cfg.runtime.mode = "resume"
    folder.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(folder)
        result = execute_run(cfg, folder)
    finally:
        os.chdir(previous)
    print(json.dumps(result.to_dict(), indent=2))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
