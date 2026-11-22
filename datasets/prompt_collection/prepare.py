#!/usr/bin/env python3
"""Prepare validated prompts through the selected Dataset and Model plugins.

Works as `python -m saebase.datasets.prepare` and as `python -m saebase.datasets.prepare`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saebase.datasets.artifacts import (  # noqa: E402
    assert_available,  # noqa: E402
    canonicalize_behavior,  # noqa: E402
)

from saebase.runtime import verify_runtime  # noqa: E402


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Prepare SAEBase Dataset prompts")
    p.add_argument("--model-key", "--model", default="pythia70m")
    p.add_argument("--behavior", "--prompt-source", default="SimpleCopying")
    p.add_argument("--data-dir", "--data-root", dest="data_root", default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Dataset cohort size; defaults to its own config (500 for the selected matrix)",
    )
    p.add_argument(
        "--n-generate",
        type=int,
        default=None,
        help="Candidate budget for legacy generators only; fixed cohorts use --sample-count",
    )
    p.add_argument(
        "--min-kept",
        type=int,
        default=None,
        help="Legacy generator filter minimum; fixed cohorts use --sample-count",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def prepare(args):
    behavior = canonicalize_behavior(args.behavior)
    n_generate = args.n_generate if args.n_generate is not None else 300
    min_kept = args.min_kept if args.min_kept is not None else 20
    if n_generate <= 0 or min_kept <= 0 or min_kept > n_generate:
        raise ValueError("Require 0 < min-kept <= n-generate")
    assert_available(behavior, args.model_key)
    verify_runtime(context="sae_prepare_prompts", mode="analysis")

    from omegaconf import OmegaConf
    from saebase.datasets.artifacts import default_data_root, prepared_dataset_root
    from saebase.datasets.prompts import prepare_selected_dataset

    from models.hooked_lm_common import print_vram_gb

    data_root = (
        Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root()
    )
    cfg = OmegaConf.create(
        {
            "model": {"name": args.model_key, "device": args.device},
            "dataset": {
                "name": behavior,
                "data_path": str(prepared_dataset_root(data_root, behavior)),
                "prompt_file": "",
            },
            "paths": {"data_dir": str(data_root)},
            "seed": args.seed,
        }
    )
    # Read the Dataset-owned configuration instead of deriving its cohort from
    # this command's historical candidate-generation flags.
    from conf import group_names, load_group

    for alias in group_names("dataset"):
        selected = load_group("dataset", alias).dataset
        if selected.name == behavior and "sample_count" in selected:
            cfg.dataset = OmegaConf.merge(selected, {"data_path": str(prepared_dataset_root(data_root, behavior))})
            if args.sample_count is not None:
                cfg.dataset.sample_count = args.sample_count
            cfg.dataset.sampling_seed = args.seed
            break
    if "sample_count" in cfg.dataset and (args.n_generate is not None or args.min_kept is not None):
        raise ValueError(
            "This Dataset uses a fixed cohort; set --sample-count instead of candidate-budget flags"
        )
    if "sample_count" not in cfg.dataset and args.sample_count is not None:
        raise ValueError("This legacy generator has no fixed-cohort interface")
    out = prepare_selected_dataset(cfg, n_generate=n_generate, min_kept=min_kept)
    print(f"[prepare] model={args.model_key}")
    print(f"[prepare] prompts and manifest: {out['path']}")
    print_vram_gb()
    return out


def main() -> None:
    args = parse_args()
    try:
        prepare(args)
    except (ValueError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"[prepare] {exc}") from exc


if __name__ == "__main__":
    main()
