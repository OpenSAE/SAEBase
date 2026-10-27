#!/usr/bin/env python3
"""Prepare validated SimpleCopying prompts for the standalone Pythia SAE demo.

Works as `python saebase/tools/sae_prepare_prompts.py` and as `python -m saebase.tools.sae_prepare_prompts`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saebase.datasets.artifact_layout import canonicalize_behavior
from saebase.datasets.prompt_schema import assert_available
from saebase.runtime import verify_runtime


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Prepare SAEBase SimpleCopying prompts")
    p.add_argument("--model-key", "--model", default="pythia70m")
    p.add_argument("--behavior", "--prompt-source", default="SimpleCopying")
    p.add_argument("--data-dir", "--data-root", dest="data_root", default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--n-generate", type=int, default=300)
    p.add_argument("--min-kept", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def prepare(args):
    behavior = canonicalize_behavior(args.behavior)
    if behavior != "SimpleCopying":
        raise ValueError(
            f"Phase A preparation supports SimpleCopying, got {behavior!r}. "
            "Other behavior config files remain available for later expansion."
        )
    if args.n_generate <= 0 or args.min_kept <= 0 or args.min_kept > args.n_generate:
        raise ValueError("Require 0 < min-kept <= n-generate")
    assert_available(behavior, args.model_key)
    verify_runtime(context="sae_prepare_prompts", mode="analysis")

    from models.hooked_lm_common import print_vram_gb
    from saebase.registry import get_model_plugin
    from saebase.datasets.SimpleCopying.prepare import prepare_for_model

    get_model_plugin(args.model_key)  # refuse unregistered models before weight loading
    import importlib

    mod = importlib.import_module(f"models.{args.model_key}.{args.model_key}")
    handle = mod.load_model(device=args.device)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    out = prepare_for_model(
        args.model_key,
        handle.model,
        model_name=mod.get_spec().model_name,
        loaded_model_name=handle.loaded_name,
        n_generate=args.n_generate,
        seed=args.seed,
        min_kept=args.min_kept,
        data_root=data_root,
    )
    print(f"[prepare] model={args.model_key} loaded={handle.loaded_name}")
    print(f"[prepare] prompts and manifest: {out}")
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
