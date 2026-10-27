"""SAEBase entry for the standalone SAE activation-health pipeline.

Typical usage::

  python main.py --task sae --model pythia70m --prompt-source SimpleCopying \
    --sae pythia70m_resid_sm --layer 3 --data-dir ./data --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from saebase.runtime import verify_runtime
from saebase.sae.registry import DEFAULT_SAE_FOR_MODEL, SAE_KEYS, get_profile
from saebase.datasets.artifact_layout import BEHAVIOR_ALIASES


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="SAEBase: evaluate SAE activation health")
    p.add_argument("--task", default="sae", help="Available built-in task: sae")
    p.add_argument("--model", default="pythia70m")
    p.add_argument("--prompt-source", default="SimpleCopying")
    p.add_argument("--sae", default=None, help="SAE profile (default: model's default SAE)")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--evaluation", default=None)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--prompt-format", default=None)
    p.add_argument("--sub-path", default="basic")
    p.add_argument(
        "--data-dir",
        "--data-root",
        dest="data_dir",
        default=None,
        help="Prepared prompt root, relative to invocation directory",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Hydra run output directory, relative to invocation directory",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument(
        "--list", action="store_true", help="List registered tasks, models and SAE profiles"
    )
    return p.parse_known_args(argv)


def _prompt_source_group(name: str) -> str:
    from saebase.datasets.artifact_layout import canonicalize_behavior

    canonical = canonicalize_behavior(name)
    # These are the canonical-to-Hydra groups shipped in conf/prompt_source/.
    group = next(
        (
            alias
            for alias, target in BEHAVIOR_ALIASES.items()
            if target == canonical and alias == alias.lower() and "-" not in alias
        ),
        None,
    )
    if (
        group is None
        or not (_project_root() / "conf" / "prompt_source" / (group + ".yaml")).is_file()
    ):
        raise ValueError(f"Prompt source {name!r} has no bundled Hydra configuration")
    return group


def _absolute_from_caller(value: str | None, caller: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else caller / path).resolve())


def build_overrides(args: argparse.Namespace, caller: Path) -> list[str]:
    if args.task != "sae":
        raise ValueError(
            f"Task {args.task!r} is not packaged in the standalone SAEBase task runner. "
            "Available: sae (activation_health)."
        )
    if args.evaluation not in (None, "activation_health"):
        raise ValueError("The standalone task runner supports evaluation=activation_health only")
    if args.n_prompts < 1:
        raise ValueError("--n-prompts must be >= 1")
    if args.model not in DEFAULT_SAE_FOR_MODEL and args.sae is None:
        raise ValueError(f"Model {args.model!r} needs an explicit --sae")
    sae_key = args.sae or DEFAULT_SAE_FOR_MODEL[args.model]
    if sae_key not in SAE_KEYS:
        raise ValueError(f"Unknown SAE profile: {sae_key!r}")
    profile = get_profile(sae_key)
    if args.model not in profile.allowed_model_keys():
        raise ValueError(f"SAE {sae_key!r} is incompatible with model {args.model!r}")
    if args.layer is not None and args.layer not in profile.layers:
        raise ValueError(f"SAE {sae_key!r} supports layers {profile.layers}, got {args.layer}")
    overrides = [
        f"paths.root={caller.resolve()}",
        f"prompt_source={_prompt_source_group(args.prompt_source)}",
        f"model={args.model}",
        f"sae={sae_key}",
        "evaluation=activation_health",
        "task=sae",
        "task.evaluation=activation_health",
        f"evaluation.n_prompts={args.n_prompts}",
        f"evaluation.device={args.device}",
        f"model.device={args.device}",
        f"sae.device={args.device}",
        f"experiment.sub_name={args.sub_path}",
        f"seed={args.seed}",
    ]
    if args.layer is not None:
        overrides.append(f"sae.layer={args.layer}")
    if args.prompt_format is not None:
        overrides.append(f"evaluation.prompt_format={args.prompt_format}")
    data_dir = _absolute_from_caller(args.data_dir, caller)
    if data_dir is not None:
        overrides.append(f"paths.data_dir={data_dir}")
    run_dir = _absolute_from_caller(args.run_dir, caller)
    if run_dir is not None:
        overrides.append(f"hydra.run.dir={run_dir}")
    return overrides


def main() -> None:
    args, unknown = parse_args()
    if args.list:
        from saebase.registry import (
            list_registered_model_keys,
            list_registered_sae_keys,
            list_registered_task_keys,
        )

        print("Tasks:", ", ".join(list_registered_task_keys()))
        print("Models:", ", ".join(list_registered_model_keys()))
        print("SAEs:", ", ".join(list_registered_sae_keys()))
        return
    caller = Path.cwd()
    overrides = build_overrides(args, caller)
    verify_runtime(context="main.py", mode="analysis")
    print("[main] project root:", _project_root(), flush=True)
    print("[main] Hydra overrides:", " ".join(overrides), flush=True)
    if unknown:
        print("[main] extra Hydra overrides:", " ".join(unknown), flush=True)
    sys.argv = ["saebase.run", *overrides, *unknown]
    runpy.run_module("saebase.run", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
