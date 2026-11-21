"""Benchmark argument conversion; real execution uses the public Task lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from conf import read_configuration


def _run_protocol(args, raw):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from saebase.runtime.experiment_paths import run_leaf
    from saebase.runtime.train import execute_run

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="config", overrides=["task=benchmark", "task.execute=true"])
    models = [
        {"name": target["model_key"], "device": args.device}
        for target in {t["model_key"]: t for t in raw["targets"]}.values()
    ]
    cfg.model = OmegaConf.create(models[0] if len(models) == 1 else {"members": models})
    dataset = raw.get("dataset", {})
    cfg.dataset = OmegaConf.create(
        {
            "name": dataset.get("id", raw.get("dataset_id", "comparison-prompts")),
            **({"samples": dataset["samples"]} if dataset.get("samples") is not None else {}),
            **({"prompts": raw["prompts"]} if "prompts" in raw else {}),
        }
    )
    cfg.paths.root = str(args.output.resolve())
    cfg.experiment.sub_name = "protocol_evaluation"
    cfg.seed = raw.get("seed", 42)
    cfg.task.device = args.device
    cfg.task.extensions = args.extension
    cfg.task.benchmark.benchmark_config = str(args.config.resolve())
    cfg.task.resume = args.resume
    folder = run_leaf(cfg)
    if args.resume and (folder / "metadata/manifest.json").is_file():
        cfg.runtime.mode = "resume"
    folder.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(folder)
        result = execute_run(cfg, folder)
    finally:
        os.chdir(previous)
    summary = result.summary
    print(
        json.dumps(
            {
                "run_dir": str(folder),
                "run_id": summary.get("benchmark_id"),
                "metrics": summary.get("metric_count"),
                "completion": result.completion,
                "report": str(
                    folder / "metrics" / str(summary.get("benchmark_id", "")) / "report.html"
                ),
            },
            indent=2,
        )
    )
    return result.exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path)
    selection.add_argument("--pair-protocol", choices=("cpu_pair", "reference_pair"))
    parser.add_argument("--sae-key")
    parser.add_argument("--model")
    parser.add_argument("--extension", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Workspace root for execution; metric root for --report-only",
    )
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)
    if args.extension:
        from saebase.extensions import discover_extensions

        discover_extensions(args.extension)
    if args.pair_protocol:
        from tasks.benchmark.operation import run_pair_benchmark

        if not args.real:
            parser.error("bounded pair protocols require --real")
        if args.pair_protocol == "cpu_pair" and not args.sae_key:
            parser.error("cpu_pair requires --sae-key")
        failure_path = args.output / "pair_failure.json"
        try:
            result = run_pair_benchmark(
                args.output,
                protocol=args.pair_protocol,
                sae_key=args.sae_key,
                model_name=args.model,
                device=args.device,
            )
        except Exception as exc:
            from tasks.precheck.report import error_result

            failure = error_result(
                exc,
                stage=f"bounded_benchmark:{args.pair_protocol}",
                attempted=[
                    "load selected pair",
                    "measure configured metrics",
                    "audit saved measurements",
                ],
            )
            failure["protocol"] = args.pair_protocol
            failure["output"] = str(args.output.resolve())
            failure["impact"] = (
                "Requested benchmark and reliability acceptance is incomplete; preserve any measured artifacts for diagnosis."
            )
            failure["available_artifacts"] = (
                sorted(str(p.relative_to(args.output)) for p in args.output.iterdir())
                if args.output.is_dir()
                else []
            )
            token = os.getenv("HF_TOKEN")
            if token:
                failure = json.loads(json.dumps(failure).replace(token, "[redacted]"))
            args.output.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            print(json.dumps(failure, indent=2, allow_nan=False))
            return 1
        failure_path.unlink(missing_ok=True)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    raw = read_configuration(args.config, resolve=True)
    if args.real:
        from models.loading_policy import resolve_device

        args.device = resolve_device(args.device)
        return _run_protocol(args, raw)
    if raw.get("schema") == "saebase.model_comparison":
        from tasks.benchmark.comparison import load_matrix, plan_matrix
        from tasks.benchmark.comparison.runner import verify_matrix

        matrix = load_matrix(args.config)
        if args.dry_run:
            print(json.dumps(plan_matrix(matrix), indent=2))
            return 0
        result = verify_matrix(args.output / matrix.matrix_id, expected_id=matrix.matrix_id)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 1
    from tasks.benchmark.execution.planner import build_plan
    from tasks.benchmark.execution.schema import load_config

    config = load_config(args.config)
    plan = build_plan(config)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from saebase.results import EvaluationResult
    from tasks.benchmark.operation import benchmark_completion
    from tasks.benchmark.results.metrics import read_metrics
    from tasks.benchmark.results.report import write_report
    from tasks.benchmark.results.validation import validate_external_sources

    root = args.output / config.run_id
    rows = read_metrics(root, config.run_id)
    validate_external_sources(config, rows, root=root)
    write_report(root, config, plan, rows)
    completed = benchmark_completion(config, rows)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "metrics": len(rows),
                "completion": completed,
                "report": str(root / "report.html"),
            }
        )
    )
    return EvaluationResult(
        True, "benchmark", "benchmark", "multiple", config.dataset_id, completion=completed
    ).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
