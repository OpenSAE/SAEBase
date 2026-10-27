"""First SAEBase experiment: two toy SAEs, one model, benchmark and reliability reports.

Run after installing SAEBase in a checkout:
    python saebase/examples/quickstart/first_experiment.py --output runs/first-experiment

All data and checkpoints in this example are synthetic and CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.report import write_report
from saebase.benchmarks.results import read_metrics
from saebase.benchmarks.runner import execute
from saebase.reliability import AuditConfig, evaluate_m3, read_audit, write_audit

SITE = ActivationSite(0, "blocks.0.hook_resid_post")


def make_experiment() -> tuple[BenchmarkConfig, ModelAdapter, dict[str, ResearchToySAE]]:
    """Eight independent prompt groups and two distinct toy SAE weights."""
    model = ModelAdapter(ResearchToyModel().eval())
    saes = {
        "toy-a": ResearchToySAE().eval(),
        "toy-b": ResearchToySAE().eval(),
    }
    with torch.no_grad():
        saes["toy-b"].bias.add_(0.125)

    config = BenchmarkConfig.from_dict(
        {
            "schema": "saebase.benchmark.config.v1",
            "name": "First synthetic SAEBase experiment",
            "targets": [
                {
                    "model_key": "synthetic-toy",
                    "sae_keys": ["toy-a", "toy-b"],
                    "layer": SITE.layer,
                    "hook_name": SITE.hook_name,
                }
            ],
            "dataset": {
                "id": "synthetic-first-run",
                "samples": [
                    {
                        "sample_id": f"sample-{i:02d}",
                        "prompt_group_id": f"group-{i:02d}",
                        "token_ids": [i + 1, i + 2, i + 3],
                        "split": "test",
                        "target_token_id": 1,
                        "reference_token_id": 0,
                    }
                    for i in range(8)
                ],
            },
            "metrics": [
                "core.l0",
                "core.reconstruction_mse",
                "intervention.logit_delta_l2",
            ],
            "conditions": [
                {"name": "identity", "operation": "identity"},
                {
                    "name": "ablate_feature_0",
                    "operation": "ablate",
                    "feature_ids": [0],
                },
            ],
            "seed": 17,
            "max_measurements": 64,
            "max_activation_rows": 64,
            "microbatch": 8,
            "sample_batch": 4,
            "options": {"tokenizer_sha256": "a" * 64},
        }
    )
    return config, model, saes


def run(output: Path, *, dry_run: bool = False) -> dict:
    """Generate persistent M3/M4 outputs and verify their read-only ledgers."""
    config, model, saes = make_experiment()
    plan = build_plan(config)

    if dry_run:
        return {
            "mode": "dry-run",
            "data": "synthetic",
            "run_id": config.run_id,
            "model_sae_pairs": plan["sae_evaluations"],
            "independent_prompt_groups": len(config.samples),
            "metric_jobs": plan["metric_jobs"],
            "planned_measurements": plan["planned_measurements"],
        }

    output.mkdir(parents=True, exist_ok=True)

    def load(model_key: str, sae_key: str, layer: int) -> SimpleNamespace:
        assert model_key == "synthetic-toy" and layer == SITE.layer
        return SimpleNamespace(model=model, sae=saes[sae_key])

    rows = execute(config, output, real=True, loader=load)
    folder = output / config.run_id
    assert rows == read_metrics(folder, config.run_id)
    assert rows and all(row["status"] == "completed" for row in rows)

    audit_config = AuditConfig(
        evaluation_seeds=(0, 1, 2),
        bootstrap_replicates=64,
        sample_sizes=(4, 8),
        metric_ids=("core.l0", "intervention.logit_delta_l2"),
        primary_comparisons=(("toy-a", "toy-b"),),
    )
    audit = evaluate_m3(folder, audit_config)
    write_audit(folder, audit)
    assert read_audit(folder) == audit
    write_report(folder, config, plan, rows)
    required = (
        "metrics.jsonl",
        "metrics_manifest.json",
        "resolved_config.json",
        "provenance.json",
        "summary.csv",
        "report.html",
        "reliability.json",
        "reliability_manifest.json",
        "reliability_summary.csv",
        "reliability.html",
    )
    if any(not (folder / name).is_file() for name in required):
        raise RuntimeError("first experiment did not write all expected artifacts")

    return {
        "mode": "completed",
        "data": "synthetic",
        "run_id": config.run_id,
        "model_sae_pairs": plan["sae_evaluations"],
        "independent_prompt_groups": len(config.samples),
        "metric_rows": len(rows),
        "completed_metric_rows": sum(row["status"] == "completed" for row in rows),
        "reliability_status": audit["status"],
        "paired_comparisons": len(audit["paired_comparisons"]),
        "output_dir": str(folder.resolve()),
        "report_html": str((folder / "report.html").resolve()),
        "metrics_csv": str((folder / "summary.csv").resolve()),
        "reliability_json": str((folder / "reliability.json").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/first-experiment"),
        help="Directory for persistent benchmark and cache artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the exact execution plan without writing files",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, dry_run=args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
