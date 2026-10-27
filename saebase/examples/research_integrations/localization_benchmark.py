"""End-to-end Localize Benchmark + Reliability on a controlled synthetic SAE grid.

Install first:
    pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git

Run:
    python saebase/examples/research_integrations/localization_benchmark.py --output runs

No pretrained model download. Outputs an SAEBase benchmark report and reliability audit
computed with the original independent Localize estimators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ModelAdapter
from saebase.data import BinaryFactor, ControlledInputDesign
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.runner import execute
from saebase.reliability import AuditConfig, evaluate_m3, write_audit
from saebase.benchmarks.report import write_report


def build_demo() -> tuple[BenchmarkConfig, ModelAdapter, dict[str, ResearchToySAE]]:
    design = ControlledInputDesign.from_renderer(
        name="three-factor-example",
        factors=(
            BinaryFactor("first", ("1", "2")),
            BinaryFactor("second", ("3", "4")),
            BinaryFactor("third", ("5", "6")),
        ),
        render=lambda x: f"{x['first']} {x['second']} {x['third']}",
    )
    model = ModelAdapter(ResearchToyModel().eval())
    records = design.sample_records(model)
    a = ResearchToySAE().eval()
    b = ResearchToySAE().eval()
    with torch.no_grad():
        b.bias.add_(0.1)  # distinct SAE checkpoint identity
    data = {
        "schema": "saebase.benchmark.config.v1",
        "name": "Input Localization benchmark demonstration",
        "targets": [
            {
                "model_key": "synthetic-toy",
                "sae_keys": ["toy-a", "toy-b"],
                "hook_name": "blocks.0.hook_resid_post",
                "layer": 0,
            }
        ],
        "dataset": {"id": "toy-three-factor-grid", "samples": [row.to_dict() for row in records]},
        "metrics": ["localization.budget_at_90pct", "localization.budget_fraction_at_90pct"],
        "conditions": [],
        "options": {
            "tokenizer_sha256": hashlib.sha256(b"ResearchToyModel.to_tokens").hexdigest(),
            "localization": {
                "name": design.name,
                "factors": [x.to_dict() for x in design.factors],
                "sample_ids_by_state": [x.sample_id for x in design.states],
                "pairs_per_scope": 64,
            },
        },
    }
    return BenchmarkConfig.from_dict(data), model, {"toy-a": a, "toy-b": b}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg, model, saes = build_demo()
    if args.dry_run:
        print(json.dumps(build_plan(cfg), indent=2))
        return
    rows = execute(
        cfg,
        args.output,
        real=True,
        loader=lambda mk, sk, layer: SimpleNamespace(model=model, sae=saes[sk]),
    )
    root = args.output / cfg.run_id
    audit_cfg = AuditConfig(
        evaluation_seeds=(0, 1, 2, 3, 4),
        bootstrap_replicates=64,
        metric_ids=("localization.budget_at_90pct",),
    )
    audit = evaluate_m3(root, audit_cfg)
    write_audit(root, audit)
    write_report(root, cfg, build_plan(cfg), rows)
    print(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "metrics": [
                    {k: row[k] for k in ("metric_id", "sae_id", "status", "value")} for row in rows
                ],
                "localization_comparisons": audit["localization_comparisons"],
                "report": str(root / "report.html"),
                "curves": str(root / "localization"),
                "reliability": str(root / "reliability.json"),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
