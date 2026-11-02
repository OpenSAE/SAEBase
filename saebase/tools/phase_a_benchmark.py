"""Real Qwen3.5-2B SAEBase Benchmark and Reliability acceptance.

Run after (or separately from) phase_a_verify on a suitable CPU/GPU runner.
Downloads genuine model+SAE weights and produces checksum-bound M3/M4 evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from saebase.api import ModelAdapter, ActivationSite, capture_features
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.runner import execute
from saebase.benchmarks.report import write_report
from saebase.reliability import AuditConfig, evaluate_m3, write_audit, read_audit
from saebase.api.sae_loading import load_model_and_sae
from saebase.tools.phase_a_verify import TARGETS, PROMPTS


def run(output: Path, *, model_name: str = "qwen35_2b", device: str = "cpu") -> dict:
    if model_name != "qwen35_2b":
        raise ValueError("real M3/M4 acceptance is enabled for qwen35_2b only")
    sae_key, layer, d_model = TARGETS[model_name]
    raw_model, sae, meta = load_model_and_sae(sae_key, layer, device=device)
    model = ModelAdapter(raw_model)
    site = ActivationSite(layer, meta["hook_name"])
    selection = capture_features(model, sae, site=site, prompt=PROMPTS[0])
    chosen = int(torch.argmax(selection.codes.abs()).item())
    if not bool(torch.isfinite(selection.codes).all()) or float(selection.codes[chosen].abs()) == 0:
        raise ValueError("no live SAE feature to ablate")
    samples = []
    for i, prompt in enumerate(PROMPTS):
        tokens = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        samples.append(
            {
                "sample_id": f"real-qwen35-2b-{i:02d}",
                "token_ids": tokens,
                "prompt_group_id": f"real-sample-{i:02d}",
                "split": "test",
                "selection": {"mode": "indices", "indices": [len(tokens) - 1]},
            }
        )
    config = BenchmarkConfig.from_dict(
        {
            "schema": "saebase.benchmark.config.v1",
            "name": "Real Qwen3.5 2B Base with 32k TopK SAE",
            "targets": [
                {
                    "model_key": model_name,
                    "sae_keys": [sae_key],
                    "layer": layer,
                    "hook_name": site.hook_name,
                }
            ],
            "dataset": {"id": "saebase-phase-a-qwen35-2b-real", "samples": samples},
            "metrics": [
                "core.l0",
                "core.reconstruction_mse",
                "core.explained_variance",
                "intervention.logit_delta_l2",
            ],
            "conditions": [
                {"name": "identity", "operation": "identity"},
                {"name": "feature-ablation", "operation": "ablate", "feature_ids": [chosen]},
            ],
            "max_activation_rows": 24,
            "max_measurements": 24,
            "sample_batch": 1,
            "microbatch": 1,
            "seed": 0,
            "tolerance": 0.01,
        }
    )
    rows = execute(
        config,
        output,
        real=True,
        loader=lambda mk, sk, l: SimpleNamespace(model=model, sae=sae),
    )
    if not rows or any(row["status"] != "completed" for row in rows):
        raise ValueError("real benchmark has incomplete metrics")
    root = output / config.run_id
    acfg = AuditConfig(
        evaluation_seeds=(0, 1, 2),
        bootstrap_replicates=32,
        sample_sizes=(2, 3),
        metric_ids=("core.l0", "intervention.logit_delta_l2"),
    )
    audit = evaluate_m3(root, acfg)
    if any(row["status"] != "completed" for row in audit["metric_audits"]):
        raise ValueError("real M4 audit has incomplete metric audits")
    write_audit(root, audit)
    write_report(root, config, build_plan(config), rows)
    if read_audit(root) != audit:
        raise ValueError("real M4 audit failed integrity roundtrip")
    record = {
        "schema": "saebase.phase_a.real_m3m4_acceptance.v1",
        "status": "passed",
        "model": model_name,
        "sae_key": sae_key,
        "layer": layer,
        "feature_ablated": chosen,
        "model_loaded": True,
        "sae_loaded": True,
        "sample_count": len(samples),
        "benchmark_metric_rows": len(rows),
        "reliability_metric_audits": len(audit["metric_audits"]),
        "reliability_pairs": len(audit["paired_comparisons"]),
        "n_model_reloads_for_audit": 0,
        "metric_results": [
            {
                "metric_id": row["metric_id"],
                "condition": row["condition_id"],
                "value": row["value"],
                "n_valid": row["n_valid"],
            }
            for row in rows
        ],
        "m3_run_id": config.run_id,
        "m3_metrics_sha256": json.loads((root / "metrics_manifest.json").read_text())["sha256"],
        "m4_audit_sha256": json.loads((root / "reliability_manifest.json").read_text())[
            "audit_sha256"
        ],
    }
    (output / "real_qwen35_2b_m3m4_summary.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, default=Path("phase-a-m3m4"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.output, device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
