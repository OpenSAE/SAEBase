#!/usr/bin/env python3
"""Audit a completed SAE benchmark from its verified activation and intervention caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.benchmark.reliability.observations import _read_benchmark
from tasks.benchmark.reliability.runner import evaluate_benchmark
from tasks.benchmark.reliability.schema import AuditConfig
from tasks.benchmark.results.audit import read_audit, write_audit


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        "--m3-root",
        dest="benchmark_root",
        type=Path,
        required=True,
        help="completed benchmark directory with metrics and source caches",
    )
    p.add_argument("--config", type=Path, required=True)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report-only", action="store_true")
    p.add_argument(
        "--ground-truth",
        action="store_true",
        help="Run independent known-truth cases with original SAR/RSS estimators",
    )
    p.add_argument("--sar-source", type=Path, help="Trusted original SAR repository or sar.py")
    p.add_argument("--rss-source", type=Path, help="Trusted original RSS repository or rss.py")
    p.add_argument(
        "--sae-only",
        action="store_true",
        help="Opt in to SAE checkpoint loading for decoded Core metrics; never loads model",
    )
    args = p.parse_args(argv)
    if args.ground_truth and (not args.sar_source or not args.rss_source or not args.run):
        p.error("--ground-truth requires --run, --sar-source and --rss-source")
    cfg = AuditConfig.load(args.config)
    benchmark_config, rows, provenance = _read_benchmark(args.benchmark_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "saebase.m4.dry_run.v1",
                    "audit_id": cfg.config_id,
                    "m3_run_id": benchmark_config.run_id,
                    "selected_metrics": list(cfg.metric_ids),
                    "n_m3_metric_rows": len(rows),
                    "evaluation_seeds": list(cfg.evaluation_seeds),
                    "bootstrap_replicates_per_seed": cfg.bootstrap_replicates,
                    "model_weights_loaded": False,
                },
                indent=2,
            )
        )
        return 0
    if args.report_only:
        report = read_audit(args.benchmark_root)
        if AuditConfig.from_dict(report["audit_config"]).config_id != cfg.config_id:
            raise ValueError(
                "--report-only configuration differs from the stored reliability audit; use --run"
            )
    else:
        resolver = None
        if args.sae_only:
            from saebase.api.sae_loading import load_sae_for_profile

            def resolver(model_key, sae_key, layer):
                return load_sae_for_profile(sae_key, layer, device="cpu")[0]

        report = evaluate_benchmark(args.benchmark_root, cfg, sae_resolver=resolver)
        if args.ground_truth:
            from tasks.benchmark.reliability.controls import (
                append_ground_truth,
                evaluate_ground_truth,
            )

            truth = evaluate_ground_truth(
                sar_source=args.sar_source, rss_source=args.rss_source, config=cfg
            )
            report = append_ground_truth(report, truth)
        write_audit(args.benchmark_root, report)
    # Rebuild the benchmark report with the reliability section. This verifies
    # that the audit belongs to the unchanged source metric ledger.
    from tasks.benchmark.execution.planner import build_plan
    from tasks.benchmark.results.report import write_report

    write_report(
        args.benchmark_root,
        benchmark_config,
        build_plan(benchmark_config),
        rows,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "m3_run_id": benchmark_config.run_id,
                "n_metric_audits": len(report["metric_audits"]),
                "n_paired_comparisons": len(report["paired_comparisons"]),
                "model_forwards_during_audit": 0,
                "report": str(args.benchmark_root / "reliability.html"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
