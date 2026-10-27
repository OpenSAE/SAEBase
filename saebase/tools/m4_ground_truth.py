#!/usr/bin/env python3
"""Run independent analytical and known-factor SAR/RSS ground-truth controls."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from saebase.reliability.schema import AuditConfig
from saebase.reliability.ground_truth_report import evaluate_ground_truth
from saebase.benchmarks.results import atomic_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sar-source", type=Path, required=True)
    p.add_argument("--rss-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    conf = AuditConfig.load(args.config)
    result = evaluate_ground_truth(
        sar_source=args.sar_source, rss_source=args.rss_source, config=conf
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "case_counts": {
                    "factors": len(result["factor_cases"]),
                    "known_order_interactions": len(result["known_interaction_cases"]),
                    "rss_directions": 1,
                    "negative_controls": len(result["negative_invariance_controls"]),
                },
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
