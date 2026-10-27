#!/usr/bin/env python3
"""Real multi-checkpoint release matrix: deterministic preflight and verified execution."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from saebase.release import load_matrix, plan_matrix
from saebase.release.runner import execute_matrix, verify_matrix


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", default="release-runs", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    kind = p.add_mutually_exclusive_group(required=True)
    kind.add_argument("--dry-run", action="store_true")
    kind.add_argument("--real", action="store_true")
    kind.add_argument("--report-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)
    matrix = load_matrix(args.config)
    if args.dry_run:
        print(json.dumps(plan_matrix(matrix), indent=2))
        return 0
    if args.report_only:
        result = verify_matrix(args.output / matrix.matrix_id, expected_id=matrix.matrix_id)
    else:
        result = execute_matrix(
            matrix, args.output, device=args.device, real=True, resume=not args.no_resume
        )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "matrix_id": matrix.matrix_id,
                "status": result["status"],
                "real_checkpoints": result["real_checkpoints"],
                "model_families": result["families"],
                "checkpoint_count": result["n_distinct_saes"],
                "evidence": str(args.output / matrix.matrix_id / "matrix_results.json"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
