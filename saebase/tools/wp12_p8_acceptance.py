#!/usr/bin/env python3
"""P8 independent acceptance report from actual 50-item evidence and raw archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from saebase.wp12.acceptance import audit_ledger, inspect_archive, render_progress


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path("harness/github_work_plan/12_capability_acceptance_matrix.csv"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=None)
    p.add_argument("--archive-sha256", type=str, default=None)
    p.add_argument("--require-release", action="store_true")
    args = p.parse_args(argv)
    if (args.archive is None) != (args.archive_sha256 is None):
        p.error("--archive and --archive-sha256 must be supplied together")
    report = audit_ledger(args.ledger)
    if args.archive is not None:
        report["independent_archive_check"] = inspect_archive(
            args.archive, expected_sha256=args.archive_sha256
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "acceptance.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "progress.md").write_text(render_progress(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "release_ready": report["release_ready"],
                "capabilities": report["capabilities"],
                "missing_acceptance": len(report["missing"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 1 if args.require_release and not report["release_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
