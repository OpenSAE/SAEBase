#!/usr/bin/env python3
"""Run one real WP12 private SAE job on your own authenticated GPU host.

One iteration only. Does not launch a public service, download datasets
implicitly, or simulate an inference result. Configure OPENSAE_PRIVATE_DATA
and OPENSAE_ENABLE_PRIVATE_JOBS=1 before invoking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    arguments = parser.parse_args(argv)
    root = os.environ.get("OPENSAE_PRIVATE_DATA")
    if not root or not Path(root).is_absolute() or not Path(root).is_dir():
        parser.error("An existing absolute OPENSAE_PRIVATE_DATA directory is required")
    if os.environ.get("OPENSAE_ENABLE_PRIVATE_JOBS") != "1":
        parser.error("OPENSAE_ENABLE_PRIVATE_JOBS must be explicitly set to 1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "research-lab"))
    from backend.private_worker import execute_one

    report = execute_one(root=Path(root), device=arguments.device)
    print(json.dumps(report, sort_keys=True))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
