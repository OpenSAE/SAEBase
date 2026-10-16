#!/usr/bin/env python3
"""Gate newly added Python files without reformatting existing research code."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def added_python_files(base_sha: str | None = None) -> list[str]:
    """Return new tracked Python files in a PR or local staging area."""
    if base_sha:
        args = ["git", "diff", "--name-only", "--diff-filter=A", "-z"]
        args.extend([f"{base_sha}...HEAD", "--"])
    else:
        args = [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=A",
            "-z",
            "--",
        ]
    output = subprocess.run(args, check=True, capture_output=True).stdout
    paths = [name.decode("utf-8") for name in output.split(b"\0") if name]
    return sorted(path for path in paths if path.endswith(".py") and Path(path).is_file())


def verify_new_files(base_sha: str | None = None) -> int:
    """Lint and format-check new files; do not alter repository contents."""
    files = added_python_files(base_sha)
    if not files:
        print("No newly added Python files to check.")
        return 0
    print("Checking new Python files: " + ", ".join(files), flush=True)
    status = 0
    for command in (
        ["ruff", "check", *files],
        ["ruff", "format", "--check", *files],
    ):
        result = subprocess.run(command, check=False)
        if result.returncode:
            status = 1
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-sha",
        default=None,
        help="PR base commit, or HEAD^ for a push; omitted uses staged additions",
    )
    args = parser.parse_args()
    try:
        return verify_new_files(args.base_sha)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        parser.exit(2, f"Could not check added Python files: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
