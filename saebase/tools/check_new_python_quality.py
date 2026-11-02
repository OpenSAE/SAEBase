#!/usr/bin/env python3
"""Format new Python files before commit, then lint them with the CI Ruff version.

Scope defaults to Git-staged *added* files. Formatting other tracked modules
requires an explicit --include-modified flag to avoid unrelated large diffs.
CI calls check mode and stays read-only; local pre-commit calls --fix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

RUFF_VERSION = "0.15.22"
FROZEN_PREFIX = "harness/tests/wp21/fixtures/"


def added_python_files(
    base_sha: str | None = None, *, include_modified: bool = False
) -> list[str]:
    """Find only staged or PR-changed Python files; never scan all research code."""
    kinds = "AM" if include_modified else "A"
    command = ["git", "diff", "--name-only", f"--diff-filter={kinds}", "-z"]
    if base_sha:
        command.append(f"{base_sha}...HEAD")
    else:
        command.append("--cached")
    command.append("--")
    output = subprocess.run(command, check=True, capture_output=True).stdout
    paths = [part.decode("utf-8") for part in output.split(b"\0") if part]
    return sorted(
        path
        for path in paths
        if path.endswith(".py")
        and not path.startswith(FROZEN_PREFIX)
        and not Path(path).is_symlink()
        and Path(path).is_file()
    )


def verify_ruff_version() -> None:
    """Match the exact Ruff binary used by lightweight GitHub CI."""
    try:
        result = subprocess.run(["ruff", "--version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Ruff unavailable; install: python -m pip install ruff=={RUFF_VERSION}"
        ) from exc
    if not re.fullmatch(rf"ruff {re.escape(RUFF_VERSION)}", result.stdout.strip()):
        raise RuntimeError(
            f"Expected ruff {RUFF_VERSION}, found {result.stdout.strip()!r}. "
            f"Install: python -m pip install ruff=={RUFF_VERSION}"
        )


def verify_new_files(
    base_sha: str | None = None, *, fix: bool = False, include_modified: bool = False
) -> int:
    """Format selected files when requested, then run non-mutating checks."""
    files = added_python_files(base_sha, include_modified=include_modified)
    if not files:
        print("No selected Python files to format/check.")
        return 0
    verify_ruff_version()
    print("Selected Python files: " + ", ".join(files), flush=True)
    if fix:
        print("Running Ruff formatter before commit (selected paths only).", flush=True)
        status = subprocess.run(["ruff", "format", *files], check=False).returncode
        if status:
            return status
    failed = False
    for command in (
        ["ruff", "check", *files],
        ["ruff", "format", "--check", *files],
    ):
        status = subprocess.run(command, check=False).returncode
        if status:
            failed = True
            if command[1] == "format":
                subprocess.run(["ruff", "format", "--diff", *files], check=False)
    if fix:
        print("Review changed files and re-stage ONLY the authorized paths before committing.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", default=None, help="Existing PR base commit; otherwise staged")
    parser.add_argument("--fix", action="store_true", help="Autoformat selected files before checking")
    parser.add_argument(
        "--include-modified",
        action="store_true",
        help="Also include modified Python files (explicit full-file reformat consent)",
    )
    args = parser.parse_args()
    if args.fix and args.base_sha:
        parser.error("--fix --base-sha is disallowed: never rewrite committed PR sources")
    try:
        return verify_new_files(
            args.base_sha, fix=args.fix, include_modified=args.include_modified
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        parser.exit(2, f"Python quality preparation blocked: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
