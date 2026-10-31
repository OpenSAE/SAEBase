#!/usr/bin/env python3
"""Check newly added Python code while protecting byte-frozen WP21 exam fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SUITE_PATH = Path("harness/tests/wp21/suite_lock.json")
FROZEN_DIGEST = "6b9fc5bdfc924529778403d9e871f45ad942246fa0b3ac4dae3b1f69611462e3"
FROZEN_PREFIX = "harness/tests/wp21/fixtures/"


def added_python_files(base_sha: str) -> list[str]:
    command = ["git", "diff", "--name-only", "--diff-filter=A", "-z", f"{base_sha}...HEAD", "--"]
    result = subprocess.run(command, check=True, capture_output=True)
    return sorted(
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path and path.endswith(b".py") and Path(path.decode("utf-8")).is_file()
    )


def frozen_sources() -> dict[str, str]:
    if not SUITE_PATH.is_file():
        return {}
    lock = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    if lock.get("files_sha256") != FROZEN_DIGEST or len(lock.get("files", {})) != 68:
        raise ValueError("WP21 frozen source manifest changed")
    result = subprocess.run(
        ["python", "harness/tests/wp21/tools/manage.py", "check-spec"],
        check=False,
    )
    if result.returncode:
        raise ValueError("WP21 frozen source integrity verification failed")
    return lock["files"]


def check_added(base_sha: str) -> int:
    files = added_python_files(base_sha)
    frozen = frozen_sources()
    checked = []
    untouched = []
    for path in files:
        if path.startswith(FROZEN_PREFIX) and path in frozen:
            observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if observed != frozen[path]:
                raise ValueError(f"frozen WP21 fixture changed: {path}")
            untouched.append(path)
        else:
            checked.append(path)
    print(f"Checked frozen WP21 fixtures by SHA256: {len(untouched)}", flush=True)
    print(f"New nonfrozen Python files to lint/format: {len(checked)}", flush=True)
    if not checked:
        return 0
    failed = False
    for command in (["ruff", "check", *checked], ["ruff", "format", "--check", *checked]):
        status = subprocess.run(command, check=False).returncode
        if status:
            failed = True
            if command[1] == "format":
                subprocess.run(["ruff", "format", "--diff", *checked], check=False)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    try:
        return check_added(args.base_sha)
    except (OSError, ValueError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        parser.exit(2, f"WP21 scoped Python quality check failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
