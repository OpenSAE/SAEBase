"""Conservative diff-based selection for lightweight PR and main CI.

Unknown paths fail open to additional *lightweight* checks, never to release
publishing. Full release verification is an explicit SHA-bound workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import PurePosixPath
import re
import subprocess
from typing import Iterable

SHA = re.compile(r"^[0-9a-f]{40}$")


def classify(paths: Iterable[str], *, unknown_diff: bool = False) -> dict[str, bool]:
    """Union checks for every changed path; no single-label shortcuts."""
    result = {"harness": False, "atlas": False, "product": False, "ci_policy": False}
    if unknown_diff:
        return {key: True for key in result}
    for path in paths:
        if not path or path.startswith("/") or "\\" in path or ".." in PurePosixPath(path).parts:
            return {key: True for key in result}
        if path.startswith(
            (
                "harness/core/",
                "harness/github/",
                "harness/contracts/",
                "harness/environment/",
                ".cursor/",
                ".github/skills/",
            )
        ) or path in ("AGENTS.md",):
            result["harness"] = True
        elif path.startswith(("harness/", "doc/knowledge/", "doc/code/", "atlas/")):
            result["atlas"] = True
        if path.startswith(
            (
                "harness/visualization/",
                "harness/code/",
                "harness/knowledge/",
                "harness/core/",
                "doc/knowledge/",
                "doc/code/",
                "atlas/",
            )
        ):
            result["atlas"] = True
        if path.startswith(
            (
                "saebase/",
                "models/",
                "datasets/",
                "tasks/",
                "conf/",
                "tests/",
                "scripts/",
            )
        ) or path in ("pyproject.toml", "setup.py", "MANIFEST.in", "requirements.txt"):
            result["product"] = True
        if (
            path.startswith(".github/workflows/")
            or path.startswith("harness/github/ci/")
            or path in (".github/dependabot.yml",)
        ):
            result["ci_policy"] = True
            result["harness"] = True
        known = path.startswith(
            (
                "harness/",
                "doc/",
                "atlas/",
                ".github/",
                ".cursor/",
                "saebase/",
                "models/",
                "datasets/",
                "tasks/",
                "conf/",
                "tests/",
                "scripts/",
            )
        ) or path in {
            "AGENTS.md",
            "README.md",
            "README_cn.md",
            "LICENSE",
            "CITATION.cff",
            "MANIFEST.in",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            ".gitignore",
            ".dockerignore",
            ".cursorignore",
            ".nojekyll",
            ".pre-commit-config.yaml",
            "index.html",
        }
        if not known:
            return {key: True for key in result}
    return result


def changed_paths(base: str, head: str) -> list[str] | None:
    if not SHA.fullmatch(base) or not SHA.fullmatch(head):
        return None
    try:
        raw = subprocess.run(
            ["git", "diff", "--name-status", "-z", "--find-renames", base, head],
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    chunks = raw.decode("utf-8", errors="strict").split("\0")
    if chunks[-1] == "":
        chunks.pop()
    paths = []
    i = 0
    while i < len(chunks):
        flag = chunks[i]
        if not re.fullmatch(r"[ACDMRTUXB][0-9]*", flag):
            return None
        count = 2 if flag.startswith(("R", "C")) else 1
        if i + count >= len(chunks):
            return None
        paths.extend(chunks[i + 1 : i + 1 + count])
        i += count + 1
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()
    paths = changed_paths(args.base, args.head)
    flags = classify(paths or [], unknown_diff=paths is None)
    values = {key: str(value).lower() for key, value in flags.items()}
    values["changed_count"] = str(len(paths) if paths is not None else -1)
    print(json.dumps({"flags": flags, "changed_paths": paths}, indent=2))
    if args.output:
        with open(args.output, "a", encoding="utf-8") as file:
            for key, value in values.items():
                file.write(key + "=" + value + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
