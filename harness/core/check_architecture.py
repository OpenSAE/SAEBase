"""Diff-based repository ownership check for new and renamed files."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {
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
OLD = (
    "harness/github_work_plan/",
    "harness/github_work_test/",
    "harness/github_development_doc/",
    "harness/github_assets/",
)


def validate_path(name: str, roots: set[str]) -> list[str]:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or str(p) != name:
        return ["Noncanonical path: " + name]
    errors = []
    if any(name.startswith(x) for x in OLD):
        errors.append("Legacy GitHub Harness directory: " + name)
    if len(p.parts) == 1:
        if name not in ROOT_FILES:
            errors.append("Unapproved root file: " + name)
    elif p.parts[0] not in roots:
        errors.append("New top-level module needs architecture approval: " + name)
    if (
        p.parts[0] in {"scripts", "tests"}
        and len(p.parts) == 2
        and p.name not in {"README.md", "README.en.md"}
    ):
        errors.append("Group new scripts/tests below their owned category: " + name)
    if p.parts[0] == "datasets" and p.name.startswith("qa") and p.name.endswith("_runner.py"):
        errors.append("Study-specific QA orchestration belongs to scripts/experiments/: " + name)
    if p.parts[0] in {"datasets", "models", "tasks"} and p.suffix == ".sh":
        errors.append("Shell launcher belongs to scripts/: " + name)
    return errors


def validate_imports(path: Path) -> list[str]:
    if path.parts[0] not in {"saebase", "datasets", "models", "tasks"} or path.suffix != ".py":
        return []
    if not (ROOT / path).is_file():
        return []
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(x == "experiments" or x.startswith("experiments.") for x in names):
            return ["Shared module imports a particular experiment: " + path.as_posix()]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    roots = set(json.loads((ROOT / ".cursor/policy.json").read_text())["approved_roots"])
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", args.base + "...HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    errors = []
    for line in diff.stdout.splitlines():
        columns = line.split("\t")
        if columns[0].startswith(("A", "R", "C")):
            path = columns[-1]
            errors.extend(validate_path(path, roots))
            errors.extend(validate_imports(Path(path)))
    for error in errors:
        print("FAIL:", error)
    if not errors:
        print("PASS: new file placement and import direction")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
