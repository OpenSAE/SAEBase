"""Format selected source changes and select CI checks from changed paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

RUFF_VERSION = "0.15.22"

FROZEN_PREFIX = "harness/tests/wp21/fixtures/"

MAX_API_SOURCE_CHARS = 2_000_000


def added_python_files(base_sha: str | None = None, *, include_modified: bool = False) -> list[str]:
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


def format_api_source(source: str, filename: str) -> str:
    """Return Ruff-formatted, lint-clean Python for ONE pending GitHub API write.

    Input/output use stdio and memory only: this function never stages, rewrites,
    commits, or pushes a file. The caller still reviews the final exact-file diff
    and separately obtains authorization to change the source.
    """
    if (
        not isinstance(filename, str)
        or not filename.endswith(".py")
        or filename.startswith("-")
        or "\\" in filename
        or PurePosixPath(filename).is_absolute()
        or PurePosixPath(filename).as_posix() != filename
        or any(piece in ("", ".", "..") for piece in filename.split("/"))
        or filename.startswith(FROZEN_PREFIX)
    ):
        raise ValueError("Expected a safe, unfrozen repository-relative .py filename")
    if not isinstance(source, str) or len(source) > MAX_API_SOURCE_CHARS:
        raise ValueError("Python input is missing or exceeds the 2 MB preparation limit")

    verify_ruff_version()
    formatted = subprocess.run(
        ["ruff", "format", "--stdin-filename", filename, "-"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
    )
    if formatted.returncode:
        raise RuntimeError("Ruff formatting failed: " + formatted.stderr.strip()[-1200:])
    result = formatted.stdout
    for command in (
        ["ruff", "check", "--stdin-filename", filename, "-"],
        ["ruff", "format", "--check", "--stdin-filename", filename, "-"],
    ):
        verified = subprocess.run(
            command,
            input=result,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
            encoding="utf-8",
            cwd=Path(__file__).resolve().parents[1],
        )
        if verified.returncode:
            detail = (verified.stdout + verified.stderr).strip()[-1200:]
            raise RuntimeError(f"Ruff verification failed ({command[1]}): {detail}")
    return result


def format_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-sha", default=None, help="Existing PR base commit; otherwise staged"
    )
    parser.add_argument(
        "--stdin", action="store_true", help="Format one source from stdin for GitHub API"
    )
    parser.add_argument("--filename", help="Exact repository-relative .py path for --stdin")
    parser.add_argument(
        "--fix", action="store_true", help="Autoformat selected files before checking"
    )
    parser.add_argument(
        "--include-modified",
        action="store_true",
        help="Also include modified Python files (explicit full-file reformat consent)",
    )
    args = parser.parse_args()
    if args.stdin:
        if not args.filename or args.fix or args.base_sha or args.include_modified:
            parser.error("--stdin requires --filename and cannot combine with Git or fix options")
        try:
            binary_in = getattr(sys.stdin, "buffer", None)
            if binary_in is not None:
                source = binary_in.read(MAX_API_SOURCE_CHARS * 4 + 1).decode("utf-8")
            else:
                source = sys.stdin.read(MAX_API_SOURCE_CHARS + 1)
            prepared = format_api_source(source, args.filename)
            binary_out = getattr(sys.stdout, "buffer", None)
            if binary_out is not None:
                binary_out.write(prepared.encode("utf-8"))
            else:
                sys.stdout.write(prepared)
            return 0
        except (ValueError, RuntimeError, OSError, UnicodeError, subprocess.SubprocessError) as exc:
            parser.exit(2, f"GitHub API Python source preparation blocked: {exc}\n")
    if args.filename:
        parser.error("--filename is only valid with --stdin")
    if args.fix and args.base_sha:
        parser.error("--fix --base-sha is disallowed: never rewrite committed PR sources")
    try:
        return verify_new_files(args.base_sha, fix=args.fix, include_modified=args.include_modified)
    except (RuntimeError, OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        parser.exit(2, f"Python quality preparation blocked: {exc}\n")


SHA = re.compile(r"^[0-9a-f]{40}$")


def classify(paths: Iterable[str], *, unknown_diff: bool = False) -> dict[str, bool]:
    """Union checks for every changed path; no single-label shortcuts."""
    result = {"harness": False, "atlas": False, "product": False, "ci_policy": False}
    if unknown_diff:
        return {key: True for key in result}
    for path in paths:
        if not path or path.startswith("/") or "\\" in path or ".." in PurePosixPath(path).parts:
            return {key: True for key in result}
        if path.lower().endswith(".md"):
            continue
        if (
            path.startswith(
                (
                    "harness/core/",
                    "harness/github/",
                    "harness/contracts/",
                    "harness/environment/",
                    ".codex/",
                    ".agents/",
                    ".researchharness/",
                    ".github/skills/",
                )
            )
            or path == "AGENTS.md"
            or path.endswith("/AGENTS.md")
        ):
            result["harness"] = True
        elif path.startswith("doc/code/development/"):
            continue
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
        if path in {
            "scripts/release.py",
            "scripts/distribution.py",
        }:
            result["harness"] = True
            result["ci_policy"] = True
        elif path.startswith(
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
            or path == "scripts/quality.py"
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
                ".codex/",
                ".agents/",
                ".researchharness/",
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


def selection_main() -> int:
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


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        del sys.argv[1]
        return selection_main()
    return format_main()


if __name__ == "__main__":
    raise SystemExit(main())
