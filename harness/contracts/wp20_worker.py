"""Candidate worker. Must be executed only inside an OS-restricted reviewer sandbox."""

from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys


def main():
    source = Path(sys.argv[1]).resolve(strict=True)
    payload = json.load(sys.stdin)
    original = copy.deepcopy(payload)
    workdir = Path.cwd()
    try:
        sys.path.insert(0, str(source.parent))
        spec = importlib.util.spec_from_file_location("candidate_wp20", source)
        if not spec or not spec.loader:
            raise ValueError("Candidate unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run(payload, workdir)
        if not isinstance(result, dict):
            raise ValueError("Non-object candidate result")
        json.dumps(result, allow_nan=False)
        ok, kind = True, None
    except Exception as exc:
        ok, kind, result = (
            False,
            "ValueError" if isinstance(exc, ValueError) else type(exc).__name__,
            None,
        )
    files = {}
    for path in workdir.rglob("*"):
        if path.is_symlink():
            ok, kind = False, "UnsafeFile"
        elif path.is_file() and ".git" not in path.relative_to(workdir).parts:
            files[path.relative_to(workdir).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    evidence = None
    if (workdir / ".git").is_dir():

        def git(*args):
            result = subprocess.run(
                ["git", "-C", str(workdir), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": str(workdir),
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_OPTIONAL_LOCKS": "0",
                },
            )
            if result.returncode:
                raise ValueError("Cannot independently inspect Git history")
            return result.stdout.strip()

        try:
            head = git("rev-list", "--parents", "-n", "1", "HEAD").split()
            evidence = {
                "base_sha": git("merge-base", "agent-a", "agent-b"),
                "head_a_sha": git("rev-parse", "agent-a"),
                "head_b_sha": git("rev-parse", "agent-b"),
                "merge_sha": head[0],
                "merge_parents": head[1:],
                "commits": int(git("rev-list", "--count", "--all")),
                "clean": not bool(git("status", "--porcelain")),
            }
        except (ValueError, subprocess.TimeoutExpired):
            evidence = None
    print(
        json.dumps(
            {
                "git_evidence": evidence,
                "ok": ok,
                "kind": kind,
                "output": result,
                "mutated": payload != original,
                "files": dict(sorted(files.items())),
            },
            allow_nan=False,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
