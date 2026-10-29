"""Candidate worker. Must be executed only inside an OS-restricted reviewer sandbox."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    source = Path(sys.argv[1]).resolve(strict=True)
    payload = json.load(sys.stdin)
    original = copy.deepcopy(payload)
    workdir = Path.cwd()
    try:
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
        ok, kind, result = False, "ValueError" if isinstance(exc, ValueError) else type(exc).__name__, None
    files = {}
    for path in workdir.rglob("*"):
        if path.is_symlink():
            ok, kind = False, "UnsafeFile"
        elif path.is_file() and ".git" not in path.relative_to(workdir).parts:
            files[path.relative_to(workdir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"ok": ok, "kind": kind, "output": result,
                      "mutated": payload != original, "files": dict(sorted(files.items()))},
                     allow_nan=False, ensure_ascii=False))


if __name__ == "__main__":
    main()
