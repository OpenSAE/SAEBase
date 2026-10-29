"""Reviewer-only subprocess adapter for frozen WP18 task interfaces.

Executed only in an isolated, disposable reviewer CI/worktree. This subprocess
reduces accidental interference but is NOT an OS security sandbox.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]).resolve(strict=True)
    data = json.load(sys.stdin)
    saved = copy.deepcopy(data)
    try:
        spec = importlib.util.spec_from_file_location("wp18_submission", path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load submitted module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solve = getattr(module, "solve")
        if not callable(solve):
            raise ValueError("Missing callable solve")
        output = solve(data)
        if not isinstance(output, dict):
            raise ValueError("solve must return a dict")
        json.dumps(output, allow_nan=False, sort_keys=True)
        result = {"ok": True, "output": output, "mutated": data != saved}
    except Exception as exc:
        result = {
            "ok": False,
            "kind": "ValueError" if isinstance(exc, ValueError) else type(exc).__name__,
            "mutated": data != saved,
        }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
