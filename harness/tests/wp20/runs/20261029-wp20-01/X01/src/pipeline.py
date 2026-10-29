from __future__ import annotations

import ast
from pathlib import Path

from src.schema import digest, require_keys


def _functions(source):
    if not isinstance(source, str):
        raise ValueError("source")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("Invalid Python source") from exc
    return {
        node.name: {"signature": ast.unparse(node.args), "line": node.lineno}
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def execute(payload, workdir: Path):
    require_keys(payload, {"before", "after"})
    before = _functions(payload["before"])
    after = _functions(payload["after"])
    return {
        "before": before,
        "after": after,
        "diff": {
            "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys()),
            "changed": sorted(
                name
                for name in before.keys() & after.keys()
                if before[name]["signature"] != after[name]["signature"]
            ),
        },
        "evidence": {
            "before_sha256": digest(payload["before"]),
            "after_sha256": digest(payload["after"]),
            "review": "unchecked",
        },
    }
