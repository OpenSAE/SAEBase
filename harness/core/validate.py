"""Validate the project-management graph independently of scientific Knowledge."""

from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.knowledge.validate import validate, GraphError

GRAPH = ROOT / "harness/core/project_graph.json"


def main():
    try:
        result = validate(json.loads(GRAPH.read_text(encoding="utf-8")))
    except (OSError, ValueError, GraphError) as exc:
        print("INVALID Core project-management tree:", exc, file=sys.stderr)
        return 1
    print("VALID Core project-management tree:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
