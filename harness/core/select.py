"""Select only requested Core project-management node IDs."""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.knowledge.select import main

if __name__ == "__main__":
    raise SystemExit(main(default_graph=ROOT / "harness/core/project_graph.json"))
