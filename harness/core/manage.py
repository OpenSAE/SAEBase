"""Core project graph proposals; approval is deliberately explicit."""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.knowledge import manage as engine

if __name__ == "__main__":
    engine.DEFAULT = ROOT / "harness/core/project_graph.json"
    engine.PROPOSALS = ROOT / "harness/core/proposals"
    raise SystemExit(engine.main())
