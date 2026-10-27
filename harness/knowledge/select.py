"""Select bounded Knowledge or Core node projections without dumping whole graphs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "doc/knowledge/context_graph.json"

def project(data, ids, depth=0, max_nodes=12):
    lookup = {n["id"]: n for n in data["nodes"]}
    missing = [key for key in ids if key not in lookup]
    if missing:
        raise ValueError("Unknown node IDs: "+", ".join(missing))
    chosen = set(ids)
    if depth == 1:
        seeds = set(chosen)
        for edge in data["edges"]:
            if edge["source"] in seeds or edge["target"] in seeds:
                chosen.add(edge["source"])
                chosen.add(edge["target"])
    if len(chosen) > max_nodes:
        raise ValueError(f"Projection has {len(chosen)} nodes (maximum {max_nodes}); choose fewer IDs")
    return {
        "kind": data.get("kind", "knowledge") + "_projection",
        "schema_version": 1,
        "nodes": [lookup[k] for k in sorted(chosen)],
        "edges": [e for e in data["edges"] if e["source"] in chosen and e["target"] in chosen],
    }

def main(default_graph=GRAPH):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--depth", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-nodes", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.max_nodes <= 40:
        parser.error("--max-nodes must be between 1 and 40")
    try:
        graph = json.loads(Path(default_graph).read_text(encoding="utf-8"))
        result = project(graph, args.ids, args.depth, args.max_nodes)
    except (OSError, ValueError, KeyError) as exc:
        print("INVALID projection:", exc, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
