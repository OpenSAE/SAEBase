"""Validate the authoritative knowledge graph without importing any training code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import copy

if __package__:
    from .status import STATES, PRIMARY, refresh_status
else:
    from status import STATES, PRIMARY, refresh_status

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "doc/knowledge/context_graph.json"
KINDS = {"question", "solve", "goal"}
RELATIONS = PRIMARY
ALLOWED = {
    "refines": {("question", "question")},
    "addresses": {("question", "solve")},
    "decomposes": {("solve", "goal"), ("goal", "goal")},
}


class GraphError(ValueError):
    pass


def fail(message: str):
    raise GraphError(message)


def acyclic(nodes: set[str], edges: list[dict], types: set[str]):
    graph = {key: [] for key in nodes}
    for edge in edges:
        if edge["type"] in types:
            graph[edge["source"]].append(edge["target"])
    color = {}

    def visit(node: str):
        if color.get(node) == 1:
            fail(f"Cycle in {','.join(sorted(types))} starting at {node}")
        if color.get(node) == 2:
            return
        color[node] = 1
        for child in graph[node]:
            visit(child)
        color[node] = 2

    for key in sorted(nodes):
        visit(key)


def validate(graph: dict) -> dict:
    if not isinstance(graph, dict) or graph.get("schema_version") != 1:
        fail("Unknown knowledge graph schema")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if "gates" in graph:
        fail("Logic gates are no longer part of the tree schema")
    if not all(isinstance(v, list) for v in (nodes, edges)):
        fail("Graph requires nodes and edges lists")
    mapping = {}
    for obj in nodes:
        if not isinstance(obj, dict):
            fail("Node must be an object")
        key, kind, status = obj.get("id"), obj.get("type"), obj.get("status")
        if not isinstance(key, str) or not key or key in mapping:
            fail(f"Duplicate or empty node id: {key!r}")
        if kind not in KINDS or status not in STATES:
            fail(f"Invalid node type/status: {key}")
        if not isinstance(obj.get("title"), str) or not obj["title"].strip():
            fail(f"Missing node title: {key}")
        if not isinstance(obj.get("description"), str) or not obj["description"].strip():
            fail(f"Missing node description: {key}")
        evidence = obj.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(x, str) or not x for x in evidence):
            fail(f"Bad evidence: {key}")
        if status == "implemented" and not evidence:
            fail(f"Evidence required for implemented node: {key}")
        mapping[key] = obj
    seen = set()
    parent_of = {}
    for edge in edges:
        if not isinstance(edge, dict):
            fail("Edge must be an object")
        a, b, kind = edge.get("source"), edge.get("target"), edge.get("type")
        if a not in mapping or b not in mapping or a == b or kind not in RELATIONS:
            fail(f"Invalid edge: {edge}")
        if (mapping[a]["type"], mapping[b]["type"]) not in ALLOWED[kind]:
            fail(f"Illegal {kind} edge from {mapping[a]['type']} to {mapping[b]['type']}")
        if (a, b, kind) in seen:
            fail(f"Duplicate edge: {(a, b, kind)}")
        seen.add((a, b, kind))
        if b in parent_of:
            fail(f"Node {b} has multiple parents; this is a tree, not an execution graph")
        parent_of[b] = a
    acyclic(set(mapping), edges, PRIMARY)
    # Status is the independently audited state of each specific node.
    # Child partial/not_implemented does not rewrite a parent's status.
    # Hierarchy constraints remain enforced by the edge/cycle checks above.
    try:
        refresh_status(copy.deepcopy(graph))
    except (ValueError, KeyError) as exc:
        fail(str(exc))
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "statuses": dict(
            (state, sum(n["status"] == state for n in nodes)) for state in sorted(STATES)
        ),
    }


def validate_scientific(graph: dict, glossary: dict, root: Path = ROOT) -> dict:
    """Validate both node logic and the one-definition-per-term dictionary."""
    stats = validate(graph)
    if graph.get("kind") != "knowledge" or graph.get("glossary") != "doc/knowledge/glossary.json":
        fail("Scientific Knowledge graph must identify its canonical glossary")
    if not isinstance(glossary, dict) or glossary.get("schema_version") != 1:
        fail("Unsupported glossary schema")
    categories = glossary.get("categories", [])
    terms = glossary.get("terms", [])
    if not isinstance(categories, list) or not isinstance(terms, list):
        fail("Glossary categories and terms must be lists")
    category_ids = [c.get("id") for c in categories if isinstance(c, dict)]
    if len(set(category_ids)) != len(category_ids) or not category_ids:
        fail("Duplicate or empty glossary category")
    by_term = {}
    by_node = {n["id"]: n for n in graph["nodes"]}
    backlink = {}
    for entry in terms:
        if not isinstance(entry, dict):
            fail("Glossary entry must be an object")
        key = entry.get("id")
        if not isinstance(key, str) or not key.startswith("term.") or key in by_term:
            fail(f"Duplicate or invalid glossary term: {key}")
        for field in ("term", "definition", "why_it_matters", "foundation"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                fail(f"Missing glossary {field}: {key}")
        if entry.get("category") not in category_ids:
            fail(f"Unknown category: {key}")
        if not isinstance(entry.get("aliases"), list) or not all(
            isinstance(a, str) and a for a in entry["aliases"]
        ):
            fail(f"Invalid aliases: {key}")
        if not isinstance(entry.get("source_evidence"), list) or not entry["source_evidence"]:
            fail(f"Missing glossary source evidence: {key}")
        for filename in [*entry["source_evidence"], entry["foundation"]]:
            if (
                not isinstance(filename, str)
                or filename.startswith("/")
                or ".." in filename.split("/")
            ):
                fail(f"Unsafe glossary evidence path: {key}")
            if not (root / filename).is_file():
                fail(f"Missing glossary evidence file: {key}: {filename}")
        refs = entry.get("related_node_ids")
        if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
            fail(f"Invalid or unused glossary term: {key}")
        if any(x not in by_node for x in refs):
            fail(f"Unknown related knowledge node in {key}")
        by_term[key] = entry
        backlink[key] = set(refs)
    forward = {key: set() for key in by_term}
    for node in graph["nodes"]:
        term_ids = node.get("term_ids")
        if not isinstance(term_ids, list) or not term_ids or len(set(term_ids)) != len(term_ids):
            fail(f"Knowledge node lacks a canonical, non-duplicated terminology list: {node['id']}")
        for tid in term_ids:
            if tid not in by_term:
                fail(f"Undefined specialized term {tid} in {node['id']}")
            forward[tid].add(node["id"])
    for term_id in by_term:
        if forward[term_id] != backlink[term_id]:
            fail(f"Glossary and Knowledge backlinks disagree: {term_id}")
    stats["glossary_terms"] = len(by_term)
    stats["glossary_categories"] = len(category_ids)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT))
    args = parser.parse_args()
    try:
        graph = json.loads(Path(args.path).read_text(encoding="utf-8"))
        glossary = json.loads((ROOT / "doc/knowledge/glossary.json").read_text(encoding="utf-8"))
        result = validate_scientific(graph, glossary)
    except (GraphError, OSError, ValueError) as e:
        print(f"INVALID knowledge graph: {e}", file=sys.stderr)
        return 1
    print("VALID knowledge graph:", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
