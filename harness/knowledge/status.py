"""Knowledge status is local to each node, never implicitly inherited.

Child status roll-up is available as an explicit analytic view only, not as an
automatic mutation when a node is approved, denied, or partially implemented.
This prevents review proposals from rewriting the statuses of ancestor nodes.
"""
from __future__ import annotations

STATES = {"not_implemented", "partial", "implemented"}
PRIMARY = {"refines", "addresses", "decomposes"}


def child_status(states):
    """All complete -> implemented; none started -> not_implemented; else partial."""
    if not states:
        raise ValueError("Expected at least one child status")
    if not set(states) <= STATES:
        raise ValueError("Unknown state in child aggregate")
    if all(value == "implemented" for value in states):
        return "implemented"
    if all(value == "not_implemented" for value in states):
        return "not_implemented"
    return "partial"


def refresh_status(graph, *, propagate=False):
    """Validate the statuses without changing ancestors by default.

    An explicitly requested roll-up is retained for offline summary reports,
    but ordinary Knowledge mutations never turn ancestors partial.
    """
    nodes = {node["id"]: node for node in graph["nodes"]}
    for key, node in nodes.items():
        if node.get("status") not in STATES:
            raise ValueError("Unknown status " + str(node.get("status")) + " for " + key)
    if not propagate:
        return graph
    children = {}
    for edge in graph["edges"]:
        if edge["type"] not in PRIMARY:
            continue
        children.setdefault(edge["source"], []).append(edge["target"])
    visiting, memo = set(), {}

    def resolve(key):
        if key in memo:
            return memo[key]
        if key in visiting:
            raise ValueError("Hierarchy cycle at " + key)
        if key not in nodes:
            raise ValueError("Unknown hierarchy node " + key)
        visiting.add(key)
        values = children.get(key, [])
        state = child_status([resolve(child) for child in values]) if values else nodes[key]["status"]
        if state not in STATES:
            raise ValueError("Unknown state " + str(state) + " for " + key)
        nodes[key]["status"] = state
        memo[key] = state
        visiting.remove(key)
        return state

    for key in nodes:
        resolve(key)
    return graph
