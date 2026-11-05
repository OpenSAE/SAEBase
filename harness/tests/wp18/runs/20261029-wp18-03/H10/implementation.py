"""Stable DFS over an explicit multi-tree parent graph."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when the graph or the requested scope is not well formed."""


def solve(payload: dict) -> dict:
    """Return one DFS order for all, tree, parent, or node scope."""
    allowed = ({"nodes", "scope"}, {"nodes", "scope", "target"})
    if not isinstance(payload, dict) or set(payload) not in allowed:
        raise InputError("schema")
    scope = payload["scope"]
    if scope not in {"all", "tree", "parent", "node"}:
        raise InputError("scope")
    if scope == "all" and "target" in payload:
        raise InputError("target")
    if scope != "all" and not isinstance(payload.get("target"), str):
        raise InputError("target")
    nodes, children, roots = _graph(payload["nodes"])
    target = payload.get("target")
    if scope == "all":
        selected = roots
    elif scope == "tree":
        if target not in {row["tree_id"] for row in nodes.values()}:
            raise InputError("tree")
        selected = [node_id for node_id in roots if nodes[node_id]["tree_id"] == target]
    elif scope == "parent":
        if target not in nodes:
            raise InputError("node")
        selected = [target]
    else:
        if target not in nodes:
            raise InputError("node")
        return {"order": [target]}
    return {"order": _walk(selected, children)}


def _graph(
    values: object,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]], list[str]]:
    """Index one parent per node and reject missing, cross-tree, or cyclic parents."""
    if not isinstance(values, list):
        raise InputError("nodes")
    nodes: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", "parent_id", "tree_id", "order"}:
            raise InputError("node")
        node_id = value["id"]
        parent = value["parent_id"]
        tree_id = value["tree_id"]
        order = value["order"]
        if not isinstance(node_id, str) or not node_id or node_id in nodes:
            raise InputError("id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise InputError("parent")
        if not isinstance(tree_id, str) or not tree_id:
            raise InputError("tree")
        if isinstance(order, bool) or not isinstance(order, int):
            raise InputError("order")
        nodes[node_id] = {"parent_id": parent, "tree_id": tree_id, "order": order}
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    roots: list[str] = []
    for node_id, row in nodes.items():
        parent = row["parent_id"]
        seen: set[str] = set()
        cursor: object = node_id
        while cursor is not None:
            if not isinstance(cursor, str) or cursor in seen or cursor not in nodes:
                raise InputError("cycle")
            seen.add(cursor)
            cursor = nodes[cursor]["parent_id"]
        if parent is None:
            roots.append(node_id)
            continue
        if not isinstance(parent, str) or parent not in nodes:
            raise InputError("parent")
        if nodes[parent]["tree_id"] != row["tree_id"]:
            raise InputError("cross-tree")
        children[parent].append(node_id)
    roots.sort(key=lambda node_id: (nodes[node_id]["order"], node_id))
    for child_ids in children.values():
        child_ids.sort(key=lambda node_id: (nodes[node_id]["order"], node_id))
    return nodes, children, roots


def _walk(roots: list[str], children: dict[str, list[str]]) -> list[str]:
    """Preorder DFS. A repeated visit is a cycle, not a second copy of the node."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen:
            raise InputError("cycle")
        seen.add(node_id)
        order.append(node_id)
        for child in children[node_id]:
            visit(child)

    for root in roots:
        visit(root)
    return order
