"""比较两份测验用代码树。未知端点不会被补成新节点。"""

from __future__ import annotations


class TreeShapeError(ValueError):
    """树的身份或关系不合法，比较没有开始。"""


_RELATIONS = frozenset({"contains", "calls", "imports", "inherits"})


def compare_code_trees(before: dict, after: dict) -> dict:
    """按稳定 node_id 报告新增、删除、签名变化和边变化。

    只使用调用方传入的对象。排列顺序不会产生差异。
    """
    left_sig, left_edges = _accept_tree(before)
    right_sig, right_edges = _accept_tree(after)
    shared = set(left_sig).intersection(right_sig)
    changed = [
        {"node_id": node_id, "before": left_sig[node_id], "after": right_sig[node_id]}
        for node_id in sorted(shared)
        if left_sig[node_id] != right_sig[node_id]
    ]
    return {
        "added_ids": sorted(set(right_sig).difference(left_sig)),
        "removed_ids": sorted(set(left_sig).difference(right_sig)),
        "signature_changed": changed,
        "edges_added": _edge_rows(right_edges.difference(left_edges)),
        "edges_removed": _edge_rows(left_edges.difference(right_edges)),
    }


def _accept_tree(tree: dict) -> tuple[dict[str, str], set[tuple[str, str, str]]]:
    """拒绝重复 ID、跨项目节点、未知端点、非法关系和 contains 环。"""
    if not isinstance(tree, dict) or tree.get("synthetic") is not True:
        raise TreeShapeError("tree must be an object marked synthetic")
    nodes = tree.get("nodes")
    edges = tree.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TreeShapeError("nodes and edges must be lists")
    signatures: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise TreeShapeError("node id must be a nonempty string")
        project = node.get("project", "SAEBase")
        if project != "SAEBase":
            raise TreeShapeError(f"cross-tree node rejected: {node['id']}")
        if node["id"] in signatures:
            raise TreeShapeError(f"duplicate id: {node['id']}")
        signature = node.get("signature", "")
        if not isinstance(signature, str):
            raise TreeShapeError("signature must be a string")
        signatures[node["id"]] = signature
    triples: set[tuple[str, str, str]] = set()
    contains: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            raise TreeShapeError("edge must be an object")
        source, target, kind = edge.get("source"), edge.get("target"), edge.get("type")
        if not all(isinstance(item, str) and item for item in (source, target, kind)):
            raise TreeShapeError("edge endpoints and type must be nonempty strings")
        if kind not in _RELATIONS:
            raise TreeShapeError(f"illegal relation: {kind}")
        if source not in signatures or target not in signatures:
            raise TreeShapeError(f"unknown edge endpoint: {source} -> {target}")
        triple = (str(source), str(kind), str(target))
        if triple in triples:
            raise TreeShapeError(f"duplicate edge: {source} -{kind}-> {target}")
        triples.add(triple)
        if kind == "contains":
            contains.setdefault(str(source), []).append(str(target))
    if _contains_cycle(contains):
        raise TreeShapeError("contains cycle rejected")
    return signatures, triples


def _contains_cycle(children: dict[str, list[str]]) -> bool:
    """沿 contains 边查找环，不把缺失端点补进图里。"""
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for child in children.get(node_id, []):
            if walk(child):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(walk(node_id) for node_id in sorted(children))


def _edge_rows(triples: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    """把边排成稳定的 source、type、target 记录。"""
    return [
        {"source": source, "type": kind, "target": target}
        for source, kind, target in sorted(triples)
    ]
