"""比较两份调用方传入的测验树，不打开仓库中的正式代码树快照。"""

from __future__ import annotations


class TreeDiffError(ValueError):
    """测验树的结构不合法。"""


_RELATIONS = frozenset({"contains", "calls", "imports", "inherits"})


def diff_trees(before: dict, after: dict) -> dict:
    """按稳定 node_id 报告新增、删除、签名变化和边变化。

    只比较调用方传入的两份对象。节点或边的排列顺序不会产生差异。
    """
    left_sig, left_edges = _validate(before)
    right_sig, right_edges = _validate(after)
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
        "edges_added": _pack(right_edges.difference(left_edges)),
        "edges_removed": _pack(left_edges.difference(right_edges)),
    }


def _validate(tree: dict) -> tuple[dict[str, str], set[tuple[str, str, str]]]:
    """拒绝重复 ID、未知端点和规定关系以外的边。"""
    if not isinstance(tree, dict):
        raise TreeDiffError("tree must be an object")
    nodes = tree.get("nodes")
    edges = tree.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TreeDiffError("tree must contain nodes and edges lists")
    signatures: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise TreeDiffError("node id must be a nonempty string")
        if node["id"] in signatures:
            raise TreeDiffError(f"duplicate id: {node['id']}")
        signatures[node["id"]] = "" if node.get("signature") is None else str(node.get("signature"))
    triples: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise TreeDiffError("edge must be an object")
        source = edge.get("source")
        target = edge.get("target")
        kind = edge.get("type")
        if not all(isinstance(item, str) and item for item in (source, target, kind)):
            raise TreeDiffError("edge endpoints and type must be nonempty strings")
        if kind not in _RELATIONS:
            raise TreeDiffError(f"illegal relation: {kind}")
        if source not in signatures or target not in signatures:
            raise TreeDiffError(f"unknown edge endpoint: {source} -> {target}")
        triple = (str(source), str(kind), str(target))
        if triple in triples:
            raise TreeDiffError(f"duplicate edge: {source} -{kind}-> {target}")
        triples.add(triple)
    return signatures, triples


def _pack(triples: set[tuple[str, str, str]]) -> list[dict[str, str]]:
    """把边整理成稳定顺序的记录。"""
    return [{"source": source, "type": kind, "target": target} for source, kind, target in sorted(triples)]
