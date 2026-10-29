"""在测验图上排 DFS 审阅队列。草稿不是人类批准，也没有写入端点。"""

from __future__ import annotations

import copy


class ReviewBoundaryError(RuntimeError):
    """试图把草稿写成正式批准。"""


def build_queue(graph: dict, scope: dict) -> list[str]:
    """按兄弟节点的 order 做深度优先，不按名称或 Work Plan 编号排序。"""
    nodes = _validate_graph(graph)
    kind = scope.get("kind") if isinstance(scope, dict) else None
    if kind == "all":
        return _forest(nodes)
    if kind == "tree":
        selected = [node for node in nodes if node.get("tree_id") == scope.get("tree_id")]
        return _forest(selected)
    if kind == "parent":
        return _dfs(nodes, str(scope["node_id"]))
    if kind == "node":
        node_id = str(scope["node_id"])
        if node_id not in {node["id"] for node in nodes}:
            raise ReviewBoundaryError(f"unknown node: {node_id}")
        return [node_id]
    raise ReviewBoundaryError("scope kind must be all, tree, parent, or node")


def empty_state(node_ids: list[str]) -> dict:
    """每个节点单独从 unchecked 草稿开始，正式批准标志保持关闭。"""
    return {
        "drafts": {node_id: {"draft": "unchecked", "formal_approval": False} for node_id in node_ids},
        "formal_approval": False,
    }


def draft_decision(state: dict, node_id: str, draft: str) -> dict:
    """记录本机草稿。不改变父节点，也不会变成正式 Approve。"""
    if draft not in {"unchecked", "approve", "deny"}:
        raise ReviewBoundaryError("unknown draft")
    if node_id not in state["drafts"]:
        raise ReviewBoundaryError(f"unknown node: {node_id}")
    updated = copy.deepcopy(state)
    updated["drafts"][node_id] = {"draft": draft, "formal_approval": False}
    updated["formal_approval"] = False
    return updated


def undo_draft(state: dict, node_id: str) -> dict:
    """把该节点草稿退回 unchecked，不把它改成通过。"""
    return draft_decision(state, node_id, "unchecked")


def next_unchecked(queue: list[str], state: dict, cursor: int) -> int | None:
    """从当前位置之后找下一个仍为 unchecked 的队列下标。"""
    for index in range(cursor + 1, len(queue)):
        if state["drafts"][queue[index]]["draft"] == "unchecked":
            return index
    return None


def previous_unchecked(queue: list[str], state: dict, cursor: int) -> int | None:
    """从当前位置之前找上一个仍为 unchecked 的队列下标。"""
    for index in range(cursor - 1, -1, -1):
        if state["drafts"][queue[index]]["draft"] == "unchecked":
            return index
    return None


def serialize_drafts(state: dict) -> dict:
    """导出所见草稿。结果明确不是正式批准。"""
    return {
        "formal_approval": False,
        "human_approval_endpoint": None,
        "drafts": copy.deepcopy(state["drafts"]),
    }


def reject_formal_approval() -> None:
    """这里没有可代替人类 Approve 的写入接口。"""
    raise ReviewBoundaryError("formal Approve is reserved for a human reviewer")


def _validate_graph(graph: dict) -> list[dict]:
    """拒绝重复 ID、悬空父节点和图中的环。"""
    if not isinstance(graph, dict) or graph.get("synthetic") is not True:
        raise ReviewBoundaryError("graph must be marked synthetic")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ReviewBoundaryError("nodes must be a list")
    seen: set[str] = set()
    cleaned = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise ReviewBoundaryError("node id must be a nonempty string")
        if node["id"] in seen:
            raise ReviewBoundaryError(f"duplicate id: {node['id']}")
        seen.add(node["id"])
        cleaned.append(node)
    for node in cleaned:
        parent = node.get("parent_id")
        if parent is not None and parent not in seen:
            raise ReviewBoundaryError(f"unknown parent: {parent}")
        cursor = node["id"]
        guard = 0
        while cursor is not None:
            guard += 1
            if guard > len(cleaned):
                raise ReviewBoundaryError("cycle in parent links")
            match = next(item for item in cleaned if item["id"] == cursor)
            cursor = match.get("parent_id")
    return cleaned


def _forest(nodes: list[dict]) -> list[str]:
    """从所选节点里的根开始做深度优先。"""
    selected = {node["id"] for node in nodes}
    roots = [node for node in nodes if node.get("parent_id") not in selected]
    roots.sort(key=lambda node: (node.get("order", 0), node["id"]))
    queue: list[str] = []
    for root in roots:
        queue.extend(_dfs(nodes, root["id"]))
    return queue


def _dfs(nodes: list[dict], node_id: str) -> list[str]:
    """先访问节点，再按 order 访问子节点。"""
    children = [node for node in nodes if node.get("parent_id") == node_id]
    children.sort(key=lambda node: (node.get("order", 0), node["id"]))
    order = [node_id]
    for child in children:
        order.extend(_dfs(nodes, child["id"]))
    return order
