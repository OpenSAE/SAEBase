"""在测验图上排 DFS 审阅队列。草稿带修订号，不能写成正式批准。"""

from __future__ import annotations

import copy


class ReviewConflict(RuntimeError):
    """图不合法，或草稿修订与正式批准边界被打破。"""


def queue_nodes(graph: dict, scope: dict) -> list[str]:
    """按兄弟 order 做深度优先，不按名称或 Work Plan 编号排序。"""
    nodes = _check_graph(graph)
    if not isinstance(scope, dict):
        raise ReviewConflict("scope must be an object")
    kind = scope.get("kind")
    if kind == "all":
        return _roots(nodes)
    if kind == "tree":
        selected = [node for node in nodes if node.get("tree_id") == scope.get("tree_id")]
        return _roots(selected)
    if kind == "parent":
        node_id = scope.get("node_id")
        if not isinstance(node_id, str) or node_id not in {node["id"] for node in nodes}:
            raise ReviewConflict("unknown node")
        return _walk_down(nodes, node_id)
    if kind == "node":
        node_id = scope.get("node_id")
        if not isinstance(node_id, str) or node_id not in {node["id"] for node in nodes}:
            raise ReviewConflict("unknown node")
        return [node_id]
    raise ReviewConflict("scope kind must be all, tree, parent, or node")


def begin_drafts(node_ids: list[str]) -> dict:
    """每个节点从 unchecked 草稿开始，修订号为 0。"""
    if not isinstance(node_ids, list) or not all(isinstance(item, str) and item for item in node_ids):
        raise ReviewConflict("node ids must be nonempty strings")
    return {
        "revision": 0,
        "formal_approval": False,
        "drafts": {node_id: {"draft": "unchecked", "formal_approval": False} for node_id in node_ids},
    }


def write_draft(state: dict, node_id: str, draft: str, *, revision: int) -> dict:
    """记录草稿。修订号不符时失败，并且不改原状态。"""
    if draft not in {"unchecked", "approve", "deny"}:
        raise ReviewConflict("unknown draft")
    if revision != state.get("revision"):
        raise ReviewConflict("stale revision")
    if node_id not in state.get("drafts", {}):
        raise ReviewConflict("unknown node")
    updated = copy.deepcopy(state)
    updated["drafts"][node_id] = {"draft": draft, "formal_approval": False}
    updated["formal_approval"] = False
    updated["revision"] = revision + 1
    return updated


def undo_draft(state: dict, node_id: str, *, revision: int) -> dict:
    """把该节点退回 unchecked，不把它改成通过。"""
    return write_draft(state, node_id, "unchecked", revision=revision)


def neighbor_unchecked(queue: list[str], state: dict, cursor: int, step: int) -> int | None:
    """沿当前队列寻找仍为 unchecked 的下标。step 只能是 1 或 -1。"""
    if step not in (1, -1):
        raise ReviewConflict("step must be 1 or -1")
    index = cursor + step
    while 0 <= index < len(queue):
        if state["drafts"][queue[index]]["draft"] == "unchecked":
            return index
        index += step
    return None


def export_drafts(state: dict) -> dict:
    """导出草稿副本，并写明没有人类批准端点。"""
    return {
        "formal_approval": False,
        "human_approval_endpoint": None,
        "revision": state["revision"],
        "drafts": copy.deepcopy(state["drafts"]),
    }


def refuse_authority_write() -> None:
    """这里没有可代替人类 Approve 的写入接口。"""
    raise ReviewConflict("formal Approve is reserved for a human reviewer")


def _check_graph(graph: dict) -> list[dict]:
    """拒绝重复 ID、悬空父节点、跨树父边和图中的环。"""
    if not isinstance(graph, dict) or graph.get("synthetic") is not True:
        raise ReviewConflict("graph must be marked synthetic")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ReviewConflict("nodes must be a list")
    seen: dict[str, dict] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise ReviewConflict("node id must be a nonempty string")
        if node["id"] in seen:
            raise ReviewConflict(f"duplicate id: {node['id']}")
        seen[node["id"]] = node
    for node in seen.values():
        parent = node.get("parent_id")
        if parent is None:
            continue
        if parent not in seen:
            raise ReviewConflict(f"unknown parent: {parent}")
        if seen[parent].get("tree_id") != node.get("tree_id"):
            raise ReviewConflict(f"cross-tree parent: {node['id']}")
        cursor = node["id"]
        guard = 0
        while cursor is not None:
            guard += 1
            if guard > len(seen):
                raise ReviewConflict("cycle in parent links")
            cursor = seen[cursor].get("parent_id")
    return list(seen.values())


def _roots(nodes: list[dict]) -> list[str]:
    """从所选集合中的根开始，按 order 展开。"""
    selected = {node["id"] for node in nodes}
    roots = [node for node in nodes if node.get("parent_id") not in selected]
    roots.sort(key=lambda node: (node.get("order", 0), node["id"]))
    queue: list[str] = []
    for root in roots:
        queue.extend(_walk_down(nodes, root["id"]))
    return queue


def _walk_down(nodes: list[dict], node_id: str) -> list[str]:
    """先放入当前节点，再按 order 放入子节点。"""
    children = [node for node in nodes if node.get("parent_id") == node_id]
    children.sort(key=lambda node: (node.get("order", 0), node["id"]))
    order = [node_id]
    for child in children:
        order.extend(_walk_down(nodes, child["id"]))
    return order
