"""H05 自测：DFS 顺序、筛选不丢草稿，以及不能正式批准。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h05_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _graph() -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / "graph.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_dfs_uses_sibling_order_not_names() -> None:
    module = _load()
    graph = _graph()
    assert module.build_queue(graph, {"kind": "all"}) == [
        "root",
        "z-child",
        "z-grand",
        "a-child",
        "wp-02",
        "wp-01",
    ]
    assert module.build_queue(graph, {"kind": "tree", "tree_id": "main"}) == [
        "root",
        "z-child",
        "z-grand",
        "a-child",
    ]
    assert module.build_queue(graph, {"kind": "parent", "node_id": "z-child"}) == ["z-child", "z-grand"]
    assert module.build_queue(graph, {"kind": "node", "node_id": "a-child"}) == ["a-child"]


def test_filter_keeps_drafts_and_parent_is_independent() -> None:
    module = _load()
    graph = _graph()
    ids = [node["id"] for node in graph["nodes"]]
    state = module.empty_state(ids)
    denied = module.draft_decision(state, "z-grand", "deny")
    assert state["drafts"]["z-grand"]["draft"] == "unchecked"
    assert denied["drafts"]["z-child"]["draft"] == "unchecked"
    assert denied["drafts"]["root"]["draft"] == "unchecked"
    queued = module.build_queue(graph, {"kind": "node", "node_id": "a-child"})
    assert queued == ["a-child"]
    assert denied["drafts"]["z-grand"]["draft"] == "deny"
    assert module.serialize_drafts(denied)["formal_approval"] is False


def test_undo_returns_to_unchecked_not_approve() -> None:
    module = _load()
    graph = _graph()
    state = module.empty_state([node["id"] for node in graph["nodes"]])
    state = module.draft_decision(state, "a-child", "deny")
    state = module.undo_draft(state, "a-child")
    assert state["drafts"]["a-child"] == {"draft": "unchecked", "formal_approval": False}
    queue = module.build_queue(graph, {"kind": "all"})
    assert module.next_unchecked(queue, state, 0) == 1
    assert module.previous_unchecked(queue, state, 0) is None


def test_formal_approval_is_refused() -> None:
    module = _load()
    try:
        module.reject_formal_approval()
    except module.ReviewBoundaryError:
        pass
    else:
        raise AssertionError("approve")
    source = Path(__file__).with_name("implementation.py").read_text(encoding="utf-8")
    assert "agent_flow" not in source
    assert "urllib" not in source
    assert "formal_approval\"]: True" not in source
    assert '"formal_approval": True' not in source


def test_candidate_binds_real_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert payload["provisional"] is True
    tree = ast.parse(source)
    by_name = {}
    for change in payload["changes"]:
        if change["path"].endswith("implementation.py") and change["signature"]:
            by_name[change["signature"].split("(")[0].split()[-1]] = change
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert by_name[node.name]["line"] == node.lineno
            assert by_name[node.name]["status"] == "unchecked"
            assert any(node.name in item for item in expected["added_ids"])
