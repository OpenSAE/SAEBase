"""H05 自测：DFS 顺序、修订冲突、父子独立，以及不能正式批准。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h05", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _graph() -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / "yard.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_order_follows_sibling_order() -> None:
    module = _load()
    graph = _graph()
    assert module.queue_nodes(graph, {"kind": "all"}) == ["pile", "kite", "nest", "ember", "slip-b", "slip-a"]
    assert module.queue_nodes(graph, {"kind": "tree", "tree_id": "yard"}) == ["pile", "kite", "nest", "ember"]
    assert module.queue_nodes(graph, {"kind": "parent", "node_id": "kite"}) == ["kite", "nest"]
    assert module.queue_nodes(graph, {"kind": "node", "node_id": "ember"}) == ["ember"]


def test_unknown_node_is_not_invented() -> None:
    module = _load()
    try:
        result = module.queue_nodes(_graph(), {"kind": "node", "node_id": "missing-post"})
    except module.ReviewConflict:
        return
    raise AssertionError(result)


def test_drafts_are_local_and_stale_revision_conflicts() -> None:
    module = _load()
    graph = _graph()
    state = module.begin_drafts([node["id"] for node in graph["nodes"]])
    denied = module.write_draft(state, "nest", "deny", revision=0)
    assert state["drafts"]["nest"]["draft"] == "unchecked"
    assert denied["drafts"]["kite"]["draft"] == "unchecked"
    assert denied["drafts"]["pile"]["draft"] == "unchecked"
    assert denied["revision"] == 1
    try:
        module.write_draft(denied, "nest", "approve", revision=0)
    except module.ReviewConflict:
        pass
    else:
        raise AssertionError("stale")
    assert denied["drafts"]["nest"]["draft"] == "deny"
    narrowed = module.queue_nodes(graph, {"kind": "node", "node_id": "ember"})
    assert narrowed == ["ember"]
    assert denied["drafts"]["nest"]["draft"] == "deny"
    undone = module.undo_draft(denied, "nest", revision=1)
    assert undone["drafts"]["nest"] == {"draft": "unchecked", "formal_approval": False}
    queue = module.queue_nodes(graph, {"kind": "all"})
    assert module.neighbor_unchecked(queue, undone, 0, 1) == 1
    assert module.export_drafts(undone)["formal_approval"] is False
    assert module.export_drafts(undone)["human_approval_endpoint"] is None


def test_cross_tree_and_authority_are_refused() -> None:
    module = _load()
    crossed = {
        "synthetic": True,
        "nodes": [
            {"id": "pile", "parent_id": None, "order": 0, "tree_id": "yard"},
            {"id": "ember", "parent_id": "pile", "order": 0, "tree_id": "lane"},
        ],
    }
    try:
        module.queue_nodes(crossed, {"kind": "all"})
    except module.ReviewConflict:
        pass
    else:
        raise AssertionError("cross")
    try:
        module.refuse_authority_write()
    except module.ReviewConflict:
        pass
    else:
        raise AssertionError("approve")
    source = Path(__file__).with_name("implementation.py").read_text(encoding="utf-8")
    assert "agent_flow" not in source
    assert '"formal_approval": True' not in source


def test_candidate_matches_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert payload["provisional"] is True
    by_name = {
        change["signature"].split("(")[0].split()[-1]: change
        for change in payload["changes"]
        if change["path"].endswith("implementation.py") and change["signature"]
    }
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert by_name[node.name]["line"] == node.lineno
            assert by_name[node.name]["status"] == "unchecked"
            assert any(node.name in item for item in expected["added_ids"])
