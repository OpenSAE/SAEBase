"""H02 自测：重排不误报，签名、丢边和非法树要精确定位。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h02_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture(name: str) -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_reorder_is_silent() -> None:
    module = _load()
    before = _fixture("before.json")
    after = _fixture("reordered.json")
    assert module.diff_trees(before, after) == {
        "added_ids": [],
        "removed_ids": [],
        "signature_changed": [],
        "edges_added": [],
        "edges_removed": [],
    }
    assert module.diff_trees(before, after) == module.diff_trees(before, after)


def test_signature_change_is_only_that_node() -> None:
    module = _load()
    diff = module.diff_trees(_fixture("before.json"), _fixture("signature.json"))
    assert diff["added_ids"] == []
    assert diff["removed_ids"] == []
    assert diff["signature_changed"] == [
        {"node_id": "code:function:demo:beta", "before": "def beta()", "after": "def beta(flag: bool)"}
    ]
    assert diff["edges_added"] == []
    assert diff["edges_removed"] == []


def test_lost_edge_and_new_node() -> None:
    module = _load()
    diff = module.diff_trees(_fixture("before.json"), _fixture("structure.json"))
    assert diff["added_ids"] == ["code:function:demo:gamma"]
    assert diff["removed_ids"] == []
    assert diff["edges_removed"] == [
        {"source": "code:module:demo", "type": "contains", "target": "code:function:demo:beta"}
    ]
    assert diff["edges_added"] == [
        {"source": "code:module:demo", "type": "contains", "target": "code:function:demo:gamma"}
    ]


def test_reject_duplicate_unknown_and_illegal() -> None:
    module = _load()
    for name in ("duplicate_id.json", "unknown_endpoint.json", "illegal_relation.json"):
        try:
            module.diff_trees(_fixture("before.json"), _fixture(name))
        except module.TreeDiffError:
            pass
        else:
            raise AssertionError(name)


def test_does_not_name_authority_tree() -> None:
    source = Path(__file__).with_name("implementation.py").read_text(encoding="utf-8")
    assert "doc/code/code_tree.json" not in source
    assert "open(" not in source


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
