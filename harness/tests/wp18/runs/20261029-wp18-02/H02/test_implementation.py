"""H02 自测：重排无差异，签名和丢边精确，坏图不会造出节点。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h02", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture(name: str) -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_reordering_reports_nothing() -> None:
    module = _load()
    diff = module.compare_code_trees(_fixture("left.json"), _fixture("left-reordered.json"))
    assert diff == {
        "added_ids": [],
        "removed_ids": [],
        "signature_changed": [],
        "edges_added": [],
        "edges_removed": [],
    }
    assert diff == module.compare_code_trees(_fixture("left.json"), _fixture("left-reordered.json"))


def test_signature_and_structure_are_exact() -> None:
    module = _load()
    signature = module.compare_code_trees(_fixture("left.json"), _fixture("signature-shift.json"))
    assert signature["signature_changed"] == [
        {
            "node_id": "code:function:exam.packet:south_span",
            "before": "def south_span()",
            "after": "def south_span(flag: bool)",
        }
    ]
    assert signature["added_ids"] == []
    assert signature["edges_added"] == []
    structure = module.compare_code_trees(_fixture("left.json"), _fixture("edge-shift.json"))
    assert structure["added_ids"] == ["code:function:exam.packet:west_span"]
    assert structure["edges_removed"] == [
        {
            "source": "code:module:exam.packet",
            "type": "contains",
            "target": "code:function:exam.packet:south_span",
        }
    ]
    assert structure["edges_added"] == [
        {
            "source": "code:module:exam.packet",
            "type": "contains",
            "target": "code:function:exam.packet:west_span",
        }
    ]


def test_bad_graphs_raise_without_inventing_nodes() -> None:
    module = _load()
    for name in ("duplicate-id.json", "missing-end.json", "odd-relation.json", "cycle.json", "other-project.json"):
        try:
            result = module.compare_code_trees(_fixture("left.json"), _fixture(name))
        except module.TreeShapeError:
            continue
        raise AssertionError((name, result))


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
