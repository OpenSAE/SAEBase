"""Self-test for WP19 H34. Frozen checks are read from the official catalog."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CASE = HERE.name
REPO = HERE.parents[5]
IMPL = HERE / "implementation.py"
SPEC = REPO / "harness/contracts/wp19_cases.json"


def _load():
    spec = importlib.util.spec_from_file_location("wp19_" + CASE, IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen():
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    case = next(item for item in document["cases"] if item["id"] == CASE)
    assert case["status"] == "READY"
    return case["checks"]


@pytest.mark.parametrize("row", _frozen(), ids=lambda row: row["name"])
def test_frozen_check(row):
    module = _load()
    raw = copy.deepcopy(row["input"])
    saved = copy.deepcopy(raw)
    if row.get("reject") is True:
        with pytest.raises(ValueError):
            module.solve(raw)
    else:
        assert module.solve(raw) == row["expected"]
    assert raw == saved


def test_extra_fixture_keeps_input():
    module = _load()
    extra = json.loads((HERE / "fixtures" / "extra.json").read_text(encoding="utf-8"))
    for row in extra["positive"]:
        raw = copy.deepcopy(row["input"])
        saved = copy.deepcopy(raw)
        assert module.solve(raw) == row["expected"]
        assert raw == saved
    for row in extra["negative"]:
        raw = copy.deepcopy(row["input"])
        saved = copy.deepcopy(raw)
        with pytest.raises(ValueError):
            module.solve(raw)
        assert raw == saved


def test_does_not_import_reference_oracle():
    tree = ast.parse(IMPL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("harness.contracts.wp19_reference")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("harness.contracts.wp19_reference")


def test_candidate_matches_implementation_ast():
    module_name = "harness.tests.wp19.runs.20261029-wp19-01." + CASE + ".implementation"
    tree = ast.parse(IMPL.read_text(encoding="utf-8"))
    candidate = json.loads((HERE / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((HERE / "expected_code_diff.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(IMPL.read_bytes()).hexdigest()
    assert candidate["schema"] == "saebase.wp19.review_candidate.v1"
    assert candidate["provisional"] is True
    assert candidate["project"] == "SAEBase"
    assert candidate["case_id"] == CASE
    assert candidate["source_content_sha256"] == digest
    assert expected["removed_ids"] == []
    by_id = {item["node_id"]: item for item in candidate["changes"]}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.ClassDef):
            node_id = f"code:class:{module_name}:{node.name}"
            signature = (
                "class "
                + node.name
                + "("
                + ", ".join(ast.unparse(item) for item in node.bases)
                + ")"
            )
        else:
            node_id = f"code:function:{module_name}:{node.name}"
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"def {node.name}({ast.unparse(node.args)}){returns}"
        item = by_id[node_id]
        assert item["line"] == node.lineno
        assert item["signature"] == signature
        assert item["status"] == "unchecked"
        assert item["title"]["zh"] != item["title"]["en"]
        assert item["summary"]["zh"] != item["summary"]["en"]
        assert node_id in expected["added_ids"]
