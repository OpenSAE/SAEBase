"""Independent WP18 round-two regression tests. Do not merge into main.

Tests reflect preexisting Contract obligations and submitted behavioral claims.
The original task-agent branch remains unchanged.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "harness/tests/wp18/runs/20261029-wp18-02"


def load(case: str):
    path = RUN / case / "implementation.py"
    spec = importlib.util.spec_from_file_location("independent_wp18_" + case, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(case: str, name: str):
    return json.loads((RUN / case / "fixtures" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", ["H01", "H02", "H03", "H04", "H05", "H06"])
def test_real_candidate_provenance_against_ast(case: str):
    src = RUN / case / "implementation.py"
    candidate = json.loads((RUN / case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((RUN / case / "expected_code_diff.json").read_text(encoding="utf-8"))
    contents = src.read_bytes()
    assert candidate["provisional"] is True
    assert candidate["source_content_sha256"] == hashlib.sha256(contents).hexdigest()
    rows = {
        (row["signature"].split("(")[0].split()[-1]): row
        for row in candidate["changes"]
        if row["path"] == src.relative_to(ROOT).as_posix()
        and row.get("signature", "").startswith(("def ", "class "))
    }
    for node in ast.parse(contents).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name in rows
            assert rows[node.name]["line"] == node.lineno
            assert rows[node.name]["status"] == "unchecked"
            assert any(node.name in item for item in expected["added_ids"])


def test_h01_independent_named_file_only(tmp_path: Path):
    module = load("H01")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "unselected.txt").write_text("PRIVATE", encoding="utf-8")
    result = module.index_named_files(tmp_path, ["b.txt", "a.txt"])
    assert [row["path"] for row in result] == ["a.txt", "b.txt"]
    assert "PRIVATE" not in json.dumps(result)


def test_h02_independent_reorder_no_diff():
    module = load("H02")
    rows = module.compare_code_trees(fixture("H02", "left.json"), fixture("H02", "left-reordered.json"))
    assert not any(rows.values())


def test_h03_traversal_is_rejected():
    module = load("H03")
    data = fixture("H03", "linked.json")
    data["nodes"][0]["source_evidence"] = "../secret-independent.md:1"
    result = module.audit_term_links(data, RUN / "H03/fixtures")
    assert any(item["code"] == "rejected_reference" for item in result["diagnostics"])
    assert result["provenance"][0]["status"] == "rejected"


def test_h03_duplicate_must_not_override_first_node():
    module = load("H03")
    data = fixture("H03", "linked.json")
    second = copy.deepcopy(data["nodes"][0])
    second["term_ids"] = []
    data["nodes"].append(second)
    result = module.audit_term_links(data, RUN / "H03/fixtures")
    assert any(item["code"] == "duplicate_node_id" for item in result["diagnostics"])
    # Candidate semantic claim: duplicates must not overwrite the first row.
    assert not any(item["code"] == "reverse_mismatch" for item in result["diagnostics"])


def test_h04_existing_nested_secret_fixture():
    module = load("H04")
    result = module.normalize_environment(fixture("H04", "device.json"))
    blob = json.dumps(result)
    assert "NESTED-SECRET-41" not in blob
    assert "SURFACE-TOKEN-41" not in blob
    assert result["observed"] is False


def test_h04_sensitive_key_variants_do_not_leak():
    module = load("H04")
    record = {
        "synthetic": True,
        "device": {"private_key": "INDEPENDENT-PRIVATE-KEY", "cookie": "INDEPENDENT-COOKIE"},
    }
    output = json.dumps(module.normalize_environment(record))
    assert "INDEPENDENT-PRIVATE-KEY" not in output
    assert "INDEPENDENT-COOKIE" not in output


def test_h05_unknown_parent_rejected():
    module = load("H05")
    with pytest.raises(module.ReviewConflict):
        module.queue_nodes(fixture("H05", "yard.json"), {"kind": "parent", "node_id": "absent"})


def test_h05_unknown_tree_rejected():
    module = load("H05")
    with pytest.raises(module.ReviewConflict):
        module.queue_nodes(fixture("H05", "yard.json"), {"kind": "tree", "tree_id": "absent"})


def test_h05_duplicate_draft_ids_rejected():
    module = load("H05")
    with pytest.raises(module.ReviewConflict):
        module.begin_drafts(["same", "same"])


def test_h06_missing_chinese_detected():
    module = load("H06")
    rows = fixture("H06", "rows.json")
    output = module.review_locale_rows(rows)
    assert any(row["code"] == "missing_zh" for row in output["detected"])


def test_h06_empty_identity_fields_rejected():
    module = load("H06")
    row = fixture("H06", "rows.json")[0]
    row["id"] = ""
    row["signature"] = ""
    row["path"] = ""
    with pytest.raises(module.LocaleRowError):
        module.review_locale_rows([row])
