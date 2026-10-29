"""Independent source-level and negative-fixture checks for the WP18 submitted run.

These tests belong to the reviewer branch, NOT the Agent's submitted branch.
Intentional failing cases diagnose remaining contract gaps, not CI setup errors.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from harness.code.build import args_signature, build

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "harness/tests/wp18/runs/20261029-wp18-01"
CASES = ("H01", "H02", "H03", "H04", "H05", "H06")
BASE_SHA = "8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f"


def load_case(case: str):
    path = RUN / case / "implementation.py"
    spec = importlib.util.spec_from_file_location("independent_wp18_" + case, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(case: str, filename: str):
    return json.loads((RUN / case / filename).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual_code_tree():
    # Actual AST extraction rather than the candidate's claimed Expected Diff.
    return build(include_harness=True)


def edge_key(edge):
    return edge["source"], edge["type"], edge["target"]


@pytest.mark.parametrize("case", CASES)
def test_candidate_is_anchored_to_real_ast_and_hash(case):
    source = RUN / case / "implementation.py"
    candidate = read(case, "review_candidate.json")
    assert candidate["base_sha"] == BASE_SHA
    assert candidate["project"] == "SAEBase"
    assert candidate["provisional"] is True
    assert candidate["case_id"] == case
    assert candidate["source_content_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    definitions = {
        node.name: node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module = ".".join(source.relative_to(ROOT).with_suffix("").parts)
    for change in candidate["changes"]:
        assert change["status"] == "unchecked"
        assert change["operation"] == "add"
        assert change["title"]["zh"] and change["title"]["en"]
        assert change["summary"]["zh"] and change["summary"]["en"]
        assert change["title"]["en"].rstrip(". ") != change["summary"]["en"].rstrip(". ")
        if change["view"] != "code":
            assert case == "H04" and change["view"] == "environment"
            assert change["provisional"] if "provisional" in change else True
            continue
        assert change["path"] == source.relative_to(ROOT).as_posix()
        assert change["parent_id"] == "code:module:" + module
        symbol = change["node_id"].split(":")[-1]
        assert symbol in definitions, (case, symbol)
        definition = definitions[symbol]
        assert change["line"] == definition.lineno, (case, symbol, definition.lineno)
        signature = (
            f"class {symbol}({', '.join(ast.unparse(x) for x in definition.bases)})"
            if isinstance(definition, ast.ClassDef)
            else args_signature(definition)
        )
        assert change["signature"] == signature, (case, symbol, signature)
        assert change["node_id"] == "code:" + (
            "class:" if isinstance(definition, ast.ClassDef) else "function:"
        ) + module + ":" + symbol


@pytest.mark.parametrize("case", CASES)
def test_expected_code_diff_matches_independently_extracted_ast(case, actual_code_tree):
    expected = read(case, "expected_code_diff.json")
    tree = actual_code_tree
    path_prefix = "harness/tests/wp18/runs/20261029-wp18-01/" + case + "/"
    observed_ids = {n["id"] for n in tree["nodes"] if n.get("path", "").startswith(path_prefix)}
    assert set(expected["added_ids"]) == observed_ids, (
        case, sorted(set(expected["added_ids"]) ^ observed_ids)
    )
    assert expected["removed_ids"] == []
    assert expected["changed"] == []
    assert expected["edges_removed"] == []
    observed_edges = {
        edge_key(e) for e in tree["edges"]
        if e["target"] in observed_ids and
        (e["source"] in observed_ids or e["source"] == "code:group:harness")
    }
    predicted_edges = {edge_key(e) for e in expected["edges_added"]}
    assert predicted_edges == observed_edges, (
        case, sorted(predicted_edges - observed_edges), sorted(observed_edges - predicted_edges)
    )


@pytest.mark.parametrize("case", CASES)
def test_contract_delivery_is_present_and_not_self_approved(case):
    directory = RUN / case
    required = (
        "implementation.py", "test_implementation.py",
        "review_candidate.json", "expected_code_diff.json",
        "selfcheck.json", "report.md"
    )
    for filename in required:
        assert (directory / filename).is_file(), (case, filename)
    check = read(case, "selfcheck.json")
    assert check["tests_passed"] > 0
    assert check["tests_failed"] == 0
    assert all("harness/tests/" in cmd for cmd in check["commands"])
    assert not any(x["status"] != "unchecked" for x in read(case, "review_candidate.json")["changes"])


def test_h01_rejects_boundary_traversal_and_does_not_scan(tmp_path):
    mod = load_case("H01")
    (tmp_path / "a.txt").write_bytes(b"a")
    assert mod.index_files(tmp_path, ["a.txt"])[0]["size"] == 1
    with pytest.raises(mod.EvidenceIndexError):
        mod.index_files(tmp_path, ["../secret.txt"])
    with pytest.raises(mod.EvidenceIndexError):
        mod.index_files(tmp_path, ["a.txt", "a.txt"])
    assert mod.index_files(tmp_path, []) == []


def test_h02_diff_ignores_order_and_detects_edges():
    mod = load_case("H02")
    a = {"nodes": [{"id": "a", "signature": "f()"}, {"id": "b"}],
         "edges": [{"source": "a", "type": "contains", "target": "b"}]}
    b = {"nodes": [{"id": "b"}, {"id": "a", "signature": "f()"}],
         "edges": [{"source": "a", "type": "contains", "target": "b"}]}
    assert not any(mod.diff_trees(a, b)[k] for k in (
        "added_ids", "removed_ids", "signature_changed", "edges_added", "edges_removed"))
    c = {"nodes": b["nodes"], "edges": []}
    assert len(mod.diff_trees(a, c)["edges_removed"]) == 1


def test_h03_rejects_traversal_as_source_evidence():
    mod = load_case("H03")
    bundle = {
        "synthetic": True,
        "nodes": [{"id": "n", "term_ids": ["term.x"], "source_evidence": "../secret.py:1"}],
        "glossary": [{"id": "term.x", "related_node_ids": ["n"],
                      "source_evidence": "../secret.py:1"}],
    }
    report = mod.check_context_links(bundle)
    assert any(d["code"] == "missing_source_evidence" for d in report["diagnostics"]), (
        "H03 accepted path traversal as verifiable source evidence", report
    )


def test_h04_redacts_secrets_nested_in_synthetic_records():
    mod = load_case("H04")
    data = {"synthetic": True,
            "device": {"token": "SECRET-NESTED-123", "home": "/private/home"},
            "gpu": None}
    output = mod.normalize_environment(data)
    serialized = json.dumps(output, ensure_ascii=False)
    assert "SECRET-NESTED-123" not in serialized, "H04 leaked a nested token"
    assert "/private/home" not in serialized, "H04 leaked a nested private path"
    assert output["gpu"] == "unknown"
    assert output["synthetic"] is True and output["observed"] is False


def test_h05_parent_scope_rejects_unknown_node():
    mod = load_case("H05")
    graph = {"synthetic": True,
             "nodes": [{"id": "root", "parent_id": None, "order": 0,
                        "tree_id": "code"}]}
    with pytest.raises(mod.ReviewBoundaryError):
        mod.build_queue(graph, {"kind": "parent", "node_id": "does-not-exist"})


def test_h06_reports_missing_chinese_content_as_bilingual_gap():
    mod = load_case("H06")
    record = {
        "synthetic": True, "project": "SAEBase",
        "id": "node.test", "signature": "def some_fn():", "path": "fixture.py",
        "title": {"en": "Calculate trace"},
        "summary": {"en": "Derive a trace from a provided fixed input."},
        "description": {"en": "Uses local data and returns a stable summary."},
    }
    out = mod.audit_overlay([record])
    assert any(
        d["code"] in ("missing_chinese", "missing_localization")
        and d["field"] in ("title", "summary", "description")
        for d in out["detected"]
    ), ("H06 accepted a non-bilingual overlay with no Chinese fields", out)
