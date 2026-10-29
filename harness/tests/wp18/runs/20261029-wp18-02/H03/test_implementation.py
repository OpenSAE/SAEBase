"""H03 自测：真实文件来源可通过，断链和未解析路径不会被写成已核实。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h03", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _root() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def _bundle(name: str) -> dict:
    return json.loads((_root() / name).read_text(encoding="utf-8"))


def test_resolved_bidirectional_links_are_clean() -> None:
    module = _load()
    report = module.audit_term_links(_bundle("linked.json"), _root())
    assert report["diagnostics"] == []
    assert report["suggestions"] == []
    assert report["glossary_written"] is False
    assert report["implemented_nodes"] == []
    assert report["provenance"] == [
        {"owner": "exam.node.span", "reference": "sources/north-note.md:2", "status": "resolved"},
        {"owner": "exam.term.span", "reference": "sources/term-note.md:1", "status": "resolved"},
    ]


def test_broken_links_stay_specific() -> None:
    module = _load()
    codes = {item["code"] for item in module.audit_term_links(_bundle("broken.json"), _root())["diagnostics"]}
    assert codes == {
        "rejected_reference",
        "missing_term",
        "provenance_unresolved",
        "reverse_mismatch",
        "missing_node",
    }


def test_shared_alias_is_not_promoted() -> None:
    module = _load()
    report = module.audit_term_links(_bundle("shared-alias.json"), _root())
    assert any(item["code"] == "duplicate_alias" for item in report["diagnostics"])
    assert all(item["applied"] == "false" for item in report["suggestions"])
    assert report["glossary_written"] is False
    assert report["implemented_nodes"] == []


def test_bad_line_is_not_resolved() -> None:
    module = _load()
    bundle = _bundle("linked.json")
    bundle["nodes"][0]["source_evidence"] = "sources/north-note.md:9"
    report = module.audit_term_links(bundle, _root())
    assert any(item["code"] == "bad_line" for item in report["diagnostics"])
    assert all(item["status"] != "resolved" or item["owner"] != "exam.node.span" for item in report["provenance"])


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
