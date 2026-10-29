"""H03 自测：完整双向引用不误报，断链和同义词不会被自动升格。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h03_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture(name: str) -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_consistent_bundle_has_no_diagnostics() -> None:
    module = _load()
    report = module.check_context_links(_fixture("consistent.json"))
    assert report["diagnostics"] == []
    assert report["suggestions"] == []
    assert report["glossary_written"] is False
    assert report["implemented_nodes"] == []


def test_broken_links_are_specific() -> None:
    module = _load()
    codes = {item["code"] for item in module.check_context_links(_fixture("broken.json"))["diagnostics"]}
    assert codes == {"missing_term", "missing_node", "reverse_mismatch", "missing_source_evidence"}


def test_synonym_is_not_promoted() -> None:
    module = _load()
    report = module.check_context_links(_fixture("aliases.json"))
    assert any(item["code"] == "duplicate_alias" for item in report["diagnostics"])
    assert all(item["applied"] == "false" for item in report["suggestions"])
    assert report["glossary_written"] is False
    blob = json.dumps(report)
    assert "implemented" not in blob or "implemented_nodes" in blob
    assert report["implemented_nodes"] == []


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
