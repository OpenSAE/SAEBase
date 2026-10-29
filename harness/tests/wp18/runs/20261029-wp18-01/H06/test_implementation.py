"""H06 自测：缺译、复述、乱码、错项目和符号误译都要留下，且不算已修复。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h06_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _records() -> list[dict]:
    path = Path(__file__).resolve().parent / "fixtures" / "overlay.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_good_record_is_clean_and_symbols_stay() -> None:
    module = _load()
    report = module.audit_overlay(_records())
    assert report["fixed"] == []
    good = [item for item in report["detected"] if item["id"] == "code:function:demo:good"]
    assert good == []
    preserved = {item["id"]: item for item in report["preserved"]}
    assert preserved["code:function:demo:good"]["signature"] == "def index_files(root)"
    assert preserved["code:function:demo:symbol"]["path"] == "harness/tests/wp18/fixture.py"


def test_each_defect_is_detected_not_fixed() -> None:
    module = _load()
    report = module.audit_overlay(_records())
    found = {(item["id"], item["code"]) for item in report["detected"]}
    assert ("code:function:demo:missing", "missing_english") in found
    assert ("code:function:demo:period", "title_repeats_summary") in found
    assert ("code:function:demo:placeholder", "translation_placeholder") in found
    assert ("code:function:demo:mojibake", "undecodable_text") in found
    assert ("code:function:demo:project", "wrong_project") in found
    assert ("code:function:demo:symbol", "symbol_translated") in found
    assert report["fixed"] == []
    assert "authority translation tables was edited" in report["note"]


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
