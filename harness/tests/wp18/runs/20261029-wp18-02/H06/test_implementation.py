"""H06 自测：中英缺项、复述、乱码、错项目和符号误译都留下，且不算已修复。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h06", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict]:
    path = Path(__file__).resolve().parent / "fixtures" / "rows.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_clean_row_keeps_signature() -> None:
    module = _load()
    report = module.review_locale_rows(_rows())
    assert report["fixed"] == []
    assert [item for item in report["detected"] if item["id"] == "code:function:exam.packet:clean_row"] == []
    preserved = {item["id"]: item for item in report["preserved"]}
    assert preserved["code:function:exam.packet:clean_row"]["signature"] == "def north_span()"
    assert preserved["code:function:exam.packet:spaced_row"]["path"] == "harness/tests/wp18/exam_packet.py"


def test_defects_are_detected_and_not_repaired() -> None:
    module = _load()
    found = {(item["id"], item["code"]) for item in module.review_locale_rows(_rows())["detected"]}
    assert ("code:function:exam.packet:missing_en", "missing_en") in found
    assert ("code:function:exam.packet:missing_zh", "missing_zh") in found
    assert ("code:function:exam.packet:period_row", "title_repeats_summary") in found
    assert ("code:function:exam.packet:placeholder_row", "translation_placeholder") in found
    assert ("code:function:exam.packet:broken_row", "undecodable_text") in found
    assert ("code:function:exam.packet:foreign_row", "wrong_project") in found
    assert ("code:function:exam.packet:spaced_row", "symbol_translated") in found
    assert module.review_locale_rows(_rows())["fixed"] == []


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
