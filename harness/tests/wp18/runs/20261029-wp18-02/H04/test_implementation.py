"""H04 自测：嵌套脱敏、未知不是无 GPU，错误类型整份拒绝。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h04", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nested_secrets_do_not_survive() -> None:
    module = _load()
    record = json.loads((Path(__file__).resolve().parent / "fixtures" / "device.json").read_text(encoding="utf-8"))
    result = module.normalize_environment(record)
    blob = json.dumps(result)
    assert result["synthetic"] is True
    assert result["observed"] is False
    assert result["measurement"] == "not_observed"
    assert result["gpu"] == "unknown"
    assert result["fields"]["note"] == "unknown"
    assert "SURFACE-TOKEN-41" not in blob
    assert "NESTED-SECRET-41" not in blob
    assert "/home/bench/hidden" not in blob
    assert "no GPU" not in blob
    assert "无 GPU" not in result["summary_zh"]
    assert set(result["redacted_fields"]) == {"$.nested.path", "$.nested.secret", "$.token"}


def test_missing_gpu_stays_unknown() -> None:
    module = _load()
    result = module.normalize_environment({"synthetic": True, "device_name": "bench-card"})
    assert result["gpu"] == "unknown"
    assert result["fields"]["device_name"] == "bench-card"


def test_unsupported_values_fail_closed() -> None:
    module = _load()
    secret = "LEAK-IF-MESSAGE-42"
    samples = (
        {"synthetic": False, "token": secret},
        {"gpu": "A100"},
        {"synthetic": True, "flag": True, "token": secret},
        {"synthetic": True, "box": {"a": {"b": {"c": {"d": {"e": secret}}}}}},
    )
    for record in samples:
        try:
            module.normalize_environment(record)
        except module.EnvironmentRefusal as exc:
            assert secret not in str(exc)
        else:
            raise AssertionError(record)


def test_candidate_matches_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert payload["provisional"] is True
    assert any(change["view"] == "environment" and change["status"] == "unchecked" for change in payload["changes"])
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
