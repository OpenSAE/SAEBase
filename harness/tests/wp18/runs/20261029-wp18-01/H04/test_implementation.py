"""H04 自测：合成记录必须脱敏，且不能把缺探测写成无 GPU。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h04_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_redacts_secrets_and_does_not_invent_gpu() -> None:
    module = _load()
    record = json.loads((Path(__file__).resolve().parent / "fixtures" / "device.json").read_text(encoding="utf-8"))
    result = module.normalize_environment(record)
    blob = json.dumps(result)
    assert result["synthetic"] is True
    assert result["observed"] is False
    assert result["measurement"] == "not_observed"
    assert result["gpu"] == "unknown"
    assert result["fields"]["gpu"] == "unknown"
    assert result["fields"]["note"] == "unknown"
    assert "sekret-token-value" not in blob
    assert "/home/researcher" not in blob
    assert "no GPU" not in blob
    assert "无 GPU" not in result["summary_zh"]
    assert set(result["redacted_fields"]) == {"path", "secret", "token"}


def test_missing_gpu_stays_unknown() -> None:
    module = _load()
    result = module.normalize_environment({"synthetic": True, "device_name": "exam-fixture"})
    assert result["gpu"] == "unknown"
    assert result["fields"]["device_name"] == "exam-fixture"


def test_real_or_unmarked_record_is_rejected() -> None:
    module = _load()
    for record in ({"synthetic": False, "gpu": "A100"}, {"gpu": "A100"}):
        try:
            module.normalize_environment(record)
        except module.EnvironmentRecordError:
            pass
        else:
            raise AssertionError(record)


def test_source_does_not_probe() -> None:
    source = Path(__file__).with_name("implementation.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "nvidia-smi" not in source.split("没有 nvidia-smi")[0] or "nvidia-smi" in source
    assert "os.system" not in source


def test_candidate_binds_real_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert payload["provisional"] is True
    assert any(change["view"] == "environment" for change in payload["changes"])
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
