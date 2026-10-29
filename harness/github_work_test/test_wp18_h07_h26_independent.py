"""Reviewer-owned WP18 H07-H26 fresh adversarial regression.

Preserve the submitted Agent branch and the frozen public score. These cases
probe already documented CT-EVIDENCE / CT-PRIVACY / CT-LOCALE / CT-DATA
requirements, without rewriting the candidate or its published fixtures.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "harness/tests/wp18/runs/20261029-wp18-03"


def candidate(case: str):
    source = RUN / case / "implementation.py"
    spec = importlib.util.spec_from_file_location("review_" + case, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h07_windows_absolute_drive_is_not_repo_relative():
    module = candidate("H07")
    with pytest.raises(ValueError):
        module.solve({"paths": ["C:/Users/private/credentials.txt"]})


def test_h09_private_key_camelcase_is_redacted():
    module = candidate("H09")
    result = module.solve({"payload": {"nested": {"privateKey": "DO_NOT_LEAK"}}})
    assert "DO_NOT_LEAK" not in str(result)


def test_h09_session_token_and_value_embedded_credentials_are_redacted():
    module = candidate("H09")
    result = module.solve({
        "payload": {
            "nested": [{"sessionToken": "DO_NOT_LEAK"}],
            "diagnostic": "token=DO_NOT_LEAK",
        },
    })
    assert "DO_NOT_LEAK" not in str(result)


def test_h13_chinese_placeholder_is_invalid():
    module = candidate("H13")
    row = {
        "id": "code:n", "project": "SAEBase",
        "title": {"zh": "中文翻译不可用", "en": "Named feature"},
        "summary": {"zh": "标记未确认的输入来源", "en": "Reports unresolved input evidence"},
        "description": {"zh": "只读原始数据", "en": "Reads only source data"},
    }
    with pytest.raises(ValueError):
        module.solve({"records": [row]})


def test_h19_finite_operands_must_not_produce_nonfinite_result():
    module = candidate("H19")
    with pytest.raises(ValueError):
        module.solve({"pred": [1e308], "target": [-1e308]})
