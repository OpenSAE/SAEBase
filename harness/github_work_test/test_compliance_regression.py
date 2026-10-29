"""Regression: general invariants detect WP18 class failures without test-specific strings."""

from __future__ import annotations

import pytest

from harness.contracts import compliance, invariants


def test_windows_paths_and_parent_escape_rejected(tmp_path):
    for value in ("C:/Users/private/file", "../private", "/home/private", "a//b", "~/private"):
        with pytest.raises(invariants.InvariantError):
            compliance.canonical_relative_path(value)
    assert compliance.canonical_relative_path("saebase/data/input.py") == "saebase/data/input.py"


def test_nested_secret_names_and_token_values_detected(tmp_path):
    payload = {
        "nested": {"privateKey": "SYNTHETIC", "sessionToken": "SYNTHETIC"},
        "message": "token=SYNTHETIC",
    }
    issues = compliance.audit_artifacts(
        ["CT-PRIVACY"], [{"kind": "structured_output", "payload": payload}], tmp_path
    )
    assert issues and issues[0].startswith("0:structured_output")
    assert (
        compliance.audit_artifacts(
            ["CT-PRIVACY"],
            [{"kind": "structured_output", "payload": {"gpu": None, "tag": "safe"}}],
            tmp_path,
        )
        == []
    )


def test_chinese_placeholder_is_caught_by_existing_locale_contract(tmp_path):
    record = {
        "title": {"zh": "中文翻译不可用", "en": "Clean title"},
        "summary": {"zh": "说明功能", "en": "Describes a purpose"},
        "description": {"zh": "指出用途", "en": "Explains valid input"},
    }
    assert compliance.audit_artifacts(
        ["CT-LOCALE"], [{"kind": "bilingual_record", "payload": record}], tmp_path
    )


def test_finite_inputs_do_not_imply_finite_results(tmp_path):
    result = 1e308 - (-1e308)
    assert result == float("inf")
    assert compliance.audit_artifacts(
        ["CT-DATA"], [{"kind": "numeric_output", "payload": {"mae": result}}], tmp_path
    )
    assert (
        compliance.audit_artifacts(
            ["CT-DATA"], [{"kind": "numeric_output", "payload": {"mae": 0.25}}], tmp_path
        )
        == []
    )


def test_bad_artifact_and_symlink_escape_fail_closed(tmp_path):
    assert compliance.audit_artifacts([], [{"kind": "unknown", "payload": {}}], tmp_path)
    outside = tmp_path.parent / "private-target"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    assert compliance.audit_artifacts(
        ["CT-EVIDENCE"], [{"kind": "relative_path", "payload": "link/file"}], tmp_path
    )
