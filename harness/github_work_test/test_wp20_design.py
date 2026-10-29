"""WP20 design validation: no false READY claims, all contracts source-backed."""

from __future__ import annotations

import copy

import pytest

from harness.contracts import wp20_design_validate as wp20


def test_wp20_has_twenty_real_multistage_descriptions():
    result = wp20.validate()
    assert result["cases"] == 20
    assert result["groups"] == 4
    assert result["stage_count"] >= 80
    assert result["normal_scenarios"] >= 40
    assert result["negative_scenarios"] >= 40
    assert result["source_references_checked"] >= 50
    assert result["status"] == "READY_FOR_BEHAVIORAL_EXERCISE"


def test_design_cannot_mark_unqualified_reference_ready():
    data = copy.deepcopy(wp20.load_spec())
    data["cases"][0]["oracle_status"] = "NOT_IMPLEMENTED"
    with pytest.raises(ValueError, match="missing qualified reference"):
        wp20.validate(data)


def test_design_requires_true_negative_controls():
    data = copy.deepcopy(wp20.load_spec())
    data["cases"][1]["negative_cases"] = []
    with pytest.raises(ValueError, match="negative_cases"):
        wp20.validate(data)


def test_design_references_existing_source_not_invented_paths():
    data = copy.deepcopy(wp20.load_spec())
    data["cases"][2]["source_refs"] = ["saebase/dlbase/retired.py", "harness/code/build.py"]
    with pytest.raises(ValueError, match="source ref missing"):
        wp20.validate(data)
