"""Trusted WP20 grader calibration: all cases, real outputs and known bad implementations."""

from __future__ import annotations


import pytest
from harness.contracts import wp20_verify as grader

FIXTURES = grader.load_scenarios()["cases"]
ALL = [(case, row) for case in sorted(FIXTURES) for row in FIXTURES[case]]


@pytest.mark.parametrize(("case", "scenario"), ALL, ids=lambda x: str(x)[:60])
def test_reference_matches_frozen_positive_and_negative_cases(case, scenario):
    result = grader._reference(case, scenario["input"])
    assert result["ok"] is scenario["valid"], (case, scenario["id"])
    assert isinstance(result["files"], dict)
    if scenario["valid"]:
        assert result["kind"] is None
        assert isinstance(result["output"], dict)
        assert result["output"] not in ({"accepted": True}, {"ok": True})
    else:
        assert result["kind"] == "ValueError"
        assert result["output"] is None


def test_auc_average_precision_hand_calculation():
    good = grader._reference("X11", {"labels": [1, 0], "scores": [0.9, 0.1], "threshold": 0.5})
    bad = grader._reference("X11", {"labels": [1, 0], "scores": [0.1, 0.9], "threshold": 0.5})
    assert good["output"]["auc"] == 1 and good["output"]["average_precision"] == 1
    assert bad["output"]["auc"] == 0 and bad["output"]["average_precision"] == 0.5


def test_archive_is_actually_written_and_read():
    result = grader._reference("X16", {"files": {"a.txt": "hello"}, "corrupt": False})
    assert result["ok"]
    assert "bundle.tar" in result["files"]
    assert "extract/a.txt" in result["files"]
    assert result["output"]["verified"] is True


def test_git_has_real_branches_conflict_resolution_and_parents():
    good = grader._reference("X18", grader.load_scenarios()["cases"]["X18"][0]["input"])
    assert good["ok"], good
    out = good["output"]
    assert out["commits"] == 4
    assert out["conflicted"] is True
    assert len({out["base_sha"], out["head_a_sha"], out["head_b_sha"], out["merge_sha"]}) == 4
    assert set(out["merge_parents"]) == {out["head_a_sha"], out["head_b_sha"]}
    assert out["source"] == "def f():\n    return 4\n"
    assert "module.py" in good["files"]
    automatic = grader._reference("X18", grader.load_scenarios()["cases"]["X18"][1]["input"])
    assert automatic["ok"], automatic
    assert automatic["output"]["conflicted"] is False


@pytest.mark.parametrize("case", sorted(grader.READY))
def test_reviewer_gate_kills_constant_and_reject_all_mutants(case, tmp_path, monkeypatch):
    path = tmp_path / "new-run" / case
    path.mkdir(parents=True)
    source = path / "implementation.py"
    source.write_text(
        "def run(payload, workdir):\n    return {'accepted': True}\n", encoding="utf-8"
    )
    monkeypatch.setattr(grader, "RUNS", tmp_path)
    wrong = grader.verify_case(case, path)
    assert wrong["failed"] >= 2, case
    source.write_text(
        "def run(payload, workdir):\n    raise ValueError('always reject')\n", encoding="utf-8"
    )
    rejecting = grader.verify_case(case, path)
    assert rejecting["failed"] >= 2, case
    assert not wrong["human_approved"] and not rejecting["human_approved"]


def test_reference_import_bypass_is_blocked(tmp_path):
    path = tmp_path / "new-run" / "X01"
    path.mkdir(parents=True)
    (path / "implementation.py").write_text(
        "from harness.contracts.wp20_reference_a import x01\n"
        "def run(payload, workdir):\n    return x01(payload, workdir)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official reference"):
        grader._candidate_source(path)


def test_scope_denies_edits_to_verifier(monkeypatch):
    import types

    monkeypatch.setattr(
        grader.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(
            stdout="A\tharness/tests/wp20/runs/fresh-run/X01/implementation.py\n"
            "M\tharness/contracts/wp20_verify.py\n"
        ),
    )
    assert grader.audit_scope("a" * 40) == ["harness/contracts/wp20_verify.py"]


def test_data_manifest_and_coverage_invariant_under_sample_permutation():
    source = grader.load_scenarios()["cases"]["X06"][0]["input"]
    moved = dict(source)
    moved["records"] = list(reversed(source["records"]))
    original = grader._reference("X06", source)
    changed = grader._reference("X06", moved)
    assert original["ok"] and changed["ok"]
    assert original["output"] == changed["output"]


def test_long_horizon_replays_nine_events_with_two_distinct_failures():
    rows = grader.load_scenarios()["cases"]["X20"]
    assert all(len(row["input"]["events"]) >= 8 for row in rows[:2])
    for row in rows[:2]:
        result = grader._reference("X20", row["input"])
        assert result["ok"]
        assert result["output"]["count"] >= 8
        assert len(result["output"]["failures"]) == 2
        assert result["output"]["final"]["status"] == "reviewed"


def test_reference_import_via_helper_module_is_blocked(tmp_path):
    path = tmp_path / "new-run" / "X01"
    (path / "src").mkdir(parents=True)
    (path / "implementation.py").write_text(
        "from src.logic import run\n", encoding="utf-8"
    )
    (path / "src" / "logic.py").write_text(
        "from harness.contracts.wp20_reference_a import x01\n"
        "def run(payload, workdir):\n    return x01(payload, workdir)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official reference"):
        grader._candidate_source(path)
