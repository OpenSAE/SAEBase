"""Governance regression for WP19 frozen subset and shared output invariants."""

from __future__ import annotations

from harness.contracts import wp19_reference, wp19_verify


def test_twenty_cases_and_all_200_distinct_ids():
    state = wp19_verify.check_spec()
    assert state == {"cases": 200, "ready": 20, "proposed": 180, "checks": 80}


def test_known_bad_candidate_cannot_fake_a_correct_verdict(tmp_path, monkeypatch):
    run = tmp_path / "fresh-run" / "H27"
    run.mkdir(parents=True)
    monkeypatch.setattr(wp19_verify, "RUNS", tmp_path)
    source = run / "implementation.py"
    source.write_text(
        "def solve(payload):\n"
        "    from harness.contracts import registry\n"
        "    selected = payload['selected']\n"
        "    known = {r['id'] for r in registry.catalog()['contracts']}\n"
        "    if not isinstance(selected, list) or any(not isinstance(x, str) or x not in known for x in selected) or len(set(selected)) != len(selected):\n"
        "        raise ValueError('bad input')\n"
        "    return {'contracts': sorted(set(registry.catalog()['required_always']) | set(selected))}\n",
        encoding="utf-8",
    )
    clean = wp19_verify.verify_case("H27", run)
    assert clean["failed"] == 0
    source.write_text("def solve(payload):\n    return {'contracts': []}\n", encoding="utf-8")
    broken = wp19_verify.verify_case("H27", run)
    assert broken["failed"] >= 2
    assert broken["human_approval"] is False


def test_unqualified_case_rejected(tmp_path):
    for case in ("H47", "H100", "H226"):
        try:
            wp19_verify.verify_case(case, tmp_path)
        except ValueError as exc:
            assert "proposal" in str(exc)
        else:
            raise AssertionError(case)


def test_reference_consistent_with_frozen_positive_and_negative_cases():
    record = wp19_verify.load_spec()
    for case in record["cases"]:
        if case["status"] != "READY":
            continue
        for row in case["checks"]:
            try:
                actual = wp19_reference.solve(case["id"], row["input"])
            except ValueError:
                assert row.get("reject") is True
            else:
                assert actual == row["expected"]


def test_candidate_direct_reference_import_is_rejected(tmp_path, monkeypatch):
    run = tmp_path / "fresh-run" / "H27"
    run.mkdir(parents=True)
    monkeypatch.setattr(wp19_verify, "RUNS", tmp_path)
    (run / "implementation.py").write_text(
        "from harness.contracts.wp19_reference import solve as oracle\n"
        "def solve(payload):\n"
        "    return oracle('H27', payload)\n",
        encoding="utf-8",
    )
    try:
        wp19_verify.verify_case("H27", run)
    except ValueError as exc:
        assert "reference oracle" in str(exc)
    else:
        raise AssertionError("Official reference was callable as candidate solution")
