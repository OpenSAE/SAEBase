"""Fail-closed Dataset source tests; fixtures are not actual scientific sources."""

from pathlib import Path

from tasks.precheck.dataset_checks import check_dataset_artifact


def test_unprepared_real_dataset_is_never_pass(tmp_path: Path):
    result = check_dataset_artifact(
        dataset_key="CounterFact",
        model_key="gemma1b",
        data_root=tmp_path,
    )
    assert result["status"] in {"BLOCKED", "FAIL"}
    assert result["status"] != "PASS"


def test_unsafe_budget_rejected(tmp_path: Path):
    try:
        check_dataset_artifact(
            dataset_key="CounterFact",
            model_key="gemma1b",
            data_root=tmp_path,
            max_records=0,
        )
    except ValueError as exc:
        assert "bounded" in str(exc)
    else:
        raise AssertionError("expected bounded dataset budget")
