"""Purpose and main functions: Exercise local run recording, lookup and metric comparison.

Scope: Temporary filesystem records; no task or model execution.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from saebase.results import EvaluationResult
from saebase.runtime.experiment_results import (
    compare_runs,
    find_reusable_runs,
    list_runs,
    read_run,
    record_run,
)


def config():
    return OmegaConf.create(
        {
            "task": {"name": "capture"},
            "model": {"name": "custom"},
            "dataset": {"name": "local"},
            "seed": 7,
        }
    )


def test_only_explicit_completed_input_identity_enables_lookup(tmp_path):
    cfg = config()
    identity = {"model_sha256": "a" * 64, "sae_sha256": "b" * 64, "data_sha256": "c" * 64}
    result = EvaluationResult.from_payload(
        "capture", cfg, {"status": "executed", "input_identity": identity}
    )
    record_run(tmp_path / "complete", cfg, result=result)
    result = EvaluationResult.from_payload(
        "capture", cfg, {"status": "configured_only", "input_identity": identity}
    )
    record_run(tmp_path / "plan", cfg, result=result)
    assert len(list_runs(tmp_path, model="custom")) == 2
    matches = find_reusable_runs(tmp_path, identity)
    assert len(matches) == 1 and matches[0]["seed"] == 7
    assert find_reusable_runs(tmp_path, {**identity, "data_sha256": "d" * 64}) == []
    with pytest.raises(ValueError, match="identity"):
        find_reusable_runs(tmp_path, {})


def test_error_record_preserves_the_failure(tmp_path):
    record_run(tmp_path, config(), error={"type": "ValueError", "message": "invalid sample"})
    result = read_run(tmp_path)
    assert result["status"] == "failed"
    assert result["error"]["type"] == "ValueError"


@pytest.mark.parametrize("absolute_precheck", [False, True])
def test_read_run_resolves_known_historical_artifacts_in_current_run(tmp_path, absolute_precheck):
    import json

    cfg = config()
    cfg.task.name = "benchmark"
    result = EvaluationResult.from_payload("benchmark", cfg, {"status": "executed"})
    record_run(tmp_path, cfg, result=result)
    manifest = tmp_path / "metadata/manifest.json"
    saved = json.loads(manifest.read_text())
    old_result = "/old/experiment/local/custom/benchmark/prior/metadata/benchmark_result.json"
    saved["result"]["summary"]["result_path"] = old_result
    old_precheck = "tmp/automatic_precheck/metadata/precheck_result.json"
    if absolute_precheck:
        old_precheck = f"/old/experiment/local/custom/benchmark/prior/{old_precheck}"
    saved["result"]["artifact_paths"] = {
        "result": old_result,
        "precheck": old_precheck,
    }
    manifest.write_text(json.dumps(saved))
    current_result = tmp_path / "metadata/benchmark_result.json"
    receipt = tmp_path / "metadata/automatic_precheck/metadata/precheck_result.json"
    receipt.parent.mkdir(parents=True)
    for path in (current_result, receipt):
        path.write_text("{}")

    found = read_run(tmp_path)["result"]
    assert found["summary"]["result_path"] == str(current_result)
    assert found["artifact_paths"] == {"result": str(current_result), "precheck": str(receipt)}
    assert json.loads(manifest.read_text()) == saved


def test_query_reads_archived_runs_and_prefers_the_restored_record(tmp_path):
    import json
    import shutil

    experiment = tmp_path / "experiment"
    original = experiment / "local/custom/capture/cohort_ranking"
    archive = tmp_path / "assets/history_experiment/local_runs/files/experiment"
    saved = archive / original.relative_to(experiment)
    cfg = config()
    result = EvaluationResult.from_payload("capture", cfg, {"status": "executed"})
    record_run(saved, cfg, result=result)
    assert read_run(original)["path"] == str(saved)

    restored = experiment / "local/custom/capture/cohort_ranking/executions/20261122T000000Z-test"
    shutil.copytree(saved, restored)
    manifest = restored / "metadata/manifest.json"
    restored_record = json.loads(manifest.read_text())
    restored_record["archived_manifest"] = str((saved / "metadata/manifest.json").relative_to(tmp_path))
    restored_record["task_status"] = "executed"
    restored_record["status"] = "succeeded"
    manifest.write_text(json.dumps(restored_record))

    records = list_runs(experiment, status="executed")
    assert [item["path"] for item in records] == [str(restored)]


def test_metric_comparison_checks_the_task_owned_ledger(tmp_path):
    import hashlib
    import json

    from tasks.benchmark.results.metrics import atomic_json

    cfg = config()
    cfg.task.name = "benchmark"
    for side, value in (("left", 1.0), ("right", 1.5)):
        root = tmp_path / side
        record_run(
            root,
            cfg,
            result=EvaluationResult.from_payload("benchmark", cfg, {"status": "executed"}),
        )
        folder = root / "metrics" / "measured"
        folder.mkdir(parents=True)
        row = {
            "metric_id": "core.l0",
            "metric_version": "1",
            "protocol_id": "same",
            "dataset_id": "same",
            "split": "test",
            "seed": 7,
            "sample_unit": "token",
            "condition_id": None,
            "site": {"layer": 0},
            "status": "completed",
            "value": value,
            "model_id": "synthetic",
            "sae_id": "toy",
            "metadata": {"data_sha256": "a" * 64, "protocol_configuration_sha256": "b" * 64},
        }
        content = (json.dumps(row) + "\n").encode()
        (folder / "metrics.jsonl").write_bytes(content)
        atomic_json(
            folder / "metrics_manifest.json",
            {"run_id": "same", "sha256": hashlib.sha256(content).hexdigest(), "rows": 1},
        )
    result = compare_runs(tmp_path / "left", tmp_path / "right")
    assert result["measurements"][0]["delta"] == 0.5
    (tmp_path / "right" / "metrics" / "measured" / "metrics.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="integrity"):
        compare_runs(tmp_path / "left", tmp_path / "right")


def test_blocked_result_explains_cause_and_next_step_after_readback(tmp_path):
    from saebase.runtime.experiment_results import inspect_run

    cfg = config()
    cfg.task.name = "benchmark"
    row = {
        "status": "BLOCKED",
        "stage": "setup",
        "code": "BLOCKED_DEPENDENCY",
        "reason": "Missing module: example_backend",
        "repair": "Install the selected backend and retry.",
    }
    result = EvaluationResult(
        ok=False,
        task="benchmark",
        evaluation="benchmark",
        model_key="custom",
        prompt_source="local",
        status="blocked",
        error="Automatic precheck did not pass",
        summary={"automatic_precheck": {"meta": {"rows": [row]}}},
    )
    record_run(tmp_path, cfg, result=result)
    saved = read_run(tmp_path)["result"]
    issue = saved["diagnostics"][0]
    assert issue["reason"] == row["reason"]
    assert issue["next_steps"] == [row["repair"]]
    assert issue["impact"]
    inspected = inspect_run(tmp_path)
    assert inspected["state"] == "blocked"
    assert inspected["diagnostics"] == saved["diagnostics"]


def test_unexplained_legacy_blockage_is_honest_and_actionable():
    result = EvaluationResult.from_payload("capture", config(), {"status": "blocked"})
    issue = result.diagnostics[0]
    assert issue["code"] == "CAUSE_NOT_RECORDED"
    assert issue["next_steps"]
    assert "rclone" not in issue["reason"]


def test_result_asset_remains_usable_after_moving_run(tmp_path, monkeypatch):
    import json
    import shutil

    from saebase.runtime.experiment_results import inspect_run
    from saebase.runtime.sae_operations import _maybe_save

    source = tmp_path / "original"
    source.mkdir()
    cfg = config()
    cfg.task.name = "benchmark"
    monkeypatch.chdir(source)
    payload = _maybe_save("benchmark", {"status": "executed", "value": 3.5}, cfg)
    payload["required_artifacts"] = [{"path": str(source / "metadata/benchmark_result.json")}]
    record_run(source, cfg, result=EvaluationResult.from_payload("benchmark", cfg, payload))
    raw = json.loads((source / "metadata/manifest.json").read_text())
    assert raw["result"]["artifact_paths"]["result"] == "metadata/benchmark_result.json"
    destination = tmp_path / "moved"
    monkeypatch.chdir(tmp_path)
    shutil.move(source, destination)
    saved = read_run(destination)
    result_file = Path(saved["result"]["artifact_paths"]["result"])
    assert result_file == destination / "metadata/benchmark_result.json"
    assert json.loads(result_file.read_text())["value"] == 3.5
    assert inspect_run(destination)["state"] == "complete"


def test_historical_absolute_result_links_resolve_inside_moved_run(tmp_path):
    import json

    cfg = config()
    cfg.task.name = "benchmark"
    record_run(
        tmp_path,
        cfg,
        result=EvaluationResult.from_payload("benchmark", cfg, {"status": "executed"}),
    )
    file = tmp_path / "metadata/benchmark_result.json"
    file.write_text('{"value": 9}')
    manifest = tmp_path / "metadata/manifest.json"
    value = json.loads(manifest.read_text())
    old = "/previous/machine/run/metadata/benchmark_result.json"
    value["result"]["artifact_paths"] = {"result": old}
    value["result"]["summary"]["result_path"] = old
    manifest.write_text(json.dumps(value))
    saved = read_run(tmp_path)
    assert saved["result"]["artifact_paths"]["result"] == str(file)
    assert json.loads(Path(saved["result"]["summary"]["result_path"]).read_text())["value"] == 9
    assert json.loads(manifest.read_text())["result"]["artifact_paths"]["result"] == old


def test_query_keeps_good_results_and_explains_unreadable_records(tmp_path):
    record_run(
        tmp_path / "good",
        config(),
        result=EvaluationResult.from_payload("capture", config(), {"status": "executed"}),
    )
    bad = tmp_path / "bad/metadata"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{")
    errors = []
    records = list_runs(tmp_path, errors=errors)
    assert len(records) == 1 and records[0]["model"] == "custom"
    assert errors[0]["path"] == str(bad / "manifest.json")
    assert errors[0]["reason"] and errors[0]["next_steps"]
    with pytest.warns(RuntimeWarning, match="Unreadable run"):
        assert len(list_runs(tmp_path)) == 1


def test_current_configuration_has_one_snapshot_and_original_is_preserved(tmp_path):
    import json

    cfg = config()
    record_run(
        tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, {"status": "executed"})
    )
    cfg.seed = 99
    record_run(
        tmp_path, cfg, result=EvaluationResult.from_payload("capture", cfg, {"status": "executed"})
    )
    saved = json.loads((tmp_path / "metadata/manifest.json").read_text())
    assert saved["configuration"]["seed"] == 99
    assert saved["original_configuration"]["seed"] == 7
    assert "config_sha256" not in saved
    assert not (tmp_path / "metadata/latest_configuration.json").exists()
    assert "execution_status" not in saved["result"]["meta"]
    assert "executed" not in saved["result"]["meta"]
