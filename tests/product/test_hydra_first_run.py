"""Hydra must not plant train.log in a fresh DLBase run leaf."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(cwd: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["HYDRA_FULL_ERROR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "saebase.runtime.train", f"paths.root={cwd}", *overrides],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _leaf(cwd: Path, sub_name: str) -> Path:
    return cwd / "experiment" / "IOI" / "gemma2_2b_base" / "capture" / sub_name


def test_clean_default_entry_creates_leaf_without_log_redirect(tmp_path: Path):
    result = _run(tmp_path, "experiment.sub_name=prompt_capture", "seed=11")
    assert result.returncode == 0, result.stderr
    leaf = _leaf(tmp_path, "prompt_capture")
    assert (leaf / "metadata" / "manifest.json").is_file()
    assert not (leaf / "train.log").exists()
    assert "hydra.job_logging" not in " ".join(result.args)


def test_project_preview_records_distinct_executions(tmp_path):
    from saebase.runtime.experiment_results import list_executions, read_run

    for filename in ("pyproject.toml", "conf/config.yaml", "base/application/commands.py"):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in ("datasets/indirect_object_identification", "models/gemma/gemma2", "tasks/capture"):
        (tmp_path / name).mkdir(parents=True)
    for seed in (41, 42):
        result = _run(tmp_path, "experiment.sub_name=configuration_contract", f"seed={seed}")
        assert result.returncode == 0, result.stderr
    study = tmp_path / "experiment/indirect_object_identification/gemma/gemma2/capture/configuration_contract"
    records = list_executions(study)
    assert [record["seed"] for record in records] == [41, 42]
    execution = Path(records[-1]["storage"])
    manifest = json.loads((execution / "metadata/manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["run_id"] == execution.name
    assert manifest["configuration"] == json.loads((execution / manifest["effective_config"]).read_text())
    assert (execution / manifest["outputs"]["task_result"]).is_file()
    assert read_run(study)["status"] == "configured_only"


def test_same_purpose_preserves_repeated_executions(tmp_path: Path):
    import tarfile
    from saebase.runtime.experiment_results import list_executions

    folder = _leaf(tmp_path, "activation_stability")
    for seed in (11, 12, 12):
        result = _run(tmp_path, "experiment.sub_name=activation_stability", f"seed={seed}")
        assert result.returncode == 0, result.stderr
        archive_marker = folder / "reports" / "execution_note.txt"
        record_id = json.loads((folder / "metadata/manifest.json").read_text())["execution_id"]
        archive_marker.write_text(record_id)
    records = list_executions(folder)
    assert len(records) == 3
    assert len({r["execution_id"] for r in records}) == 3
    assert sorted(r["seed"] for r in records) == [11, 12, 12]
    for record in records:
        if not record["current"]:
            with tarfile.open(record["storage"]) as archive:
                assert archive.extractfile("reports/execution_note.txt").read().decode() == record["execution_id"]


def test_failed_task_keeps_the_run_leaf(tmp_path: Path):
    discovery = tmp_path / "discovery.json"
    heldout = tmp_path / "heldout.json"
    discovery.write_text("[]")
    heldout.write_text("[]")
    result = _run(
        tmp_path,
        "task=explain",
        "task.execute=true",
        "task.explain.approved=true",
        f"task.explain.discovery_file={discovery}",
        f"task.explain.heldout_file={heldout}",
        "experiment.sub_name=answer_attribution",
        "seed=13",
    )
    assert result.returncode != 0
    leaf = tmp_path / "experiment" / "IOI" / "gemma2_2b_base" / "explain" / "answer_attribution"
    assert (leaf / "metadata" / "manifest.json").is_file()
    assert not (leaf / "train.log").exists()
    assert "generator" in result.stderr.lower() or "RuntimeError" in result.stderr


@pytest.mark.parametrize("name", ["conf/config.yaml"])
def test_openresearch_config_disables_hydra_file_log(name: str):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "override hydra/job_logging: stdout" in text
    assert "max_epochs:" not in text
    assert "mode: null" in text


def test_default_experiment_name_reflects_actual_capture_method(tmp_path: Path):
    result = _run(tmp_path, "task.capture.mode=non_whiten", "experiment.run_id=answer_stability")
    assert result.returncode == 0, result.stderr
    run = _leaf(tmp_path, "non_whiten") / "answer_stability"
    manifest = json.loads((run / "metadata" / "manifest.json").read_text())
    assert manifest["configuration"]["experiment"] == {
        "sub_name": "non_whiten",
        "run_id": "answer_stability",
    }


def test_invalid_experiment_name_is_rejected_before_creating_run(tmp_path: Path):
    result = _run(tmp_path, "experiment.sub_name=qa_capture_expansion_20261119")
    assert result.returncode != 0
    assert "one to three lowercase words" in result.stderr
    assert not (tmp_path / "experiment").exists()
