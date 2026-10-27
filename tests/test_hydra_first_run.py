"""Hydra must not plant train.log in a fresh DLBase run leaf."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(cwd: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["HYDRA_FULL_ERROR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "dlbase.train", *overrides],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _leaf(cwd: Path, sub_name: str, seed: str) -> Path:
    return cwd / "experiment" / "IOI" / "gemma2_2b_base" / "capture" / f"{sub_name}_{seed}"


def test_clean_default_entry_creates_leaf_without_log_redirect(tmp_path: Path):
    result = _run(tmp_path, "experiment.sub_name=d1clean", "seed=11")
    assert result.returncode == 0, result.stderr
    leaf = _leaf(tmp_path, "d1clean", "11")
    assert (leaf / "metadata" / "run.json").is_file()
    assert not (leaf / "train.log").exists()
    assert "hydra.job_logging" not in " ".join(result.args)


def test_same_run_id_is_refused_and_different_seed_is_independent(tmp_path: Path):
    first = _run(tmp_path, "experiment.sub_name=d1same", "seed=11")
    assert first.returncode == 0, first.stderr
    second = _run(tmp_path, "experiment.sub_name=d1same", "seed=11")
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr or "artifacts" in second.stderr
    assert (_leaf(tmp_path, "d1same", "11") / "metadata" / "run.json").is_file()
    other = _run(tmp_path, "experiment.sub_name=d1same", "seed=12")
    assert other.returncode == 0, other.stderr
    assert (_leaf(tmp_path, "d1same", "12") / "metadata" / "run.json").is_file()


def test_failed_task_keeps_the_run_leaf(tmp_path: Path):
    result = _run(
        tmp_path,
        "task=explain",
        "task.execute=true",
        "task.approved=true",
        "experiment.sub_name=d1fail",
        "seed=13",
    )
    assert result.returncode != 0
    leaf = tmp_path / "experiment" / "IOI" / "gemma2_2b_base" / "explain" / "d1fail_13"
    assert (leaf / "metadata" / "run.json").is_file()
    assert not (leaf / "train.log").exists()
    assert "generator" in result.stderr.lower() or "RuntimeError" in result.stderr


@pytest.mark.parametrize("name", ["conf/openresearch.yaml"])
def test_openresearch_config_disables_hydra_file_log(name: str):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "override hydra/job_logging: stdout" in text
    assert "max_epochs:" in text
