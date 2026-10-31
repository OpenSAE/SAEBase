"""Independent WP22 R2 architecture and execution recovery checks (CPU only)."""

from __future__ import annotations

import sys

import pytest

from experiments.qa500.config import (
    get_assets_lock,
    get_dataset_ids,
    get_protocol,
    get_resources,
    get_sites,
    get_tasks,
)
from experiments.qa500.assets import required_publication_lock
from models.qa_input import _prompt_format
from saebase.runtime.hardware import detect
from saebase.runtime.supervisor import run_worker


def test_model_specific_chat_policy_is_owned_by_model_spec():
    assert _prompt_format("gemma3_4b_it") == "chat_template"
    assert _prompt_format("gemma2_2b_base") == "plain"
    assert _prompt_format("pythia410m") == "plain"
    with pytest.raises(ValueError):
        _prompt_format("../malformed")


def test_scientific_configuration_has_component_boundaries():
    assert get_protocol()["protocol_id"] == "qa_standard_500_v1"
    assert len(get_dataset_ids()) == 10
    assert len(get_sites()) == 10
    assert get_tasks()["budgets"]["capture"] == 500
    assert get_resources()["max_oom_retries"] <= 3
    assert get_assets_lock()["status"] != "pinned"
    # A first-run snapshot or a successful offline test is not publishing authority.
    assert required_publication_lock() is False


def test_hardware_facts_are_real_or_explicitly_unknown(tmp_path):
    report = detect(probe_gpu=False, cache_path=tmp_path)
    assert report["schema"] == "saebase.environment.v1"
    assert report["disk_free_bytes"] >= 0
    assert isinstance(report["cuda_available"], bool)
    if not report["cuda_available"]:
        assert report["gpu"] == []
    else:
        assert len(report["gpu"]) >= 1
        for gpu in report["gpu"]:
            assert gpu["total_vram_bytes"] > 0
            assert 0 <= gpu["free_vram_bytes"] <= gpu["total_vram_bytes"]


def test_worker_success_has_one_real_attempt(tmp_path):
    report = run_worker(
        argv=[sys.executable, "-c", "print('real worker')"],
        receipt_file=tmp_path / "nonexistent.json",
        log_dir=tmp_path / "success",
        retry_limit=1,
    )
    assert report["status"] == "completed"
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["returncode"] == 0
    assert (tmp_path / "success" / "attempt_00.stdout.log").is_file()


def test_worker_does_not_retry_arbitrary_errors(tmp_path):
    report = run_worker(
        argv=[sys.executable, "-c", "raise RuntimeError('not an OOM')"],
        receipt_file=tmp_path / "missing.json",
        log_dir=tmp_path / "ordinary",
        retry_limit=2,
    )
    assert report["status"] == "failed"
    assert len(report["attempts"]) == 1


def test_worker_oom_simulation_is_not_fake_gpu_acceptance(tmp_path):
    report = run_worker(
        argv=[
            sys.executable,
            "-c",
            "import sys; print('CUDA out of memory', file=sys.stderr); sys.exit(8)",
        ],
        receipt_file=tmp_path / "missing.json",
        log_dir=tmp_path / "oom",
        retry_limit=1,
    )
    assert report["status"] == "blocked_resource"
    assert len(report["attempts"]) == 2
    assert report["precision_changed"] is False
    assert report["model_substituted"] is False


def test_publication_asset_lock_rejects_partial_and_unverified_revision_ids(monkeypatch):
    from experiments.qa500 import assets

    sites = get_sites()
    complete = {
        "status": "pinned",
        "dataset_revisions": {key: "a" * 40 for key in get_dataset_ids()},
        "model_revisions": {row["model_key"]: "b" * 40 for row in sites},
        "sae_revisions": {
            f"{row['model_key']}:{row['sae_key']}:L{layer}": "c" * 40
            for row in sites for layer in row["layers"]
        },
    }
    monkeypatch.setattr(assets, "get_assets_lock", lambda: complete)
    assert assets.required_publication_lock() is True

    incomplete = {
        **complete,
        "sae_revisions": dict(list(complete["sae_revisions"].items())[:-1]),
    }
    monkeypatch.setattr(assets, "get_assets_lock", lambda: incomplete)
    assert assets.required_publication_lock() is False

    malformed = {
        **complete,
        "dataset_revisions": {**complete["dataset_revisions"], "mmlu_pro": "x" * 40},
    }
    monkeypatch.setattr(assets, "get_assets_lock", lambda: malformed)
    assert assets.required_publication_lock() is False


def test_gpu_planner_selects_by_current_free_memory_not_device_name():
    from saebase.runtime.resource_planner import select_cuda_device

    profile = {
        "cuda_available": True,
        "gpu": [
            {"index": 0, "name": "A", "free_vram_bytes": 12 << 30,
             "total_vram_bytes": 32 << 30, "cuda_kernel_ok": True},
            {"index": 1, "name": "B", "free_vram_bytes": 20 << 30,
             "total_vram_bytes": 24 << 30, "cuda_kernel_ok": True},
            {"index": 2, "name": "C", "free_vram_bytes": 22 << 30,
             "total_vram_bytes": 32 << 30, "cuda_kernel_ok": False},
        ],
    }
    plan = select_cuda_device(profile)
    assert plan["device"] == "cuda:1"
    assert plan["precision_change"] is False
    assert plan["offload_requested"] is False
    profile["cuda_available"] = False
    with pytest.raises(RuntimeError, match="BLOCKED_ENV_NO_CUDA"):
        select_cuda_device(profile)
