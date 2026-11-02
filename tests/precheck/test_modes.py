"""CPU-only Precheck contract tests. None of these runs model weights."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tasks.precheck.report import blocked, error_result, summarize
from tasks.precheck.sae_checks import select_sites
from tasks.precheck.operation import _selected_tasks, _budget, run_precheck


def test_model_default_and_explicit_sae_selection():
    model = OmegaConf.create({
        "name": "gemma2_2b_base",
        "sae_profiles": [
            "gemma2_2b_pt_l25_canonical",
            "gemma2_2b_it_scope_resid_l12",
        ],
        "default_saes": ["gemma2_2b_pt_l25_canonical"],
    })
    defaults = select_sites(model, "default")
    assert len(defaults) == 1
    assert defaults[0]["sae_key"] == "gemma2_2b_pt_l25_canonical"
    assert defaults[0]["layer"] == 25
    assert len(select_sites(model, "all")) == 2
    assert select_sites(model, ["gemma2_2b_pt_l25_canonical"], layer=13)[0]["layer"] == 13
    with pytest.raises(ValueError, match="not in model"):
        select_sites(model, "qwen35_2b_resid_w32k")
    with pytest.raises(ValueError, match="Duplicate"):
        select_sites(model, ["gemma2_2b_pt_l25_canonical"] * 2)


def test_unconfigured_sae_model_is_not_silently_substituted():
    model = OmegaConf.create({"name": "qwen25_3b", "sae_profiles": [], "default_saes": []})
    with pytest.raises(ValueError, match="no default SAE"):
        select_sites(model, "default")


def test_modes_and_budgets_are_bounded():
    task = OmegaConf.create({"task": {"mode": "task", "check_tasks": ["capture"], "max_seconds": 30, "max_samples": 2, "max_tokens": 8}})
    assert _selected_tasks(task, "task") == ["capture"]
    assert _selected_tasks(task, "sae") == []
    assert _budget(task) == (30, 2, 8)
    task.task.check_tasks = ["capture", "capture"]
    with pytest.raises(ValueError, match="unique"):
        _selected_tasks(task, "task")
    task.task.max_seconds = 0
    with pytest.raises(ValueError, match="bounded"):
        _budget(task)


def test_precheck_configuration_does_not_execute_any_test(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({
        "dataset": {"name": "IOI"},
        "model": {"name": "gemma2_2b_base", "sae_profiles": [], "default_saes": []},
        "task": {"name": "precheck", "mode": "sae", "execute": False},
    })
    outcome = run_precheck(cfg)
    assert outcome.ok is False
    assert outcome.summary["status"] == "NOT_RUN"
    assert (tmp_path / "metadata/precheck_result.json").is_file()
    assert (tmp_path / "metadata/precheck_audit.json").is_file()


def test_blocked_report_has_actionable_diagnostics():
    row = blocked(stage="setup", code="BLOCKED_ASSET", reason="Not cached",
                  attempted=["local cache lookup"], repair="Authorize download or stage weights")
    assert row["status"] == "BLOCKED"
    assert row["attempted"] and row["repair"] and row["retryable"]
    assert summarize([{"status": "PASS"}, row])["status"] == "BLOCKED"
    assert summarize([row, {"status": "FAIL"}])["status"] == "FAIL"
    assert summarize([])["status"] == "NOT_RUN"
    unknown = error_result(RuntimeError("real model runtime failed"), stage="sae", attempted=["run"])
    assert unknown["status"] == "FAIL"


def test_setup_is_preparation_only_and_no_smoke_module_exists():
    import saebase.setup.assets as assets
    assert not Path(assets.__file__).with_name("smoke.py").exists()
    receipt = assets.prepare_environment(required_modules=(), assets=[])
    assert receipt["status"] == "READY"
    assert receipt["scientific_test"] == "NOT_RUN"
    assert receipt["gpu_test"] == "NOT_RUN"
