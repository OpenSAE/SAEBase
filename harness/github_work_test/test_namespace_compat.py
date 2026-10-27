"""Namespace compatibility tests without model weights or network access."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from saebase import cli
from saebase.plugin_bootstrap import ensure_plugins_loaded
from saebase.registry import (
    list_registered_model_keys,
    list_registered_sae_keys,
    list_registered_task_keys,
)

ROOT = Path(__file__).resolve().parents[2]

ENTRYPOINTS = {
    "saebase": ("main", "saebase.cli_compat"),
    "saebase-prepare-prompts": ("prepare_prompts", "saebase.tools.sae_prepare_prompts"),
    "saebase-check-env": ("check_env", "saebase.tools.check_env"),
    "saebase-intervene": ("intervene", "saebase.tools.m2_batch"),
    "saebase-benchmark": ("benchmark", "saebase.tools.m3_benchmark"),
    "saebase-audit": ("audit", "saebase.tools.m4_reliability"),
    "saebase-ground-truth": ("ground_truth", "saebase.tools.m4_ground_truth"),
    "saebase-release-matrix": ("release_matrix", "saebase.tools.m5_release_matrix"),
    "saebase-release-gate": ("release_gate", "saebase.tools.m5_release_gate"),
}


@pytest.mark.parametrize("command,definition", ENTRYPOINTS.items())
def test_namespaced_cli_preserves_existing_dispatch(monkeypatch, command, definition):
    method, module = definition
    seen = []

    def fake_dispatch(path: str) -> int:
        seen.append(path)
        return 17

    monkeypatch.setattr(cli, "_dispatch", fake_dispatch)
    assert getattr(cli, method)() == 17, command
    assert seen == [module]


def test_built_entrypoints_use_namespaced_cli():
    info = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    configured = info["project"]["scripts"]
    assert set(configured) == set(ENTRYPOINTS)
    for command, (method, _) in ENTRYPOINTS.items():
        assert configured[command] == f"saebase.cli:{method}"


def test_legacy_plugin_paths_share_single_registration_state():
    import models.plugins as legacy_models
    import saebase.model_plugins as named_models
    import saebase.tasks.plugins as named_tasks
    import tasks.plugins as legacy_tasks

    ensure_plugins_loaded()
    ensure_plugins_loaded()
    assert legacy_models._HOOKED_KEYS is named_models._HOOKED_KEYS
    assert legacy_models._register_builtin_models is named_models._register_builtin_models
    assert legacy_tasks.SAETask is named_tasks.SAETask
    assert "sae" in list_registered_task_keys()
    assert "pythia70m" in list_registered_model_keys()
    assert "pythia70m_resid_sm" in list_registered_sae_keys()


def test_source_manifest_keeps_evidence_in_git_but_out_of_distribution():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    packaging = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "recursive-exclude conf/coverage/stage_d_evidence *" in manifest
    excluded = packaging["tool"]["setuptools"]["exclude-package-data"]["conf"]
    assert "coverage/stage_d_evidence/*" in excluded
    assert (ROOT / "conf/coverage/stage_d_evidence/final/stage_d_final_matrix.json").is_file()
