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
    "saebase": ("main", "saebase.runtime.train"),
    "saebase-prepare-prompts": ("prepare_prompts", "saebase.datasets.prepare"),
    "saebase-check-env": ("check_env", "saebase.runtime"),
    "saebase-intervene": ("intervene", "tasks.intervene.commands"),
    "saebase-benchmark": ("benchmark", "tasks.benchmark.commands"),
    "saebase-audit": ("audit", "tasks.benchmark.reliability.commands"),
    "saebase-ground-truth": ("ground_truth", "tasks.benchmark.reliability.controls_cli"),
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


def test_components_share_the_public_registry():
    from models.plugins import MODEL_KEYS

    ensure_plugins_loaded()
    ensure_plugins_loaded()
    assert set(list_registered_model_keys()) == set(MODEL_KEYS)
    from saebase.registry import get_task_plugin

    registered = set(list_registered_task_keys())
    shipped = {path.parent.name for path in (ROOT / "tasks").glob("*/task.py")}
    assert registered == shipped
    assert all(get_task_plugin(name).task_class().name == name for name in registered)
    assert "pythia70m_resid_sm" in list_registered_sae_keys()


def test_source_manifest_excludes_local_research_evidence():
    manifest = (ROOT / "doc/code/distribution/source/manifest.in").read_text(encoding="utf-8")
    assert (
        "recursive-exclude assets/history_experiment/benchmark_evidence/files/stage_d_evidence *"
        in manifest
    )
