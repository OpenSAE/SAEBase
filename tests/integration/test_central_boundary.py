"""Thin integration stays separate from the installable scientific runtime."""

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from scripts import documentation as check_documentation

ROOT = Path(__file__).resolve().parents[2]


def check_central(action, root=ROOT, *, expected_engine=None):
    central_root = os.getenv("RH_HARNESS_ROOT")
    binding = root / ".researchharness/central.json"
    if not central_root and binding.is_file():
        central_root = json.loads(binding.read_text())["harness_root"]
    if not central_root:
        raise RuntimeError(
            "Set RH_HARNESS_ROOT or .researchharness/central.json to select a central workspace"
        )
    try:
        from harness.central import load_harness
        from harness.cli import main
    except ImportError as exc:
        raise RuntimeError("ResearchHarness is not installed in the current environment") from exc
    if expected_engine is not None:
        from harness.runtime import engine_identity

        expected = json.loads(Path(expected_engine).read_text())
        if engine_identity() != expected:
            raise RuntimeError("The central engine does not match the requested version")
    context = load_harness(central_root)
    project = context.project("saebase")
    if project.root != root.resolve():
        raise RuntimeError(
            "Central registration SAEBase Different path to current project directory"
        )
    args = ["--harness-root", str(context.root), "--project", project.project_id]
    return main(args + [action])


def test_project_contains_inputs_and_no_local_management_engine():
    config = yaml.safe_load((ROOT / ".researchharness/project.yaml").read_text())
    assert set(config) == {
        "schema_version",
        "project_id",
        "display_name",
        "paths",
        "code_roots",
    }
    assert config["schema_version"] == 2 and config["project_id"] == "saebase"
    assert not (ROOT / "harness").exists()
    assert not set(config["paths"]) & {
        "core",
        "environment",
        "contracts",
        "workplans",
        "state",
        "atlas",
        "adapter_sources",
    }
    for relative in config["paths"].values():
        path = ROOT / relative
        assert path.resolve().is_relative_to(ROOT) and path.is_file()
    annotations = json.loads((ROOT / config["paths"]["code_annotations"]).read_text())
    for row in annotations["mappings"]:
        assert all(value.startswith("rh.core") for value in row["core_ids"])


def test_active_workflows_call_existing_product_tests_and_no_retired_generators():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        content = path.read_text()
        assert not re.search(
            r"python(?:3)?\s+(?:-m\s+harness\.(?:github|contracts)|harness/)", content
        ), path
        for target in re.findall(r"tests/product/[A-Za-z0-9_/.]+\.py", content):
            assert (ROOT / target).is_file(), (path, target)


def test_missing_central_workspace_fails_with_actionable_diagnostic(monkeypatch, tmp_path):
    monkeypatch.delenv("RH_HARNESS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="select a central workspace"):
        check_central("validate", root=tmp_path)


def test_scientific_install_does_not_supply_management_engine(monkeypatch):
    monkeypatch.setenv("RH_HARNESS_ROOT", "/explicitly-provisioned-central")
    monkeypatch.setitem(sys.modules, "harness", None)
    with pytest.raises(RuntimeError, match="ResearchHarness is not installed"):
        check_central("validate")


def test_document_navigation_uses_current_pages_and_skips_historical_sources(tmp_path, monkeypatch):
    historical = tmp_path / "history.md"
    historical.write_text("Historical source [old](retired/path.md)\n")
    current = tmp_path / "current.md"
    target = tmp_path / "target.md"
    target.write_text("Current destination\n")
    current.write_text("[read](target.md)\n")
    receipt = tmp_path / ".researchharness/historical-documents.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps({"files": [{"path": "history.md", "authorization": "HISTORICAL_ONLY"}]})
    )
    monkeypatch.setattr(check_documentation, "markdown_files", lambda root: [historical, current])
    assert check_documentation.validate(tmp_path) == (1, [])
    target.unlink()
    assert "current.md" in check_documentation.validate(tmp_path)[1][0]


def test_status_uses_current_engine_and_version_acceptance_is_explicit(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("RH_HARNESS_ROOT", str(tmp_path / "central"))
    calls = []
    modules = {
        name: ModuleType(name)
        for name in ("harness", "harness.central", "harness.cli", "harness.runtime")
    }
    context = SimpleNamespace(
        root=tmp_path / "central",
        project=lambda name: SimpleNamespace(root=tmp_path, project_id=name),
    )
    modules["harness.central"].load_harness = lambda path: context
    modules["harness.cli"].main = lambda args: calls.append(args) or 0
    modules["harness.runtime"].engine_identity = lambda: {"version": "current"}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    assert check_central("status", root=tmp_path) == 0
    assert calls[0][-1] == "status"
    expected = tmp_path / "engine.json"
    expected.write_text(json.dumps({"version": "selected-for-acceptance"}))
    with pytest.raises(RuntimeError, match="does not match the requested version"):
        check_central("status", root=tmp_path, expected_engine=expected)
