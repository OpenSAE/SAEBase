"""Synthetic central maintenance candidate tests; production authorization is never changed."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


CANDIDATE = Path(__file__).resolve().parents[2] / "experiment/central-maintenance/src/harness"


def load_candidate(monkeypatch, name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, CANDIDATE / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_candidate(monkeypatch, tmp_path):
    central = tmp_path / "central"
    project = tmp_path / "project"
    central.mkdir()
    project.mkdir()

    class Harness:
        root = central

        def resolve(self, relative):
            return self.root / relative

        def project_root(self, project_id):
            assert project_id == "saebase"
            return project

        def profile(self, project_id):
            return self.root / "src/project" / project_id

    harness = Harness()
    identity_policy = harness.resolve(".workspace/harness/core/security/identity_policy.json")
    identity_policy.parent.mkdir(parents=True)
    for directory in identity_policy.parents:
        if directory == harness.root:
            break
        directory.chmod(0o755)
    identity_policy.write_text(json.dumps({"maintainer_uid": os.geteuid()}))
    maintenance = load_candidate(monkeypatch, "harness.code.maintenance", "code/maintenance.py")
    monkeypatch.setattr(maintenance, "require_role", lambda *args: None)
    policy = load_candidate(
        monkeypatch, "harness.core._maintenance_candidate_policy", "core/tool_policy.py"
    )
    profile = harness.profile("saebase")
    profile.mkdir(parents=True)
    (profile / "tool-policy.json").write_text(
        json.dumps(
            {
                "approved_roots": ["tasks", "tests", "saebase"],
                "approved_files": [],
                "protected_roots": ["saebase/runtime", "atlas"],
            }
        )
    )
    context = SimpleNamespace(
        harness=harness, project=SimpleNamespace(root=project, project_id="saebase")
    )
    monkeypatch.setattr(policy, "bind", lambda root: context)
    monkeypatch.delenv("OPENRESEARCH_ACTIVE_AGENT_FLOW", raising=False)
    monkeypatch.setenv("RH_MAINTENANCE_TASK", "test-maintenance")
    return maintenance, policy, harness, project


def enable(setup, paths):
    maintenance, _, harness, _ = setup
    return maintenance.set_permission(
        harness,
        "saebase",
        "test-maintenance",
        enabled=True,
        human_instruction="Synthetic test human instruction; no production permission",
        paths=[str(path) for path in paths],
        ttl=60,
    )


def gate(setup, path):
    _, policy, _, project = setup
    return policy.path_gate(project, "write_file", {"path": str(path)})


def test_default_closed_and_scoped_runtime_write(setup_candidate):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    assert gate(setup, target) == "Protected governance or Runtime target"
    enable(setup, [target])
    assert gate(setup, target) is None
    assert gate(setup, target.with_name("audit.py")) is not None


def test_central_write_requires_its_own_scope(setup_candidate):
    setup = setup_candidate
    target = setup[2].root / "src/harness/core/tool_policy.py"
    assert gate(setup, target) == "Mutation escapes the project root"
    enable(setup, [target])
    assert gate(setup, target) is None
    assert gate(setup, target.with_name("security.py")) is not None


def test_revocation_takes_effect_immediately(setup_candidate):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    enable(setup, [target])
    setup[0].set_permission(
        setup[2],
        "saebase",
        "test-maintenance",
        enabled=False,
        human_instruction="Synthetic revocation",
    )
    assert gate(setup, target) is not None


def test_expiry_automatically_closes_permission(setup_candidate, monkeypatch):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    record = enable(setup, [target])
    monkeypatch.setattr(setup[0].time, "time", lambda: record["expires_at"])
    assert gate(setup, target) is not None
    assert setup[0].status(setup[2], "saebase", "test-maintenance")["enabled"] is False


def test_context_closes_after_failure(setup_candidate):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with setup[0].permission(
            setup[2],
            "saebase",
            "test-maintenance",
            human_instruction="Synthetic task approval",
            paths=[str(target)],
        ):
            assert gate(setup, target) is None
            raise RuntimeError("synthetic interruption")
    assert gate(setup, target) is not None


def test_research_identity_cannot_open_or_use_permission(setup_candidate, monkeypatch):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    enable(setup, [target])

    def denied(*args):
        raise PermissionError("Synthetic research identity")

    monkeypatch.setattr(setup[0], "require_role", denied)
    assert gate(setup, target) is not None
    with pytest.raises(PermissionError):
        enable(setup, [target])


def test_environment_selects_but_does_not_grant_permission(setup_candidate, monkeypatch):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    enable(setup, [target])
    monkeypatch.setenv("RH_MAINTENANCE_TASK", "unapproved-task")
    assert gate(setup, target) is not None


@pytest.mark.parametrize("kind", ["world_writable", "wrong_uid", "wrong_project", "wrong_task"])
def test_invalid_receipt_fails_closed(setup_candidate, kind):
    setup = setup_candidate
    target = setup[3] / "saebase/runtime/train.py"
    record = enable(setup, [target])
    location = setup[0]._location(setup[2], "saebase", "test-maintenance")
    if kind == "world_writable":
        location.chmod(0o666)
    else:
        field = {"wrong_uid": "uid", "wrong_project": "project_id", "wrong_task": "task_id"}[kind]
        record[field] = os.geteuid() + 1 if field == "uid" else "wrong"
        location.write_text(json.dumps(record))
    assert gate(setup, target) is not None


@pytest.mark.parametrize("relative", [".workspace/state", ".git/config", "atlas/index.html"])
def test_authorization_state_and_publication_remain_excluded(setup_candidate, relative):
    setup = setup_candidate
    with pytest.raises(ValueError):
        enable(setup, [setup[2].root / relative])


def test_empty_instruction_and_scope_are_rejected(setup_candidate):
    setup = setup_candidate
    with pytest.raises(ValueError):
        enable(setup, [])
    with pytest.raises(ValueError):
        setup[0].set_permission(
            setup[2],
            "saebase",
            "test-maintenance",
            enabled=True,
            human_instruction="",
            paths=[str(setup[3] / "saebase/runtime")],
        )


def test_active_task_is_selected_without_environment_configuration(setup_candidate, monkeypatch):
    setup = setup_candidate
    monkeypatch.delenv("RH_MAINTENANCE_TASK")
    target = setup[3] / "saebase/runtime/train.py"
    assert gate(setup, target) is not None
    enable(setup, [target])
    assert gate(setup, target) is None


def test_directory_scope_rejects_siblings_and_symlinks(setup_candidate):
    setup = setup_candidate
    scope = setup[3] / "saebase/runtime"
    scope.mkdir(parents=True)
    enable(setup, [scope])
    assert gate(setup, scope / "train.py") is None
    outside = setup[2].root / "src/other.py"
    (scope / "alias.py").symlink_to(outside)
    assert gate(setup, scope / "alias.py") is not None
    assert gate(setup, setup[3] / "atlas/index.html") is not None


def test_cli_can_enable_report_and_disable(setup_candidate, monkeypatch, capsys):
    import harness.code

    setup = setup_candidate
    cli = load_candidate(monkeypatch, "harness._candidate_maintenance_cli", "cli.py")
    monkeypatch.setattr(cli, "load_harness", lambda root: setup[2])
    monkeypatch.setattr(setup[2], "project", lambda project_id: object(), raising=False)
    monkeypatch.setattr(harness.code, "maintenance", setup[0], raising=False)
    prefix = ["--harness-root", str(setup[2].root), "--project", "saebase", "maintenance"]
    target = setup[3] / "saebase/runtime/train.py"
    assert (
        cli.main(
            prefix
            + [
                "enable",
                "--task",
                "test-maintenance",
                "--path",
                str(target),
                "--instruction",
                "Synthetic explicit permission",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["enabled"] is True
    assert cli.main(prefix + ["status", "--task", "test-maintenance"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is True
    assert gate(setup, target) is None
    assert (
        cli.main(
            prefix
            + ["disable", "--task", "test-maintenance", "--instruction", "Synthetic completion"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["enabled"] is False
    assert gate(setup, target) is not None


def test_intake_requires_maintainer_for_switch_commands(setup_candidate, monkeypatch):
    from harness.core import access

    setup = setup_candidate
    intake = load_candidate(
        monkeypatch, "harness.core._candidate_maintenance_intake", "core/intake.py"
    )
    context = SimpleNamespace(harness=setup[2], project=SimpleNamespace(project_id="saebase"))
    monkeypatch.setattr(intake, "bind", lambda root: context)
    command = (
        f"rh --harness-root {setup[2].root} --project saebase "
        "maintenance status --task test-maintenance"
    )
    monkeypatch.setattr(access, "require_role", lambda *args: None)
    assert intake.inspection(setup[3], command) == (True, None)

    def denied(*args):
        raise PermissionError("Synthetic research identity")

    monkeypatch.setattr(access, "require_role", denied)
    assert intake.inspection(setup[3], command) == (True, "Synthetic research identity")


def test_source_write_permission_does_not_authorize_arbitrary_commands(setup_candidate):
    setup = setup_candidate
    enable(setup, [setup[3] / "saebase/runtime"])
    assert setup[1].shell_gate(setup[3], "python train.py") is not None


def test_group_writable_storage_cannot_hold_permissions(setup_candidate):
    setup = setup_candidate
    (setup[2].root / ".workspace").chmod(0o775)
    with pytest.raises(PermissionError, match="protected"):
        enable(setup, [setup[3] / "saebase/runtime"])
