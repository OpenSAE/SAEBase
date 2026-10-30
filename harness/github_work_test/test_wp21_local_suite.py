"""Preparation/evidence gate tests, never a substitute for WP21 model experiments.

All artifacts made here are explicitly synthetic fixtures. The forged-claim
negative test asks the gate to reject configuration-only output; no test loads,
mocks, or reports a fake model as a genuine checkpoint execution.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "harness/tests/wp21/tools"


def _load_tools():
    """Load script modules without leaving generic names in pytest's imports."""
    modules = {}
    previous = {}
    try:
        for name in ("common", "manage", "record", "preflight", "audit_evidence", "behavior", "review"):
            previous[name] = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location("_wp21_suite_test_" + name, TOOLS / (name + ".py"))
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return SimpleNamespace(**modules)


T = _load_tools()


def _json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


@pytest.fixture
def spec_tree(tmp_path):
    """A small copy of actual public specification inputs, not invented cases."""
    root = tmp_path / "spec"
    spec = T.common.load_spec()
    paths = {"harness/tests/wp21/suite_lock.json"}
    paths.update(T.common.read_json(ROOT / "harness/tests/wp21/suite_lock.json")["files"])
    for case in spec["cases"]:
        paths.add("harness/tests/wp21/cases/" + case["id"] + ".md")
        paths.update(case["context_paths"])
    for relative in sorted(paths):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for case in spec["cases"]:
        if case.get("seed_fixture"):
            relative = "harness/tests/wp21/fixtures/" + case["seed_fixture"]
            shutil.copytree(ROOT / relative, root / relative, dirs_exist_ok=True)
    assert T.manage.check_spec(root)["status"] == "SPEC_VALID"
    return root


@pytest.fixture
def scoped_workspace(tmp_path):
    """Use actual R01 scope with a tiny disposable candidate/control pair."""
    workspace, control = tmp_path / "candidate", tmp_path / "control"
    case = T.common.get_case("R01")
    for relative in (
        "harness/tests/wp21/cases.json",
        "harness/contracts/catalog.json",
        "harness/contracts/REAL_WORKFLOW_PROTOCOL.md",
    ):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    allowed = "tests/wp21/R01/test_candidate.py"
    assert T.common.allowed_source(allowed, case), "R01 must expose its declared test path"
    _text(workspace / allowed, "# original synthetic test fixture\n")
    (workspace / "experiment").mkdir()
    evidence_rel = "harness/tests/wp21/runs/pytest_fixture/R01"
    evidence = workspace / evidence_rel
    evidence.mkdir(parents=True)
    before = T.common.snapshot(workspace)
    assignment = {
        "schema": "saebase.wp21.assignment.v1", "case_id": "R01", "run_id": "pytest_fixture",
        "base_commit": T.common.BASE_COMMIT, "case_sha256": T.common.canonical_hash(case),
        "spec_sha256": T.common.sha256(workspace / "harness/tests/wp21/cases.json"),
        "catalog_sha256": T.common.sha256(workspace / "harness/contracts/catalog.json"),
        "protocol_sha256": T.common.sha256(workspace / "harness/contracts/REAL_WORKFLOW_PROTOCOL.md"),
        "source_before_sha256": T.common.canonical_hash(before),
        "state": "prepared_not_assigned", "assignment_ref": None,
        "allowed_paths": case["allowed_paths"], "budget": case["budget"],
        "evidence_root": evidence_rel, "runtime_output_root": "experiment",
    }
    _json(control / "source_before.json", before)
    _json(control / "assignment.json", assignment)
    _json(workspace / ".cursor/wp21_assignment.json", assignment)
    return SimpleNamespace(workspace=workspace, control=control, case=case, evidence=evidence,
                           allowed=allowed, assignment=assignment)




def _assign_for_synthetic_gate_test(f):
    """Synthetic permission fixture, never a real user assignment."""
    assignment=dict(f.assignment)
    assignment["state"]="assigned"
    assignment["assignment_ref"]="SYNTHETIC_GATE_TEST_ONLY"
    _json(f.control / "assignment.json", assignment)
    _json(f.workspace / ".cursor/wp21_assignment.json", assignment)

def test_actual_public_cases_are_complete_and_explicitly_not_executed():
    result = T.manage.check_spec()
    assert result["status"] == "SPEC_VALID"
    assert result["cases"] == 20
    assert result["seeded_cases"] >= 4
    assert result["runtime_results"] == "NOT_RUN"
    assert result["base_commit"] == T.common.BASE_COMMIT


def test_duplicate_case_identity_is_rejected(spec_tree):
    path = spec_tree / "harness/tests/wp21/cases.json"
    spec = T.common.read_json(path)
    spec["cases"][1]["id"] = spec["cases"][0]["id"]
    _json(path, spec)
    with pytest.raises(ValueError, match="ordered and unique|frozen exam content changed"):
        T.manage.check_spec(spec_tree)


@pytest.mark.parametrize("mutation", ["content", "checksum", "protected_destination"])
def test_seed_integrity_rejects_corruption_and_out_of_scope_installation(spec_tree, mutation):
    case = next(c for c in T.common.load_spec(spec_tree)["cases"] if c.get("seed_fixture"))
    seed = spec_tree / "harness/tests/wp21/fixtures" / case["seed_fixture"]
    path = seed / "manifest.json"
    manifest = T.common.read_json(path)
    first = manifest["files"][0]
    if mutation == "content":
        with (seed / first["source"]).open("ab") as file:
            file.write(b"\n# altered instructor fixture\n")
    elif mutation == "checksum":
        first["sha256"] = "0" * 64
        _json(path, manifest)
    else:
        first["destination"] = "saebase/runtime/unauthorized.py"
        _json(path, manifest)
    with pytest.raises(ValueError, match="checksum mismatch|outside case scope|frozen exam content changed"):
        T.manage.check_spec(spec_tree)


@pytest.mark.parametrize("path", ["../escape", "a/../escape", "/tmp/escape", "a//b", "a/./b", "C:/escape", "C:\\escape", "a\\b", "a\x00b", " spaced", "a/*"])
def test_noncanonical_source_and_evidence_paths_are_rejected(path):
    with pytest.raises(ValueError):
        T.common.relative_name(path)


def test_containment_rejects_external_and_internal_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _text(root / "real.txt", "inside\n")
    _text(tmp_path / "outside.txt", "outside\n")
    (root / "inside_link").symlink_to(root / "real.txt")
    (root / "outside_link").symlink_to(tmp_path / "outside.txt")
    (root / "linked_directory").symlink_to(tmp_path, target_is_directory=True)
    for name in ["inside_link", "outside_link", "linked_directory/outside.txt"]:
        with pytest.raises(ValueError, match="symlink"):
            T.common.contained(root, name, exists=True)
    assert T.common.contained(root, "real.txt", exists=True).read_text() == "inside\n"


@pytest.mark.parametrize("payload", ['{"metric": NaN}', '{"metric": Infinity}', '{"metric": -Infinity}', '{"metric": 1e999}', '{"rows": [{"metric": 1e999}]}'])
def test_nonfinite_metrics_cannot_hide_in_serialized_evidence(tmp_path, payload):
    path = tmp_path / "invalid_metric.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        T.common.read_json(path)


def test_authorized_source_edit_and_deletion_have_observed_diff(scoped_workspace):
    f = scoped_workspace
    _text(f.workspace / f.allowed, "# changed synthetic test fixture\n")
    result = T.manage.scope_audit(f.workspace, f.control)
    assert result["status"] == "SCOPE_VALID"
    assert [(x["path"], x["operation"]) for x in result["changes"]] == [(f.allowed, "modify")]
    (f.workspace / f.allowed).unlink()
    result = T.manage.scope_audit(f.workspace, f.control)
    assert result["status"] == "SCOPE_VALID"
    assert [(x["path"], x["operation"]) for x in result["changes"]] == [(f.allowed, "delete")]


@pytest.mark.parametrize("path", ["models/unrelated.py", "saebase/runtime/unauthorized.py", "harness/contracts/unauthorized.py", ".cursor/unauthorized.json", ".github/workflows/unauthorized.yml", "doc/knowledge/unauthorized.md", "experiments/hidden_source.py"])
def test_unrelated_and_governance_source_changes_are_scope_violations(scoped_workspace, path):
    f = scoped_workspace
    _text(f.workspace / path, "unauthorized synthetic fixture\n")
    result = T.manage.scope_audit(f.workspace, f.control)
    assert result["status"] == "SCOPE_VIOLATION"
    assert path in result["violations"]


def test_candidate_cannot_promote_its_own_assignment(scoped_workspace):
    f = scoped_workspace
    path = f.workspace / ".cursor/wp21_assignment.json"
    record = T.common.read_json(path)
    record["state"] = "assigned"
    record["assignment_ref"] = "invented approval in deliberately invalid fixture"
    _json(path, record)
    result = T.manage.scope_audit(f.workspace, f.control)
    assert result["status"] == "SCOPE_VIOLATION"
    assert ".cursor/wp21_assignment.json" in result["violations"]


def test_corrupt_trusted_baseline_is_rejected(scoped_workspace):
    f = scoped_workspace
    path = f.control / "source_before.json"
    before = T.common.read_json(path)
    before[f.allowed] = "0" * 64
    _json(path, before)
    with pytest.raises(ValueError, match="snapshot is corrupt"):
        T.manage.scope_audit(f.workspace, f.control)


def test_prepare_one_actual_case_does_not_assign_or_execute(tmp_path):
    destination = tmp_path / "dispatch"
    result = T.manage.prepare(destination, ["R01"], "pytest_prepare", None)
    assert result["status"] == "PREPARED"
    assert result["models_executed"] == 0
    assert len(result["cases"]) == 1
    candidate = destination / "candidates/R01"
    assignment = T.common.read_json(destination / "control/R01/assignment.json")
    assert assignment["state"] == "prepared_not_assigned"
    assert assignment["assignment_ref"] is None
    assert not list((candidate / "experiment").iterdir())
    evidence = candidate / assignment["evidence_root"]
    report = T.common.read_json(evidence / "evidence.json")
    assert report["status"] == "NOT_RUN"
    assert report["evidence_kind"] == "unexecuted"
    assert report["command_refs"] == []
    assert T.common.read_json(evidence / "contract_receipt.json")["schema"] == "saebase.wp21.contract_receipt.v1"
    assert T.manage.scope_audit(candidate, destination / "control/R01")["status"] == "SCOPE_VALID"
    before = T.common.sha256(destination / "dispatch.json")
    with pytest.raises(ValueError, match="NEW directory"):
        T.manage.prepare(destination, ["R01"], "pytest_prepare", None)
    assert T.common.sha256(destination / "dispatch.json") == before


def test_recorder_preserves_actual_nonzero_exit_and_both_logs(tmp_path):
    output = tmp_path / "command.json"
    code = "import sys; print('actual stdout'); print('actual stderr', file=sys.stderr); sys.exit(7)"
    result = T.record.record([sys.executable, "-c", code], tmp_path, output, timeout=15)
    assert result["exit_code"] == 7
    assert result["timed_out"] is False
    assert result["execution_claim"] == "process_only"
    assert result["hub_offline_requested"] is True
    assert "offline" not in result, "Hub offline flags do not establish OS network isolation"
    for name, expected in [("stdout", b"actual stdout\n"), ("stderr", b"actual stderr\n")]:
        path = output.parent / result[name]["path"]
        assert path.read_bytes() == expected
        assert result[name]["sha256"] == T.common.sha256(path)
        assert result[name]["size_bytes"] == len(expected)
    assert T.common.read_json(output) == result


def test_recorder_does_not_evaluate_shell_syntax_and_output_is_immutable(tmp_path):
    output = tmp_path / "command.json"
    literal = "$(touch must_not_exist); echo not_a_shell"
    result = T.record.record([sys.executable, "-c", "import sys; print(sys.argv[1])", literal], tmp_path, output, timeout=15)
    assert result["exit_code"] == 0
    assert (tmp_path / result["stdout"]["path"]).read_text().strip() == literal
    assert not (tmp_path / "must_not_exist").exists()
    old_hashes = {p.name: T.common.sha256(p) for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="immutable"):
        T.record.record([sys.executable, "-c", "open('must_not_exist','w').write('bad')"], tmp_path, output, timeout=15)
    assert not (tmp_path / "must_not_exist").exists()
    assert old_hashes == {p.name: T.common.sha256(p) for p in tmp_path.iterdir()}


def test_recorder_timeout_and_launch_failure_are_not_success(tmp_path):
    timed = T.record.record([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(20)"], tmp_path, tmp_path / "timeout.json", timeout=1)
    assert timed["timed_out"] is True and timed["exit_code"] == 124
    assert (tmp_path / timed["stdout"]["path"]).is_file()  # startup may exceed a 1s timeout on loaded hosts
    missing = T.record.record([str(tmp_path / "does_not_exist")], tmp_path, tmp_path / "launch.json", timeout=15)
    assert missing["exit_code"] == 127 and missing["timed_out"] is False
    assert b"process launch failed" in (tmp_path / missing["stderr"]["path"]).read_bytes()


def test_freezing_synthetic_asset_bytes_does_not_claim_a_real_checkpoint(tmp_path):
    asset = tmp_path / "synthetic_asset"
    _text(asset / "weights.bin", "synthetic test bytes, not trained weights\n")
    output = tmp_path / "assets.json"
    result = T.preflight.freeze_assets([("model", asset)], output)
    lock = T.common.read_json(output)
    assert result["status"] == "FILES_HASHED_NOT_RUNTIME_VERIFIED"
    assert result["file_count"] == 1
    assert lock["files"][0]["sha256"] == T.common.sha256(asset / "weights.bin")
    original = output.read_bytes()
    _text(asset / "weights.bin", "changed synthetic test bytes\n")
    with pytest.raises(FileExistsError):
        T.preflight.freeze_assets([("model", asset)], output)
    assert output.read_bytes() == original


def test_prerequisite_inspection_does_not_execute_a_model(tmp_path):
    model = tmp_path / "empty_model_snapshot"
    sae = tmp_path / "empty_sae_snapshot"
    model.mkdir()
    sae.mkdir()
    result = T.preflight.preflight("R01", model_snapshot=model, sae_snapshot=sae)
    assert result["executed_model"] is False
    assert result["downloaded_weights"] is False
    assert result["status"] in {"ENV_BLOCKED", "PREREQUISITES_PRESENT_UNVERIFIED"}


@pytest.mark.parametrize("kind,status", [("synthetic", "SELF_TESTED"), ("configuration", "SELF_TESTED"), ("unexecuted", "NOT_RUN"), ("real_checkpoint", "configured_only")])
def test_nonexecution_and_synthetic_claims_cannot_enter_real_replay(scoped_workspace, kind, status):
    f = scoped_workspace
    _assign_for_synthetic_gate_test(f)
    _json(f.evidence / "evidence.json", {"status": status, "evidence_kind": kind})
    with pytest.raises(ValueError, match="SELF_TESTED real_checkpoint"):
        T.audit_evidence.audit(f.workspace, f.control)


def test_configured_only_detection_covers_nested_outputs():
    assert T.audit_evidence.contains_configured_only({"rows": [{"metadata": {"status": "configured_only"}}]})
    assert T.audit_evidence.contains_configured_only({"output": [{"status": "dry_run"}]})
    assert not T.audit_evidence.contains_configured_only({"rows": [{"status": "completed", "metric": 1.0}]})


def test_forged_real_label_cannot_hide_actual_configured_only_output(scoped_workspace):
    """Deliberately invalid synthetic evidence; rejection is the only valid result."""
    f = scoped_workspace
    _assign_for_synthetic_gate_test(f)
    config = {"purpose": "negative gate fixture; no model was executed"}
    config_rel, result_rel = "experiment/negative/config.json", "experiment/negative/result.json"
    _json(f.workspace / config_rel, config)
    _json(f.workspace / result_rel, {"rows": [{"status": "configured_only", "synthetic_fixture": True}]})
    collected = T.audit_evidence.collect(f.workspace, f.control)
    assert collected["status"] == "COLLECTED_NOT_GRADED"
    _json(f.evidence / "evidence.json", {
        "schema": "saebase.wp21.evidence.v1", "case_id": "R01", "run_id": "pytest_fixture",
        "status": "SELF_TESTED", "evidence_kind": "real_checkpoint",  # intentionally false claim
        "base_commit": T.common.BASE_COMMIT,
        "source_snapshot_sha256": collected["source_snapshot_sha256"],
        "input_identity": {"sha256": "1" * 64, "synthetic_fixture": True},
        "model_identity": {"revision": "2" * 40, "files_sha256": "3" * 64, "synthetic_fixture": True},
        "sae_identity": {"synthetic_fixture": True}, "environment": {"synthetic_fixture": True},
        "resolved_config_path": config_rel, "canonical_config_sha256": T.common.canonical_hash(config),
        "runtime_config_sha256": None, "executed_result_paths": [result_rel],
    })
    with pytest.raises(ValueError, match="configured_only/dry-run"):
        T.audit_evidence.audit(f.workspace, f.control)


def test_artifact_collection_rejects_symlink_escape(scoped_workspace, tmp_path):
    f = scoped_workspace
    external = tmp_path / "external.txt"
    _text(external, "must remain outside evidence\n")
    (f.workspace / "experiment/escaped.txt").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        T.audit_evidence.collect(f.workspace, f.control)
    assert external.read_text() == "must remain outside evidence\n"


@pytest.mark.parametrize("mutation", ["changed_bytes", "extra_file"])
def test_artifact_manifest_cannot_survive_silent_changes(scoped_workspace, mutation):
    f = scoped_workspace
    _assign_for_synthetic_gate_test(f)
    result_rel = "experiment/synthetic/result.json"
    _json(f.workspace / result_rel, {"synthetic_fixture": True, "value": 1})
    collected = T.audit_evidence.collect(f.workspace, f.control)
    # Deliberately invalid claim, constructed only to reach the artifact gate.
    _json(f.evidence / "evidence.json", {
        "schema": "saebase.wp21.evidence.v1", "case_id": "R01", "run_id": "pytest_fixture",
        "status": "SELF_TESTED", "evidence_kind": "real_checkpoint",
        "base_commit": T.common.BASE_COMMIT,
        "source_snapshot_sha256": collected["source_snapshot_sha256"],
    })
    if mutation == "changed_bytes":
        _json(f.workspace / result_rel, {"synthetic_fixture": True, "value": 999})
    else:
        _text(f.workspace / "experiment/synthetic/extra.txt", "unmanifested output\n")
    with pytest.raises(ValueError, match="artifact manifest mismatch"):
        T.audit_evidence.audit(f.workspace, f.control)


@pytest.mark.parametrize("status", ["BLOCKED", "ENV_BLOCKED"])
def test_blocked_report_preserves_absence_of_a_grade(scoped_workspace, status):
    f = scoped_workspace
    _json(f.evidence / "evidence.json", {"status": status, "unresolved": ["blocked synthetic gate test"]})
    result = T.audit_evidence.audit(f.workspace, f.control)
    assert result["status"] == status
    assert result["grade"] is None


@pytest.fixture
def hook_workspace(scoped_workspace, monkeypatch):
    """Run actual hooks on a tiny worktree; no real approval is created."""
    f = scoped_workspace
    for relative in [".cursor/policy.json", ".cursor/hooks/wp21_scope.py", ".cursor/hooks/guard_paths.py", ".cursor/hooks/guard_shell.py"]:
        target = f.workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    monkeypatch.setattr(sys, "path", list(sys.path))
    loaded = {}
    for name in ["guard_paths", "guard_shell"]:
        spec = importlib.util.spec_from_file_location("_wp21_hook_test_" + name, f.workspace / ".cursor/hooks" / (name + ".py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    f.paths, f.shell = loaded["guard_paths"], loaded["guard_shell"]
    return f


def _activate_synthetic_assignment(f):
    """Test-only dispatcher fixture, never approval or a real agent assignment."""
    assignment = dict(f.assignment)
    assignment["state"] = "assigned"
    assignment["assignment_ref"] = "SYNTHETIC_PERMISSION_TEST_ONLY"
    _json(f.workspace / ".cursor/wp21_assignment.json", assignment)


@pytest.mark.parametrize("target", ["harness/tests/wp21/fixtures/new.py", "experiment/forbidden.json", "tests/wp21/R01/test_new.py"])
def test_unassigned_preparation_does_not_grant_write_authority(hook_workspace, target):
    f = hook_workspace
    assert f.paths.gate("edit_file", {"path": target}) is not None


def test_assigned_profile_allows_own_sources_evidence_and_runtime_output(hook_workspace):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    for target in [f.allowed, f.assignment["evidence_root"] + "/actual_log.json", "experiment/owned_run/metrics.json"]:
        assert f.paths.gate("edit_file", {"path": target}) is None, target


@pytest.mark.parametrize("target", ["tests/wp21/R02/test_other.py", "harness/tests/wp21/cases.json", "harness/tests/wp21/fixtures/new.py", "saebase/runtime/train.py", ".cursor/wp21_assignment.json", ".github/workflows/new.yml"])
def test_assignment_cannot_write_another_case_or_governance(hook_workspace, target):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    assert f.paths.gate("edit_file", {"path": target}) is not None


def test_frozen_protocol_tampering_revokes_hook_assignment(hook_workspace):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    with (f.workspace / "harness/contracts/REAL_WORKFLOW_PROTOCOL.md").open("a") as file:
        file.write("\nSynthetic unauthorized governance edit.\n")
    assert f.paths.gate("edit_file", {"path": f.allowed}) is not None


def test_missing_profile_with_a_present_assignment_fails_closed(hook_workspace):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    (f.workspace / ".cursor/hooks/wp21_scope.py").unlink()
    assert f.paths.gate("edit_file", {"path": f.allowed}) is not None
    assert f.shell.gate("python tests/wp21/R01/driver.py") is not None


def test_generic_acceptance_protection_and_legacy_wp18_sandbox(hook_workspace):
    f = hook_workspace
    (f.workspace / ".cursor/wp21_assignment.json").unlink()
    assert f.paths.gate("edit_file", {"path": "harness/tests/new_topic/acceptance/result.json"}) is not None
    assert f.shell.gate("mkdir -p harness/tests/new_topic/acceptance/new_run") is not None
    assert f.paths.gate("edit_file", {"path": "harness/tests/wp18/runs/new_run/H01/implementation.py"}) is None
    assert f.shell.gate("python -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/new_run/H01") is None


def test_assigned_driver_and_module_commands_are_allowed(hook_workspace):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    for command in ["python -m saebase.runtime.train task=capture task.execute=true", "python -m saebase.runtime.train --multirun task=capture seed=1,2", "python tests/wp21/R01/driver.py", "python -m pytest -p no:cacheprovider -q tests/wp21/R01"]:
        assert f.shell.gate(command) is None, command


@pytest.mark.parametrize("command", [
    "python -c 'print(1)'",
    "python tests/wp21/R02/driver.py",
    "python -m saebase.runtime.train paths.root=/tmp/outside_wp21",
    "python -m saebase.runtime.train hydra.run.dir=/tmp/outside_wp21",
    "python -m saebase.runtime.train experiment.overwrite=true",
    "python -m saebase.runtime.train --config-dir /tmp/other_config",
    "python -m saebase.runtime.train --config-path=/tmp/other_config",
    "python -m saebase.runtime.train paths={root:/tmp/other_output}",
    "python -m pytest -p no:cacheprovider tests/wp21/R01 /tmp/unrelated.py",
    "python -m pytest -p no:cacheprovider tests/wp21/R01 -p unrelated_plugin",
    "python -m pytest -p no:cacheprovider tests/wp21/R01 ../unrelated.py",
])
def test_command_profile_rejects_unscoped_execution_and_output_overrides(hook_workspace, command):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    assert f.shell.gate(command) is not None


def test_recorder_wrapper_does_not_expand_underlying_command_scope(hook_workspace):
    f = hook_workspace
    _activate_synthetic_assignment(f)
    prefix = f"python harness/tests/wp21/tools/record.py --cwd {f.workspace} --output {f.assignment['evidence_root']}/commands/one.json -- "
    assert f.shell.gate(prefix + "python tests/wp21/R01/driver.py") is None
    assert f.shell.gate(prefix + "python -c 'print(1)'") is not None
    assert f.shell.gate(prefix + "python tests/wp21/R02/driver.py") is not None
