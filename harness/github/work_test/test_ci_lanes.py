"""Separate lightweight PR/main checks from explicit full release validation."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"

MANUAL = {
    "real-smoke.yml",
    "software-release.yml",
    "m3-real.yml",
    "m3-saebench-native.yml",
    "m4-real.yml",
    "m5-matrix.yml",
    "phase-a-cpu-probes.yml",
    "phase-a-real.yml",
    "phase-a-sae-probes.yml",
    "stage-bc-cpu-pairs.yml",
    "stage-bc-cpu-sae.yml",
    "stage-bc-gemma9b-recheck.yml",
    "stage-bc-source-head.yml",
    "saebase-atlas-snapshot.yml",
    "atlas-project-pages.yml",
    "research-lab-v04-release-dispatch.yml",
    "saebase-v050-release-dispatch.yml",
    "saebase-v060-release-dispatch.yml",
    "saebase-v065-release-dispatch.yml",
    "stage-d-evidence-release.yml",
    "archive-preparation.yml",
}


def workflow(name: str) -> dict:
    data = yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(data, dict), name
    assert isinstance(data.get("on"), dict), name
    assert isinstance(data.get("jobs"), dict), name
    return data


def test_pr_and_main_use_only_lightweight_ci():
    paths = sorted(WORKFLOWS.glob("*.yml"))
    auto_pr = {path.name for path in paths if "pull_request" in workflow(path.name)["on"]}
    assert auto_pr == {"ci.yml"}
    for name in MANUAL:
        assert set(workflow(name)["on"]) == {"workflow_dispatch"}, name
    research = workflow("research-regression.yml")
    assert set(research["on"]) == {"workflow_call", "workflow_dispatch"}
    ci = workflow("ci.yml")
    assert "push" in ci["on"] and ci["on"]["push"]["branches"] == ["main"]
    assert set(ci["jobs"]) == {
        "changes", "quality", "harness-targeted", "atlas-targeted",
        "product-targeted", "ci-required",
    }
    assert "analysis-install" not in ci["jobs"]
    assert "package" not in ci["jobs"]
    assert "offline-tests" not in ci["jobs"]
    assert "localization-native" not in ci["jobs"]
    assert ci["concurrency"]["cancel-in-progress"] == "true"


def test_manual_archive_workflow_never_pushes_or_publishes():
    flow = workflow("archive-preparation.yml")
    assert set(flow["on"]) == {"workflow_dispatch"}
    assert flow["permissions"]["contents"] == "read"
    assert "archive prepare" in (WORKFLOWS / "archive-preparation.yml").read_text()
    assert "actions/upload-artifact" in (WORKFLOWS / "archive-preparation.yml").read_text()


def test_lightweight_gate_is_fail_closed_and_scope_aware():
    jobs = workflow("ci.yml")["jobs"]
    selected = ("harness-targeted", "atlas-targeted", "product-targeted")
    for name in selected:
        assert jobs[name]["needs"] == "changes"
        assert "needs.changes.outputs." in jobs[name]["if"]
    gate = jobs["ci-required"]
    assert gate["if"] == "${{ always() }}"
    assert set(gate["needs"]) == {
        "changes", "quality", *selected,
    }
    script = gate["steps"][0]["run"]
    assert 'test "$QUALITY_RESULT" = success' in script
    assert 'test "$CHANGE_RESULT" = success' in script
    assert 'test "${!result_key}" = success' in script


def test_full_release_validation_has_exact_source_gate_and_all_full_jobs():
    flow = workflow("release-validation.yml")
    assert set(flow["on"]) == {"workflow_dispatch"}
    assert flow["on"]["workflow_dispatch"]["inputs"]["source_sha"]["required"] == "true"
    jobs = flow["jobs"]
    required = {
        "source-verify", "offline-tests", "analysis-install",
        "package", "localization-native", "quality",
        "hydra-composition", "harness-source", "pinned-research-regression",
    }
    assert required <= set(jobs)
    for name in required - {"source-verify"}:
        assert jobs[name]["needs"] == "source-verify", name
    assert "main" == jobs["source-verify"]["steps"][0]["with"]["ref"]
    assert "inputs.source_sha" in str(jobs["source-verify"]["steps"])
    for name in ("analysis-install", "package", "offline-tests", "quality"):
        checkouts = [
            step for step in jobs[name]["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert checkouts and checkouts[0]["with"]["ref"] == "${{ inputs.source_sha }}"
    assert "release-validation-required" in jobs
    assert "release-validation-required" not in workflow("ci.yml")["jobs"]


def test_release_candidate_refuses_lightweight_ci_evidence():
    source = (WORKFLOWS / "software-release.yml").read_text(encoding="utf-8")
    assert "actions/workflows/release-validation.yml/runs?" in source
    assert 'select(.event == "workflow_dispatch"' in source
    assert "actions/workflows/ci.yml/runs?" not in source
    assert '--name saebase-distributions' in source
    assert "PUBLISH-v$VERSION" in source
    from saebase.tools.release_candidate import REQUIRED_CI

    assert set(REQUIRED_CI) == {
        "offline-tests (3.11)", "analysis-install",
        "package", "localization-native", "quality",
    }


def test_every_pinned_research_job_uses_explicit_original_source_shas():
    jobs = workflow("research-regression.yml")["jobs"]
    assert set(jobs) == {"original-estimators", "synthetic-ground-truth"}
    expected = {
        "OpenSAE/SAE-Behavioral-Compositionality": "c253dca303437318c1f4f4937557ac9d22143b19",
        "OpenSAE/Redistribution-Sensitivity-Score": "0b321851e7b79564bf72656e1dbcaaca513048d8",
    }
    for job in jobs.values():
        sources = {
            step["with"]["repository"]: step["with"].get("ref")
            for step in job["steps"]
            if "with" in step and "repository" in step["with"]
        }
        assert sources == expected
        assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in sources.values())


def test_upstream_monitor_is_non_gating_and_historical_real_smoke_is_manual():
    monitor = workflow("upstream-compatibility.yml")
    assert set(monitor["on"]) == {"schedule", "workflow_dispatch"}
    for step in monitor["jobs"]["latest-original-estimators"]["steps"]:
        if "repository" in step.get("with", {}):
            assert "ref" not in step["with"]
    smoke = workflow("real-smoke.yml")
    assert set(smoke["on"]) == {"workflow_dispatch"}
    assert smoke["on"]["workflow_dispatch"]["inputs"]["suite"]["options"] == [
        "m0", "m1", "m2", "all",
    ]


def test_historical_m5_source_bound_release_keeps_its_separate_gate():
    old = workflow("m5-release.yml")
    assert set(old["on"]) == {"workflow_run"}
    assert old["on"]["workflow_run"]["workflows"] == [
        "M5 real two-family checkpoint matrix acceptance",
    ]
    gate = old["jobs"]["publish"]["if"]
    assert "workflow_run.event == 'push'" in gate
    assert "head_branch == 'main'" in gate
