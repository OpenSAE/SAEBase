"""Validate separation of automatic, pinned, and manual GitHub Actions lanes."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

MANUAL = {
    "real-smoke.yml",
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
}


def workflow(name: str) -> dict:
    data = yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(data, dict)
    assert isinstance(data.get("on"), dict), name
    assert isinstance(data.get("jobs"), dict), name
    return data


def test_only_essential_ci_and_pinned_research_run_on_pull_requests():
    paths = sorted(WORKFLOWS.glob("*.yml"))
    assert len(paths) == 17
    auto = {path.name for path in paths if "pull_request" in workflow(path.name)["on"]}
    assert auto == {"ci.yml", "research-regression.yml"}
    for name in MANUAL:
        assert set(workflow(name)["on"]) == {"workflow_dispatch"}, name


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


def test_latest_upstream_monitor_is_separate_and_non_gating():
    monitor = workflow("upstream-compatibility.yml")
    assert set(monitor["on"]) == {"schedule", "workflow_dispatch"}
    steps = monitor["jobs"]["latest-original-estimators"]["steps"]
    sources = [step["with"] for step in steps if "with" in step and "repository" in step["with"]]
    assert len(sources) == 2
    assert all("ref" not in source for source in sources)


def test_real_m0_m1_m2_smoke_requires_manual_selection():
    smoke = workflow("real-smoke.yml")
    assert set(smoke["on"]) == {"workflow_dispatch"}
    suites = smoke["on"]["workflow_dispatch"]["inputs"]["suite"]["options"]
    assert suites == ["m0", "m1", "m2", "all"]
    content = (WORKFLOWS / "real-smoke.yml").read_text(encoding="utf-8")
    for script in ("m0_real_smoke.sh", "m1_real_smoke.py", "m2_real_smoke.py"):
        assert script in content


def test_immutable_historical_release_guards_are_preserved():
    old = workflow("m5-release.yml")
    assert set(old["on"]) == {"workflow_run"}
    assert old["on"]["workflow_run"]["workflows"] == [
        "M5 real two-family checkpoint matrix acceptance"
    ]
    gate = old["jobs"]["publish"]["if"]
    assert "workflow_run.event == 'push'" in gate
    assert "head_branch == 'main'" in gate
    assert set(workflow("m5-matrix.yml")["on"]) == {"workflow_dispatch"}
    stage_d = workflow("stage-d-evidence-release.yml")["jobs"]["archive"]["if"]
    assert "head_repository.full_name == github.repository" in stage_d
