"""Release integrity and scope of v0.4.0, not GPU scientific replay."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def test_release_version_and_notes_are_consistent():
    version = tomllib.loads((BASE / "pyproject.toml").read_text())["project"]["version"]
    assert tuple(int(x) for x in version.split(".")) >= (0, 4, 0)
    module = ast.parse((BASE / "saebase" / "__init__.py").read_text())
    constants = [
        node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == "__version__" for name in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert constants == [version]
    assert re.search(r"(?m)^## 0\.4\.0\b", (BASE / "CHANGELOG.md").read_text())
    assert f"version: {version}" in (BASE / "CITATION.cff").read_text()
    assert (BASE / "harness/github_development_doc/releases/v0.4.0.md").exists()


def test_research_lab_report_distinguishes_experiment_scope_from_software():
    report = json.loads(
        (
            BASE / "harness/github_development_doc/research-lab/release-evidence-v0.4.0.json"
        ).read_text()
    )
    assert report["software_version"] == "0.4.0"
    assert report["historical_reference"]["prompt_groups_per_site"] == 64
    assert report["new_cohort"]["parent_groups"] == 500
    assert report["new_cohort"]["discovery"] + report["new_cohort"]["evaluation"] == 500
    assert report["new_cohort"]["license_reviewed"] is False
    gpu = report["gpu_executor_recorded"]
    sar = gpu["sar"]
    assert sar["groups_K16"] + sar["groups_K8"] + sar["groups_K4"] == 1500
    assert sar["groups_K2"] == 0
    assert sar["protocol"] == "decoder_mediated_residual_preserving_v1"
    assert (
        gpu["rss"]["clean_site_inputs"]
        + gpu["rss"]["selection_site_variants"]
        + gpu["rss"]["held_out_site_variants"]
        == 138000
    )
    assert gpu["rss"]["semantic_review"] is False
    assert gpu["full_steered_text_generation"] == "pending"
    assert gpu["raw_experiment_archive"] == "private_not_in_git"
    assert report["verification"]["full_gpu_raw_independent_verification"] == "not_completed"
    assert not any(
        (BASE / name).exists()
        for name in (
            "experiments/research-lab/7fb2be8f29a1-20261019T035727Z/private",
            "saebase/research_lab/data/raw_prompts.jsonl",
        )
    )


def test_readmes_and_workplan_refer_to_actual_preview():
    for name in ("README.md", "README_cn.md"):
        txt = (BASE / name).read_text(encoding="utf-8")
        assert "harness/github_development_doc/releases/v0.4.0.md" in txt
        assert "Research Lab" in txt
    plan = (
        BASE / "harness/github_work_plan/05_research_lab_cpu_closure_and_gpu_handoff.md"
    ).read_text()
    assert "harness/github_development_doc/research-lab/release-acceptance.md" in plan
    assert "K16" in plan and "RSS" in plan
