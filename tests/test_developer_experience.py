"""Offline onboarding, documentation and restored GitHub community checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.check_documentation import ROOT, missing_links, validate


def test_repository_docs_have_valid_relative_navigation():
    count, broken = validate()
    assert count >= 35
    assert not broken
    paths = {
        "docs/first-experiment.md",
        "docs/troubleshooting.md",
        "docs/contributor-guide.md",
        "docs/maintainer-workflow.md",
        ".github/CODEOWNERS",
    }
    assert all((ROOT / name).is_file() for name in paths)


def test_relative_link_checker_detects_bad_paths(tmp_path):
    guide = tmp_path / "README.md"
    guide.write_text("[tutorial](docs/missing.md)\n", encoding="utf-8")
    assert missing_links(guide, tmp_path) == [
        "README.md:1: docs/missing.md"
    ]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "missing.md").write_text("# Present\n", encoding="utf-8")
    assert missing_links(guide, tmp_path) == []


def test_issue_forms_and_code_owners_are_present():
    forms = ROOT / ".github" / "ISSUE_TEMPLATE"
    expected = {"bug_report.yml", "feature_request.yml", "model_sae_support.yml"}
    for name in expected:
        obj = yaml.safe_load((forms / name).read_text(encoding="utf-8"))
        assert obj["name"] and obj["description"] and obj["body"], name
        ids = [item["id"] for item in obj["body"] if "id" in item]
        assert len(ids) == len(set(ids)), name
    options = yaml.safe_load((forms / "config.yml").read_text(encoding="utf-8"))
    assert options["blank_issues_enabled"] is True
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "* @AtheavenMa" in codeowners
    assert "/saebase/extensions/ @AtheavenMa" in codeowners


def _run_first_experiment(tmp_path, *arguments):
    result = subprocess.run(
        [
            sys.executable,
            "examples/19_first_experiment_offline.py",
            "--output",
            str(tmp_path / "example-output"),
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_first_experiment_dry_run_does_not_write_cache(tmp_path):
    result = _run_first_experiment(tmp_path, "--dry-run")
    assert result["mode"] == "dry-run"
    assert result["model_sae_pairs"] == 2
    assert result["independent_prompt_groups"] == 8
    assert not (tmp_path / "example-output").exists()


def test_first_experiment_writes_persistent_metrics_and_audit(tmp_path):
    first = _run_first_experiment(tmp_path)
    assert first["mode"] == "completed"
    assert first["data"] == "synthetic"
    assert first["model_sae_pairs"] == 2
    assert first["independent_prompt_groups"] == 8
    assert first["metric_rows"] == first["completed_metric_rows"]
    assert first["reliability_status"] == "completed"
    root = Path(first["output_dir"])
    assert root.is_relative_to(tmp_path)
    assert (root / "metrics.jsonl").is_file()
    assert (root / "metrics_manifest.json").is_file()
    assert (root / "reliability_manifest.json").is_file()
    assert (root / "summary.csv").is_file()
    assert (root / "reliability_summary.csv").is_file()
    assert (root / "report.html").is_file()
    assert "SAEBase Benchmark" in (root / "report.html").read_text(encoding="utf-8")
    again = _run_first_experiment(tmp_path)
    assert again["run_id"] == first["run_id"]
    assert again["metric_rows"] == first["metric_rows"]
