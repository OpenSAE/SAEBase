#!/usr/bin/env python3
"""Validate built SAEBase wheel/sdist contents without installing optional packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import tomllib
import zipfile


NEEDED_WHEEL = {
    "saebase/cli.py",
    "saebase/cli_compat.py",
    "saebase/model_plugins.py",
    "saebase/extensions/__init__.py",
    "saebase/extensions/contracts.py",
    "saebase/extensions/registry.py",
    "saebase/extensions/runtime.py",
    "saebase/extensions/metrics.py",
    "saebase/tasks/__init__.py",
    "saebase/tasks/plugins.py",
    "saebase/benchmarks/__init__.py",
    "saebase/benchmarks/runner.py",
    "saebase/benchmarks/planner.py",
    "saebase/benchmarks/results.py",
    "saebase/reliability/__init__.py",
    "saebase/reliability/runner.py",
    "saebase/reliability/ground_truth.py",
    "saebase/release/__init__.py",
    "saebase/release/schema.py",
    "saebase/release/planner.py",
    "saebase/release/runner.py",
    "saebase/release/gates.py",
    "models/gpt2small/gpt2small.py",
    "saebase/tools/m3_benchmark.py",
    "saebase/tools/m4_reliability.py",
    "saebase/tools/m4_ground_truth.py",
    "saebase/tools/m5_release_matrix.py",
    "saebase/tools/m5_release_gate.py",
    "conf/release/m5_matrix.json",
    "conf/release/m5_extended_gpt2.json",
    "conf/release/m5_asset_provenance.json",
    "saebase/__init__.py",
    "saebase/api/__init__.py",
    "saebase/api/features.py",
    "saebase/interventions/__init__.py",
    "saebase/interventions/engine.py",
    "saebase/interventions/specs.py",
    "saebase/interventions/operators.py",
    "saebase/interventions/results.py",
    "saebase/interventions/steering.py",
    "saebase/tools/m2_batch.py",
    "conf/interventions/pythia_feature_edits.json",
    "saebase/store/__init__.py",
    "saebase/store/identity.py",
    "saebase/store/shards.py",
    "saebase/store/collect.py",
    "saebase/integrations/sar.py",
    "saebase/integrations/rss.py",
    "saebase/sae/registry.py",
    "saebase/data/prompt_store.py",
    "models/plugins.py",
    "tasks/plugins.py",
    "saebase/tools/sae_prepare_prompts.py",
    "conf/__init__.py",
    "conf/config.yaml",
    "conf/model/pythia70m.yaml",
    "conf/sae/pythia70m_resid_sm.yaml",
    "conf/prompt_source/simple_copying.yaml",
    "conf/evaluation/activation_health.yaml",
    "conf/task/sae.yaml",
}
NEEDED_SDIST = NEEDED_WHEEL | {
    ".github/workflows/m5-matrix.yml",
    ".github/workflows/m5-release.yml",
    "harness/github_development_doc/M5_RELEASE_NOTES.md",
    "harness/github_development_doc/M5_RELEASE_MATRIX.md",
    "harness/github_work_test/test_m5_matrix.py",
    "harness/github_work_test/test_m5_gates.py",
    "pyproject.toml",
    "README.md",
    "README_cn.md",
    "LICENSE",
    "harness/github_development_doc/legal/NOTICE",
    ".github/CONTRIBUTING.md",
    ".github/SECURITY.md",
    "harness/github_development_doc/CHANGELOG.md",
    "CITATION.cff",
    "MANIFEST.in",
    ".github/workflows/ci.yml",
    "harness/github_development_doc/RELEASE_CHECKLIST.md",
    "harness/github_development_doc/PROVENANCE.md",
    "harness/github_development_doc/API.md",
    "harness/github_work_test/test_phase_a.py",
    "harness/github_work_test/test_phase_d.py",
    "harness/github_work_test/test_m1_store.py",
    "harness/github_work_test/test_m2_interventions.py",
    "harness/github_work_test/fixtures/m0_reference_public.json",
    "harness/github_development_doc/M1_ACTIVATION_STORE.md",
    "saebase/examples/activations/activation_store.py",
    ".github/workflows/real-smoke.yml",
    ".github/workflows/research-regression.yml",
    ".github/workflows/upstream-compatibility.yml",
    ".github/workflows/software-release.yml",
    "saebase/tools/release_candidate.py",
    "harness/github_development_doc/release-process.md",
    "harness/github_work_test/test_release_candidate.py",
    "harness/github_development_doc/extensions.md",
    "harness/github_work_test/test_external_extensions.py",
    "saebase/examples/extensions/external_providers.py",
    "harness/github_development_doc/first-experiment.md",
    "harness/github_development_doc/troubleshooting.md",
    "harness/github_development_doc/contributor-guide.md",
    "harness/github_development_doc/maintainer-workflow.md",
    "saebase/tools/check_documentation.py",
    "harness/github_work_test/test_developer_experience.py",
    "saebase/examples/quickstart/first_experiment.py",
    ".github/ISSUE_TEMPLATE/model_sae_support.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    "harness/github_development_doc/M2_BATCH_INTERVENTIONS.md",
    "saebase/tools/m2_real_smoke.py",
    "saebase/examples/interventions/batch_interventions.py",
}
BAD_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
BAD_TOP_LEVEL = {"experiment", "data", "dist", "build"}


def project_version_from_sdist(source: Path) -> str:
    """Read authoritative release version from the built source distribution."""
    with tarfile.open(source, "r:gz") as archive:
        files = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/pyproject.toml")
        ]
        if len(files) != 1:
            raise ValueError("Expected one pyproject.toml in source distribution")
        content = archive.extractfile(files[0])
        if content is None:
            raise ValueError("Cannot read source distribution project metadata")
        project = tomllib.loads(content.read().decode("utf-8"))["project"]
    if project["name"] != "opensae-saebase":
        raise ValueError("Source distribution project name mismatch")
    return str(project["version"])


def validate(dist_dir: Path) -> dict[str, object]:
    wheels = sorted(dist_dir.glob("opensae_saebase-*.whl"))
    sdists = sorted(dist_dir.glob("opensae_saebase-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"Expected one wheel and one sdist in {dist_dir}; got {len(wheels)}, {len(sdists)}"
        )
    version = project_version_from_sdist(sdists[0])
    if not wheels[0].name.startswith(f"opensae_saebase-{version}-"):
        raise AssertionError("Wheel filename/version does not match source archive")
    if sdists[0].name != f"opensae_saebase-{version}.tar.gz":
        raise AssertionError("Source archive filename/version mismatch")
    with zipfile.ZipFile(wheels[0]) as z:
        wheel_files = set(z.namelist())
        missing = sorted(NEEDED_WHEEL - wheel_files)
        if missing:
            raise AssertionError(f"Wheel missing {missing}")
        forbidden_wheel = sorted(
            p
            for p in wheel_files
            if p.startswith("conf/coverage/stage_d_evidence/")
            or p.startswith("harness/github_work_plan/")
            or p.endswith((".safetensors", ".pt", ".pth", ".ckpt"))
        )
        if forbidden_wheel:
            raise AssertionError(f"Non-runtime assets leaked into wheel: {forbidden_wheel[:5]}")
        entry_file = next(
            (p for p in wheel_files if p.endswith(".dist-info/entry_points.txt")), None
        )
        if entry_file is None:
            raise AssertionError("Wheel entry points are missing")
        entries = z.read(entry_file).decode("utf-8")
        expected_entries = {
            "saebase": "saebase.cli:main",
            "saebase-prepare-prompts": "saebase.cli:prepare_prompts",
            "saebase-check-env": "saebase.cli:check_env",
            "saebase-intervene": "saebase.cli:intervene",
            "saebase-benchmark": "saebase.cli:benchmark",
            "saebase-audit": "saebase.cli:audit",
            "saebase-ground-truth": "saebase.cli:ground_truth",
            "saebase-release-matrix": "saebase.cli:release_matrix",
            "saebase-release-gate": "saebase.cli:release_gate",
        }
        for name, target in expected_entries.items():
            if f"{name} = {target}" not in entries.splitlines():
                raise AssertionError(f"Missing or misdirected CLI entry point: {name}")
        metadata_file = next((p for p in wheel_files if p.endswith(".dist-info/METADATA")), None)
        if metadata_file is None:
            raise AssertionError("Wheel metadata missing")
        metadata = z.read(metadata_file).decode("utf-8")
        if "Name: opensae-saebase" not in metadata or f"Version: {version}" not in metadata:
            raise AssertionError("Distribution identity mismatch")
        if "License-Expression: Apache-2.0" not in metadata:
            raise AssertionError("Wheel license metadata missing")
        if len([p for p in wheel_files if p.startswith("conf/") and p.endswith(".yaml")]) < 100:
            raise AssertionError("Bundled Hydra configuration set is unexpectedly incomplete")
    with tarfile.open(sdists[0], "r:gz") as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            segments = Path(member.name).parts
            if len(segments) < 2:
                continue
            rest = segments[1:]
            if rest[:3] == ("conf", "coverage", "stage_d_evidence") or (
                rest and rest[0] == "work_plan"
            ):
                raise AssertionError(f"Archived experimental evidence in sdist: {member.name}")
            if any(part in BAD_PARTS for part in rest) or (rest and rest[0] in BAD_TOP_LEVEL):
                raise AssertionError(f"Unexpected generated/runtime file: {member.name}")
            if member.isfile():
                names.add("/".join(rest))
        missing = sorted(NEEDED_SDIST - names)
        if missing:
            raise AssertionError(f"Sdist missing {missing}")
    return {
        "wheel": str(wheels[0]),
        "wheel_files": len(wheel_files),
        "sdist": str(sdists[0]),
        "version": version,
        "sdist_files": len(names),
        "bundled_config_yaml": len(
            [p for p in wheel_files if p.startswith("conf/") and p.endswith(".yaml")]
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dist_dir", nargs="?", default="dist")
    args = p.parse_args()
    info = validate(Path(args.dist_dir))
    for k, v in info.items():
        print(f"[distribution] {k}: {v}")
    print("[distribution] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
