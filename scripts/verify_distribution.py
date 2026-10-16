#!/usr/bin/env python3
"""Validate built SAEBase wheel/sdist contents without installing optional packages."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import tarfile
import zipfile


NEEDED_WHEEL = {
    "main.py",
    "saebase/cli.py", "saebase/model_plugins.py",
    "saebase/tasks/__init__.py", "saebase/tasks/plugins.py",
    "saebase/benchmarks/__init__.py", "saebase/benchmarks/runner.py",
    "saebase/benchmarks/planner.py", "saebase/benchmarks/results.py",
    "saebase/reliability/__init__.py", "saebase/reliability/runner.py",
    "saebase/reliability/ground_truth.py",
    "saebase/release/__init__.py", "saebase/release/schema.py",
    "saebase/release/planner.py", "saebase/release/runner.py",
    "saebase/release/gates.py",
    "models/gpt2small/gpt2small.py",
    "scripts/m3_benchmark.py", "scripts/m4_reliability.py",
    "scripts/m4_ground_truth.py", "scripts/m5_release_matrix.py",
    "scripts/m5_release_gate.py",
    "conf/release/m5_matrix.json", "conf/release/m5_extended_gpt2.json",
    "conf/release/m5_asset_provenance.json",
    "saebase/__init__.py",
    "saebase/api/__init__.py",
    "saebase/api/features.py",
    "saebase/interventions/__init__.py", "saebase/interventions/engine.py",
    "saebase/interventions/specs.py", "saebase/interventions/operators.py",
    "saebase/interventions/results.py", "saebase/interventions/steering.py",
    "scripts/m2_batch.py", "conf/interventions/m2_pythia_example.json",
    "saebase/store/__init__.py", "saebase/store/identity.py",
    "saebase/store/shards.py", "saebase/store/collect.py",
    "saebase/integrations/sar.py",
    "saebase/integrations/rss.py",
    "saebase/sae/registry.py",
    "saebase/data/prompt_store.py",
    "models/plugins.py",
    "tasks/plugins.py",
    "scripts/sae_prepare_prompts.py",
    "conf/__init__.py",
    "conf/config.yaml",
    "conf/model/pythia70m.yaml",
    "conf/sae/pythia70m_resid_sm.yaml",
    "conf/prompt_source/simple_copying.yaml",
    "conf/evaluation/activation_health.yaml",
    "conf/task/sae.yaml",
}
NEEDED_SDIST = NEEDED_WHEEL | {
    ".github/workflows/m5-matrix.yml", ".github/workflows/m5-release.yml",
    "docs/M5_RELEASE_NOTES.md", "docs/M5_RELEASE_MATRIX.md",
    "tests/test_m5_matrix.py", "tests/test_m5_gates.py",
    "pyproject.toml", "README.md", "README_cn.md", "LICENSE", "NOTICE",
    "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "CITATION.cff",
    "MANIFEST.in", ".github/workflows/ci.yml",
    "docs/RELEASE_CHECKLIST.md", "docs/PROVENANCE.md", "docs/API.md",
    "tests/test_phase_a.py", "tests/test_phase_d.py",
    "tests/test_m1_store.py", "tests/test_m2_interventions.py", "tests/fixtures/m0_reference_public.json",
    "docs/M1_ACTIVATION_STORE.md", "examples/10_activation_store_offline.py",
    ".github/workflows/m1-real.yml",
    ".github/workflows/m2-real.yml", "docs/M2_BATCH_INTERVENTIONS.md",
    "scripts/m2_real_smoke.py", "examples/11_batch_interventions_offline.py",
}
BAD_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
BAD_TOP_LEVEL = {"experiment", "data", "dist", "build"}


def validate(dist_dir: Path) -> dict[str, object]:
    wheels = sorted(dist_dir.glob("opensae_saebase-*.whl"))
    sdists = sorted(dist_dir.glob("opensae_saebase-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"Expected one wheel and one sdist in {dist_dir}; got {len(wheels)}, {len(sdists)}")
    with zipfile.ZipFile(wheels[0]) as z:
        wheel_files = set(z.namelist())
        missing = sorted(NEEDED_WHEEL - wheel_files)
        if missing:
            raise AssertionError(f"Wheel missing {missing}")
        forbidden_wheel = sorted(p for p in wheel_files if p.startswith(
            "conf/coverage/stage_d_evidence/") or p.startswith("work_plan/")
            or p.endswith((".safetensors", ".pt", ".pth", ".ckpt")))
        if forbidden_wheel:
            raise AssertionError(f"Non-runtime assets leaked into wheel: {forbidden_wheel[:5]}")
        entry_file = next((p for p in wheel_files if p.endswith(".dist-info/entry_points.txt")), None)
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
        if "Name: opensae-saebase" not in metadata or "Version: 0.2.0" not in metadata:
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
    return {"wheel": str(wheels[0]), "wheel_files": len(wheel_files),
            "sdist": str(sdists[0]), "sdist_files": len(names),
            "bundled_config_yaml": len([p for p in wheel_files if p.startswith('conf/') and p.endswith('.yaml')])}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dist_dir", nargs="?", default="dist")
    args = p.parse_args()
    info = validate(Path(args.dist_dir))
    for k,v in info.items():
        print(f"[distribution] {k}: {v}")
    print("[distribution] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
