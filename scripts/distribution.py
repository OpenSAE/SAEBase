"""Inspect distribution contents, test external installation, and rebuild source archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

NEEDED_WHEEL = {
    "saebase/api/__init__.py",
    "saebase/datasets/plugins.py",
    "tasks/benchmark/reliability/cli.py",
    "conf/config.yaml",
    "conf/prompt_source/catalog.yaml",
    "tasks/precheck/task.py",
    "saebase/extensions/metrics.py",
    "saebase/api/sae_loading.py",
    "tasks/benchmark/metrics/health_cli.py",
    "tasks/benchmark/execution/runner.py",
    "saebase/extensions/registry.py",
    "saebase/api/sae_profiles.py",
    "tasks/intervene/cli.py",
    "tasks/benchmark/comparison/schema.py",
    "conf/sae/catalog.yaml",
    "tasks/intervene/backend/specs.py",
    "conf/__init__.py",
    "saebase/store/shards.py",
    "tasks/benchmark/methods/rss.py",
    "tasks/benchmark/cli.py",
    "tasks/plugins.py",
    "saebase/__init__.py",
    "conf/interventions/pythia_feature_edits.json",
    "tasks/capture/sar.py",
    "saebase/extensions/contracts.py",
    "tasks/benchmark/reliability/controls_cli.py",
    "models/plugins.py",
    "saebase/store/__init__.py",
    "saebase/api/paper_sources.py",
    "conf/evaluation/activation_health.yaml",
    "saebase/store/collect.py",
    "models/gpt2/gpt2/specs.py",
    "tasks/benchmark/execution/planner.py",
    "conf/benchmark/model_comparison.json",
    "saebase/datasets/artifacts.py",
    "conf/benchmark/model_comparison_gpt2.json",
    "saebase/api/features.py",
    "tasks/intervene/backend/steering.py",
    "tasks/intervene/backend/engine.py",
    "tasks/intervene/backend/operators.py",
    "saebase/datasets/datamodule.py",
    "conf/task/precheck.yaml",
    "tasks/capture/sar_mode.py",
    "tasks/benchmark/reliability/runner.py",
    "tasks/benchmark/reliability/__init__.py",
    "saebase/extensions/runtime.py",
    "tasks/generate/backend/decoder_steering.py",
    "saebase/runtime/experiment_results.py",
    "tasks/generate/backend/engine.py",
    "tasks/intervene/backend/results.py",
    "saebase/cli.py",
    "tasks/benchmark/reliability/ground_truth.py",
    "conf/model/catalog.yaml",
    "tasks/precheck/operation.py",
    "tasks/generate/backend/sae_interventions.py",
    "saebase/extensions/__init__.py",
    "saebase/store/identity.py",
    "tasks/benchmark/results/metrics.py",
    "tasks/intervene/backend/__init__.py",
    "saebase/datasets/prepare.py",
    "tasks/benchmark/execution/__init__.py",
    "conf/dataset/catalog.yaml",
}

NEEDED_SDIST = (
    NEEDED_WHEEL
    - {
        "saebase/datasets/prepare.py",
        "saebase/datasets/plugins.py",
        "saebase/datasets/artifacts.py",
        "saebase/datasets/datamodule.py",
    }
) | {
    "datasets/prepare.py",
    "datasets/plugins.py",
    "datasets/artifacts.py",
    "datasets/datamodule.py",
    "tests/product/test_model_comparison.py",
    "pyproject.toml",
    "README.md",
    "README_cn.md",
    "LICENSE",
    "doc/code/development/legal/NOTICE",
    ".github/CONTRIBUTING.md",
    ".github/SECURITY.md",
    "doc/code/development/CHANGELOG.md",
    "CITATION.cff",
    "MANIFEST.in",
    ".github/workflows/ci.yml",
    "doc/code/development/RELEASE_CHECKLIST.md",
    "doc/code/development/PROVENANCE.md",
    "doc/code/development/API.md",
    "tests/product/test_reference_model_coverage.py",
    "tests/product/test_model_site_evaluation.py",
    "tests/product/test_activation_store.py",
    "tests/product/test_feature_interventions.py",
    "tests/product/fixtures/m0_reference_public.json",
    "doc/code/development/activation-store.md",
    ".github/workflows/real-smoke.yml",
    ".github/workflows/research-regression.yml",
    ".github/workflows/upstream-compatibility.yml",
    ".github/workflows/software-release.yml",
    "scripts/release.py",
    "scripts/distribution.py",
    "scripts/quality.py",
    "doc/code/development/release-process.md",
    "tests/product/test_release_candidate.py",
    "doc/code/development/extensions.md",
    "tests/product/test_external_extensions.py",
    "scripts/examples/external_providers.py",
    "doc/code/development/first-experiment.md",
    "doc/code/development/troubleshooting.md",
    ".github/CONTRIBUTING.md",
    ".github/CONTRIBUTING.md",
    "scripts/documentation.py",
    "tests/product/test_developer_experience.py",
    "scripts/examples/first_experiment.py",
    ".github/ISSUE_TEMPLATE/model_sae_support.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    "tests/product/test_feature_interventions.py",
    "doc/code/development/feature-interventions.md",
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
            or p.startswith("harness/github/work_plan/")
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
        for group in ("model", "sae", "dataset", "prompt_source"):
            resource = f"conf/{group}/catalog.yaml"
            if resource not in wheel_files or not z.read(resource).strip():
                raise AssertionError(f"Missing or empty component catalog: {resource}")
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


def smoke(dist_dir: Path) -> dict[str, str]:
    wheels = list(dist_dir.glob("opensae_saebase-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Pass a dist directory containing exactly one SAEBase wheel")
    with tempfile.TemporaryDirectory(prefix="saebase-wheel-") as temp:
        temp_dir = Path(temp)
        target = temp_dir / "site"
        run_dir = temp_dir / "external-workdir"
        run_dir.mkdir()
        installed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(target),
                str(wheels[0].resolve()),
            ],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if installed.returncode:
            raise RuntimeError(
                "Wheel installation failed:\n" + installed.stdout + "\n" + installed.stderr
            )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        env["PYTHONNOUSERSITE"] = "1"
        script = target / "bin" / "saebase"
        commands = [
            [str(script), "--cfg", "job"],
            [str(target / "bin" / "saebase-prepare-prompts"), "--help"],
            [str(target / "bin" / "saebase-intervene"), "--help"],
            [str(target / "bin" / "saebase-ground-truth"), "--help"],
            [str(target / "bin" / "saebase-check-env"), "--help"],
            [str(target / "bin" / "saebase-benchmark"), "--help"],
            [str(target / "bin" / "saebase-audit"), "--help"],
            [sys.executable, "-m", "saebase.runtime.train", "--cfg", "job", "--help"],
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; from importlib.resources import files; "
                    "import saebase, conf; "
                    "from saebase.api import ModelAdapter, ActivationSite; "
                    "import saebase.cli, models.plugins, tasks.plugins; "
                    "import models.plugins, tasks.plugins, tasks.benchmark.cli; "
                    "from saebase.registry import list_registered_task_keys; "
                    "assert {'precheck','capture','intervene','generate','circuits','explain','benchmark','train'}.issubset(set(list_registered_task_keys())); "
                    "assert callable(tasks.plugins.BenchmarkTask); "
                    "assert (files('conf')/'config.yaml').is_file(); "
                    "import saebase.runtime.train, saebase.datasets.plugins; "
                    "assert (files('conf')/'benchmark'/'model_comparison.json').is_file(); "
                    ""
                    "assert 'pythia70m_resid_sm' in conf.group_names('sae'); "
                    "from saebase.cli import main as cli_main; assert callable(cli_main); "
                    "print(Path(saebase.__file__).resolve()); "
                    "print('wheel configuration and API OK')"
                ),
            ],
        ]
        outputs = []
        for cmd in commands:
            r = subprocess.run(
                cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=35
            )
            if r.returncode:
                raise RuntimeError(f"Installed wheel command failed: {cmd}\n{r.stdout}\n{r.stderr}")
            outputs.append(r.stdout)
        if "wheel configuration and API OK" not in outputs[-1]:
            raise AssertionError("Installed API and package resource smoke test did not finish")
        if str(target) not in outputs[-1]:
            raise AssertionError("Wheel smoke imported the checkout, not the installed package")
        return {
            "installation": "OK",
            "registry_cli": "OK",
            "preparation_cli": "OK",
            "environment_cli": "OK",
            "package_resources": "OK",
            "external_working_directory": "OK",
        }


def check_roundtrip(dist_dir: Path) -> None:
    sources = list(dist_dir.glob("opensae_saebase-*.tar.gz"))
    if len(sources) != 1:
        raise ValueError("expected exactly one SAEBase source archive")
    with tempfile.TemporaryDirectory(prefix="saebase-sdist-check-") as work:
        root = Path(work)
        with tarfile.open(sources[0], "r:gz") as archive:
            archive.extractall(root, filter="data")
        packages = [p for p in root.iterdir() if p.is_dir()]
        if len(packages) != 1 or not (packages[0] / "pyproject.toml").is_file():
            raise ValueError("source archive has an unexpected package layout")
        output = root / "rebuilt-dist"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
                str(output),
            ],
            check=True,
            cwd=packages[0],
        )
        shutil.copy2(sources[0], output / sources[0].name)
        validate(output)
        print("Clean sdist -> wheel roundtrip OK")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contents", "install", "roundtrip"))
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    if args.action == "contents":
        print(json.dumps(validate(args.dist_dir), indent=2))
    elif args.action == "install":
        print(json.dumps(smoke(args.dist_dir), indent=2))
    else:
        check_roundtrip(args.dist_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
