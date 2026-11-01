#!/usr/bin/env python3
"""Fail-closed, version-independent SOFTWARE release candidate verification.

This program validates distribution provenance, not the scientific validity of
new models, SAEs or experimental results. The historical M5 v0.2.0 release gate
remains separate and immutable.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "saebase.software_release_evidence.v1"
REQUIRED_CI = (
    "offline-tests (3.11)",
    "analysis-install",
    "package",
    "localization-native",
    "quality",
)
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
BANNED_SUFFIX = {".safetensors", ".ckpt", ".pth", ".pkl", ".gguf", ".pt"}
BANNED_NAME = {".env", "id_rsa", "id_ed25519"}
EVIDENCE_PREFIX = "conf/coverage/stage_d_evidence/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def source_sha(root: Path = ROOT) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not SHA.fullmatch(value):
        raise ValueError("source checkout has no exact 40-hex commit identity")
    return value


def _static_version(text: str) -> str:
    tree = ast.parse(text)
    matches = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(matches) != 1:
        raise ValueError("expected a single literal saebase.__version__")
    return matches[0]


def project_info(root: Path = ROOT) -> tuple[str, str, dict[str, str]]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = str(project["version"])
    if not VERSION.fullmatch(version):
        raise ValueError(f"expected a stable semantic version x.y.z, got {version!r}")
    if project["name"] != "opensae-saebase":
        raise ValueError("unexpected Python project name")
    declared = _static_version((root / "saebase" / "__init__.py").read_text(encoding="utf-8"))
    if declared != version:
        raise ValueError("pyproject.toml and saebase.__version__ disagree")
    changelog = (root / "harness/github/development_doc/CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^##[ \t]+{re.escape(version)}(?=\s|$)", changelog, re.M) is None:
        raise ValueError(f"CHANGELOG.md lacks a release heading for {version}")
    notes = root / "harness/github/development_doc" / "releases" / f"v{version}.md"
    if not notes.is_file() or not notes.read_text(encoding="utf-8").strip():
        raise ValueError(f"release notes missing for v{version}")
    entrypoints = project["scripts"]
    if len(entrypoints) != 9 or any(not k.startswith("saebase") for k in entrypoints):
        raise ValueError("unexpected public command registry")
    if any(not value.startswith("saebase.cli:") for value in entrypoints.values()):
        raise ValueError("non-namespaced public command entrypoint")
    return version, f"v{version}", entrypoints


def _safe_archive_name(name: str) -> None:
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts or not item.parts:
        raise ValueError(f"unsafe distribution member path {name!r}")
    if any(part in BANNED_NAME for part in item.parts):
        raise ValueError(f"private/credential file in distribution: {name}")
    if item.suffix.lower() in BANNED_SUFFIX:
        raise ValueError(f"model binary in software distribution: {name}")
    if EVIDENCE_PREFIX in name or "harness/github/work_plan/" in name:
        raise ValueError(f"non-runtime evidence in software distribution: {name}")


def inspect_distributions(
    root: Path, dist: Path, version: str, entrypoints: dict[str, str]
) -> dict[str, dict[str, object]]:
    wheels = sorted(dist.glob("*.whl"))
    sources = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("expected exactly one wheel and one source archive")
    wheel, sdist = wheels[0], sources[0]
    if not wheel.name.startswith(f"opensae_saebase-{version}-"):
        raise ValueError("wheel filename version does not match pyproject")
    if sdist.name != f"opensae_saebase-{version}.tar.gz":
        raise ValueError("sdist filename version does not match pyproject")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            _safe_archive_name(name)
        meta_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        entry_names = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        if len(meta_names) != 1 or len(entry_names) != 1:
            raise ValueError("wheel metadata or entrypoints missing/duplicated")
        metadata = Parser().parsestr(archive.read(meta_names[0]).decode("utf-8"))
        if (
            metadata.get("Name") != "opensae-saebase"
            or metadata.get("Version") != version
            or metadata.get("License-Expression") != "Apache-2.0"
        ):
            raise ValueError("wheel version, name or software license mismatch")
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser.read_string(archive.read(entry_names[0]).decode("utf-8"))
        if not parser.has_section("console_scripts"):
            raise ValueError("no installed console scripts")
        if dict(parser.items("console_scripts")) != entrypoints:
            raise ValueError("installed console scripts differ from source project")
        if "saebase/cli.py" not in names or "conf/config.yaml" not in names:
            raise ValueError("wheel lacks public API or runtime Hydra configuration")

    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = []
        for member in members:
            _safe_archive_name(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported sdist member: {member.name}")
            if member.isfile():
                names.append(member.name)
        root_path = f"opensae_saebase-{version}/"
        configs = [n for n in names if n == root_path + "pyproject.toml"]
        versions = [n for n in names if n == root_path + "saebase/__init__.py"]
        if len(configs) != 1 or len(versions) != 1:
            raise ValueError("sdist lacks version metadata or package init")
        src_project = tomllib.loads(archive.extractfile(configs[0]).read().decode("utf-8"))[
            "project"
        ]
        if src_project["version"] != version or src_project["name"] != "opensae-saebase":
            raise ValueError("sdist project metadata differs from source")
        if _static_version(archive.extractfile(versions[0]).read().decode("utf-8")) != version:
            raise ValueError("sdist package version differs from wheel")
        for required in (
            "LICENSE",
            "harness/github/development_doc/CHANGELOG.md",
            f"harness/github/development_doc/releases/v{version}.md",
        ):
            if root_path + required not in names:
                raise ValueError(f"sdist missing release source: {required}")

    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (wheel, sdist)
    }


def verify_ci_jobs(record: dict, commit: str) -> dict[str, str]:
    if not SHA.fullmatch(commit):
        raise ValueError("not a full source commit")
    jobs = record.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("CI jobs record missing")
    relevant = {}
    for item in jobs:
        name = item.get("name")
        if name not in REQUIRED_CI:
            continue
        if name in relevant:
            raise ValueError(f"duplicate CI check: {name}")
        if (
            item.get("status") != "completed"
            or item.get("conclusion") != "success"
            or item.get("head_sha") != commit
        ):
            raise ValueError(f"CI job not successful for requested source: {name}")
        relevant[name] = "success"
    if set(relevant) != set(REQUIRED_CI):
        raise ValueError(f"missing required CI jobs: {sorted(set(REQUIRED_CI) - set(relevant))}")
    return relevant


def prepare(
    root: Path,
    dist: Path,
    output: Path,
    commit: str,
    expected_version: str | None = None,
    ci_run_id: int | None = None,
) -> dict:
    if not SHA.fullmatch(commit) or source_sha(root) != commit:
        raise ValueError("release source commit does not equal checked-out HEAD")
    version, tag, commands = project_info(root)
    if expected_version is not None and expected_version != version:
        raise ValueError("requested version differs from pyproject and package")
    artifacts = inspect_distributions(root, dist, version, commands)
    evidence = {
        "schema": SCHEMA,
        "status": "software_distribution_verified",
        "tag": tag,
        "version": version,
        "source_commit": commit,
        "ci_run_id": ci_run_id,
        "ci_gate": "five successful full Release Validation jobs checked by release workflow"
        if ci_run_id
        else "not independently checked by this offline command",
        "software_license": "Apache-2.0",
        "distribution_artifacts": artifacts,
        "release_notes": f"harness/github/development_doc/releases/{tag}.md",
        "model_or_sae_weights_included": False,
        "scientific_scope": (
            "software packaging/API/integration checks; no new real checkpoint "
            "or scientific benchmark claims certified by this evidence"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "release-evidence.json"
    manifest.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"{data['sha256']}  {name}" for name, data in sorted(artifacts.items())]
    lines.append(f"{sha256_file(manifest)}  release-evidence.json")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    verify_assets(dist, output, commit, version)
    return evidence


def verify_assets(
    dist: Path, output: Path, commit: str | None = None, version: str | None = None
) -> dict:
    manifest_path = output / "release-evidence.json"
    evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema") != SCHEMA
        or evidence.get("status") != "software_distribution_verified"
    ):
        raise ValueError("invalid software release evidence")
    if commit is not None and evidence.get("source_commit") != commit:
        raise ValueError("release evidence source commit mismatch")
    if version is not None and evidence.get("version") != version:
        raise ValueError("release evidence version mismatch")
    artifacts = evidence.get("distribution_artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 2:
        raise ValueError("expected wheel and sdist digests")
    lines = (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    declared = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.+-]+)", line)
        if match is None:
            raise ValueError("invalid SHA256SUMS record")
        digest, name = match.groups()
        if name in declared:
            raise ValueError("duplicate SHA256SUMS record")
        declared[name] = digest
    if set(declared) != set(artifacts) | {"release-evidence.json"}:
        raise ValueError("asset checksum list is incomplete or contains extras")
    for name, digest in declared.items():
        file = manifest_path if name == "release-evidence.json" else dist / name
        if not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f"SHA256 mismatch: {name}")
        if name in artifacts:
            info = artifacts[name]
            if info["sha256"] != digest or info["bytes"] != file.stat().st_size:
                raise ValueError(f"asset manifest disagrees with data: {name}")
    return evidence


def publish_eligible(root: Path = ROOT, expected_version: str | None = None) -> dict:
    """Reject old, reused or lower version tags before a privileged publish."""
    version, tag, _ = project_info(root)
    if expected_version is not None and version != expected_version:
        raise ValueError("requested publication version does not match project metadata")
    tags = subprocess.check_output(
        ["git", "-C", str(root), "tag", "--list", "v*"], text=True
    ).splitlines()
    found = []
    for label in tags:
        if VERSION.fullmatch(label.removeprefix("v")) and label.startswith("v"):
            found.append((tuple(int(x) for x in label[1:].split(".")), label))
    candidate = tuple(int(x) for x in version.split("."))
    if found and candidate <= max(x[0] for x in found):
        raise ValueError("release version must be newer than all existing software tags")
    if any(label == tag for _, label in found):
        raise ValueError("release tag already exists")
    return {
        "tag": tag,
        "version": version,
        "newer_than_existing_software_tags": True,
        "existing_software_tag_count": len(found),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    a = mode.add_parser("prepare", help="check candidate and write auditable checksums")
    a.add_argument("--dist", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--source-sha", required=True)
    a.add_argument("--expected-version", default=None)
    a.add_argument("--ci-run-id", type=int, default=None)
    b = mode.add_parser("verify-assets", help="recheck artifact checksums and source identity")
    b.add_argument("--dist", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--source-sha", default=None)
    b.add_argument("--expected-version", default=None)
    c = mode.add_parser("verify-ci", help="confirm all five jobs passed on one commit")
    c.add_argument("--ci-jobs", type=Path, required=True)
    c.add_argument("--source-sha", required=True)
    d = mode.add_parser("publish-eligible", help="reject existing and older release tags")
    d.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            result = prepare(
                ROOT,
                args.dist,
                args.output_dir,
                args.source_sha,
                args.expected_version,
                args.ci_run_id,
            )
        elif args.mode == "verify-assets":
            result = verify_assets(
                args.dist, args.output_dir, args.source_sha, args.expected_version
            )
        elif args.mode == "verify-ci":
            result = verify_ci_jobs(
                json.loads(args.ci_jobs.read_text(encoding="utf-8")), args.source_sha
            )
        else:
            result = publish_eligible(ROOT, args.expected_version)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as err:
        parser.exit(1, f"Release candidate rejected: {err}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
