"""M5 source, distribution, and real matrix verification for GitHub release gates."""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import tarfile
import zipfile

from .runner import verify_matrix
from .schema import load_matrix


def audit_inventory(source: dict, config, *, license_path: Path) -> dict:
    if source.get("schema") != "saebase.m5.asset_provenance.v1":
        raise ValueError("asset inventory schema invalid")
    if source.get("bundled_model_or_sae_weights") is not False:
        raise ValueError("external checkpoint weight bundling forbidden")
    if source.get("software_license") != "Apache-2.0":
        raise ValueError("SAEBase software license mismatch")
    license_text = license_path.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Apache 2.0 LICENSE file not detected")
    entries = source.get("third_party_assets")
    if not isinstance(entries, list):
        raise ValueError("missing source provenance")
    index = {}
    for item in entries:
        key = item.get("key", "")
        if not key or key in index or item.get("kind") not in {"external_model", "external_sae"}:
            raise ValueError("invalid or duplicate asset key")
        if not item.get("source", "").startswith("https://") or not item.get("declared_license"):
            raise ValueError("asset source and license status must be documented")
        if "no weights bundled" not in item.get("distribution", ""):
            raise ValueError("asset distribution must explicitly prohibit weight bundling")
        index[key] = item
    required = {t.model_key for t in config.targets} | {
        s for t in config.targets for s in t.sae_keys
    }
    if not required.issubset(index):
        raise ValueError(
            f"missing M5 asset license inventory keys: {sorted(required - set(index))}"
        )
    for t in config.targets:
        if index[t.model_key]["source"] != t.model_card:
            raise ValueError("model card provenance differs between matrix and asset inventory")
        for key in t.sae_keys:
            if index[key]["source"] != t.sae_cards[key]:
                raise ValueError("SAE card provenance differs between matrix and asset inventory")
    return {
        "status": "documented",
        "count": len(required),
        "legal_review": "upstream licenses retained, not a legal certification",
    }


def audit_result(record: dict, config) -> dict:
    if record.get("schema") != "saebase.m5.results.v1" or record.get("status") != "passed":
        raise ValueError("release requires passed real M5 matrix")
    if not record.get("real_checkpoints"):
        raise ValueError("synthetic-only checkpoints are not eligible for release")
    if record.get("matrix_id") != config.matrix_id or record.get("release_version") != "0.2.0":
        raise ValueError("release matrix version/id mismatch")
    if (
        len(record.get("models", [])) != len(config.targets)
        or len({x["family"] for x in record["models"]}) < 2
    ):
        raise ValueError("M5 release requires at least two verified families")
    all_hashes = set()
    for result in record["models"]:
        target = next((t for t in config.targets if t.model_key == result["model_key"]), None)
        if target is None or result["family"] != target.family:
            raise ValueError("M5 target identity mismatch")
        if result["audit_groups"] < 3 or result["n_prompts"] != len(config.prompts):
            raise ValueError("M5 insufficient real prompt-group evidence")
        if not 0 <= result["elapsed_seconds"] <= config.max_runtime_seconds:
            raise ValueError("M5 result exceeded documented runtime budget")
        if result["peak_process_rss_bytes"] <= 0:
            raise ValueError("missing memory profile")
        if set(result["checkpoints"]) != set(target.sae_keys):
            raise ValueError("checkpoint coverage incomplete")
        if len(result["checkpoints"]) < 2:
            raise ValueError("at least two distinct SAEs required per model")
        for key, proof in result["checkpoints"].items():
            digest = proof.get("weight_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"missing checkpoint SHA-256 for {key}")
            if digest in all_hashes:
                raise ValueError("distinct checkpoint claims share a weight SHA-256")
            all_hashes.add(digest)
            if proof.get("sae_card") != target.sae_cards[key]:
                raise ValueError("SAE origin mismatch")
            if proof.get("metrics", 0) < len(config.metrics):
                raise ValueError("incomplete M3 metric coverage")
    if len(all_hashes) < 4:
        raise ValueError("insufficient independently validated checkpoint hashes")
    return {
        "status": "passed",
        "families": len({x["family"] for x in record["models"]}),
        "checkpoints": len(all_hashes),
        "no_composite_sae_score": True,
    }


def audit_distributions(dist: Path, expected_version: str = "0.2.0") -> dict:
    wheel = list(dist.glob("opensae_saebase-*.whl"))
    source = list(dist.glob("opensae_saebase-*.tar.gz"))
    if len(wheel) != 1 or len(source) != 1:
        raise ValueError("expected exactly one SAEBase wheel and one source distribution")
    banned_suffix = {".safetensors", ".ckpt", ".pth", ".pkl", ".gguf", ".pt"}
    with zipfile.ZipFile(wheel[0]) as archive:
        files = archive.namelist()
        if any(Path(x).suffix.lower() in banned_suffix for x in files):
            raise ValueError("binary model/SAE weights accidentally bundled in wheel")
        path = next((x for x in files if x.endswith(".dist-info/METADATA")), None)
        if path is None:
            raise ValueError("wheel metadata missing")
        data = archive.read(path).decode()
        for expected in (f"Version: {expected_version}", "License-Expression: Apache-2.0"):
            if expected not in data:
                raise ValueError(f"wheel release metadata missing {expected}")
        if "saebase/release/runner.py" not in files:
            raise ValueError("M5 release runner absent from wheel")
        ep = next((x for x in files if x.endswith(".dist-info/entry_points.txt")), None)
        commands = archive.read(ep).decode() if ep else ""
        for cli in (
            "saebase-release-matrix",
            "saebase-benchmark",
            "saebase-audit",
            "saebase-ground-truth",
        ):
            if cli not in commands:
                raise ValueError(f"missing official CLI {cli}")
    with tarfile.open(source[0], "r:gz") as archive:
        files = [x.name for x in archive if x.isfile()]
        if any(Path(x).suffix.lower() in banned_suffix for x in files):
            raise ValueError("binary checkpoint bundled in source distribution")
        if not any(
            x.endswith("/harness/github/development_doc/M5_RELEASE_NOTES.md") for x in files
        ):
            raise ValueError("M5 release notes absent from source archive")
    paths = wheel + source
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"version": expected_version, "artifacts": digests, "model_weights_bundled": False}


def generate_release_evidence(
    matrix_root: Path,
    dist: Path,
    config_path: Path,
    inventory_path: Path,
    license_path: Path,
    source_commit: str,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("release source commit must be exact 40-hex SHA")
    config = load_matrix(config_path)
    record = verify_matrix(matrix_root, expected_id=config.matrix_id)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    gates = {
        "matrix": audit_result(record, config),
        "provenance": audit_inventory(inventory, config, license_path=license_path),
        "distributions": audit_distributions(dist),
    }
    return {
        "schema": "saebase.m5.release_evidence.v1",
        "status": "passed",
        "tag": "v0.2.0",
        "source_commit": source_commit,
        "matrix_id": config.matrix_id,
        "matrix_manifest_sha256": hashlib.sha256(
            (matrix_root / "matrix_manifest.json").read_bytes()
        ).hexdigest(),
        "gates": gates,
        "real_matrix_evidence_sha256": hashlib.sha256(
            (matrix_root / "matrix_results.json").read_bytes()
        ).hexdigest(),
        "notes": "No external weights distributed; model/SAE license rights not certified.",
    }
