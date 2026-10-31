#!/usr/bin/env python3
"""Verify and package Stage D evidence without rerunning a model or SAE.

Two deliberately distinct outputs:
  compact: the 200 committed evidence files, supporting provenance and verifier
  raw:     only original GPU archives whose stored byte hashes have been checked

The compact bundle is never presented as the original M3/M4 raw run cache.
Uses Python standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("conf/coverage/stage_d_evidence")
SNAPSHOT = "stage-d-evidence-20261016"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
SUPPORT_FILES = (
    Path("conf/coverage/stage_bc.json"),
    Path("conf/coverage/stage_b_cpu_verified.json"),
    Path("conf/coverage/stage_c_cpu_verified.json"),
    Path("conf/coverage/stage_d_population.jsonl"),
    Path("harness/github/development_doc/verified-model-matrix.md"),
    Path("harness/github/development_doc/stage-d-evidence-archive.md"),
    Path("saebase/tools/stage_d_archive.py"),
)
SHA_LINE = re.compile(r"^([a-f0-9]{64})  (\S+)$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_obj(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe evidence file: {relative}")
    return path


def verify_checksums(root: Path) -> int:
    base = root / EVIDENCE
    manifest = base / "final/checksums.sha256"
    if not manifest.is_file():
        raise ValueError("missing original checksum manifest")
    included = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid checksum line {number}: {line!r}")
        sha, rel = match.groups()
        if rel in included or rel == "final/checksums.sha256":
            raise ValueError(f"duplicate or self-referential checksum: {rel}")
        file = _relative_file(base, rel)
        if digest(file) != sha:
            raise ValueError(f"checksum mismatch for {rel}")
        included[rel] = sha
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()} - {
        "final/checksums.sha256"
    }
    if set(included) != actual:
        raise ValueError(
            f"checksum coverage mismatch: missing {sorted(actual - set(included))}; "
            f"orphaned {sorted(set(included) - actual)}"
        )
    if len(included) != 199:
        raise ValueError(f"expected 199 checksummed compact evidence files, found {len(included)}")
    return len(included)


def verify_compact(root: Path = ROOT) -> dict:
    root = root.resolve()
    n_hashed = verify_checksums(root)
    cat = json_obj(root / "conf/coverage/stage_bc.json")
    final = json_obj(root / EVIDENCE / "final/stage_d_final_matrix.json")
    status = json_obj(root / EVIDENCE / "status.json")
    raw = json_obj(root / EVIDENCE / "final/raw_run_archive_index.json")
    if cat.get("schema") != "saebase.coverage.stage_bc.v1":
        raise ValueError("unexpected catalog schema")
    if final.get("schema") != "saebase.stage_d.final_matrix.v1":
        raise ValueError("unexpected final matrix schema")
    if status.get("schema") != "saebase.stage_d.status.v1":
        raise ValueError("unexpected status ledger schema")
    targets, rows = cat["targets"], final["rows"]
    expected = {
        (p["model_key"], p["sae_key"], layer) for p in targets for layer in p["selected_layers"]
    }
    found = set()
    keys = set()
    for item in rows:
        key = (item["model_key"], item["sae_key"], item["layer"])
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate site: {key}")
        found.add(key)
        entry = f"{key[0]}|{key[1]}|L{key[2]}|{item['hook_name']}"
        keys.add(entry)
        if (
            item.get("pair_verified") is not True
            or item.get("benchmark_verified") is not True
            or item.get("reliability_verified") is not True
            or item.get("protocol") != "population"
        ):
            raise ValueError(f"non-verifiable final row: {key}")
        if status.get("sites", {}).get(entry, {}).get("status") != item.get("status"):
            raise ValueError(f"status record mismatch: {key}")
        report_path = item.get("m3m4_summary")
        if not isinstance(report_path, str):
            raise ValueError(f"no M3/M4 report: {key}")
        report = json_obj(_relative_file(root / EVIDENCE, report_path))
        for name, expected_value in zip(
            ("model_key", "sae_key", "layer", "hook_name"), (*key, item["hook_name"])
        ):
            if report.get(name) != expected_value:
                raise ValueError(f"report identity mismatch for {key}: {name}")
        if (
            report.get("status") != "passed"
            or report.get("protocol") != "population"
            or report.get("feature_selection_split") != "selection"
            or report.get("n_independent_evaluation_groups", 0) < 64
            or report.get("reliability_metric_audits", 0) < 1
        ):
            raise ValueError(f"incomplete 64-group report: {key}")
        if item.get("acceptance_path"):
            _relative_file(root / EVIDENCE, item["acceptance_path"])
        elif not item.get("prior_pair_verified"):
            raise ValueError(f"missing new pair evidence: {key}")
    if found != expected or keys != set(status.get("sites", {})):
        raise ValueError("catalog/ledger/final matrix coverage disagreement")
    if (
        len(targets) != 18
        or len(rows) != 48
        or final.get("n_sites") != 48
        or final.get("n_pair_verified") != 48
        or final.get("n_population_m3m4") != 48
        or final.get("n_smoke_only_m3m4") != 0
        or final.get("counts") != {"reliability_verified": 48}
        or status.get("n_sites") != 48
    ):
        raise ValueError("Stage D source-declared totals inconsistent with final rows")
    if len(list((root / EVIDENCE / "reports").glob("*_m3m4.json"))) != 48:
        raise ValueError("unexpected per-site report count")
    for kind in ("evaluation", "selection"):
        if len(list((root / EVIDENCE / kind).rglob("L*.json"))) != 48:
            raise ValueError(f"incomplete {kind} token ledgers")
    records = raw.get("archives")
    if raw.get("schema") != "saebase.stage_d.raw_archive_index.v1":
        raise ValueError("unexpected original raw archive index schema")
    if not isinstance(records, list) or len(records) != 18:
        raise ValueError("expected 18 original GPU raw archives in index")
    if len(set(entry["file"] for entry in records)) != len(records):
        raise ValueError("duplicated original archive names")
    for entry in records:
        name = entry["file"]
        if (
            PurePosixPath(name).name != name
            or not name.endswith(".tar.gz")
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] <= 0
        ):
            raise ValueError(f"malformed original archive index row: {entry}")
    for path in SUPPORT_FILES:
        _relative_file(root, path.as_posix())
    return {
        "schema": "opensae.stage_d.compact_archive_audit.v1",
        "stage_d_recorded_at": final["updated_at_utc"],
        "sites": 48,
        "profiles": 18,
        "real_pair_acceptance": 48,
        "population_benchmark": 48,
        "population_reliability": 48,
        "checksummed_evidence_files": n_hashed,
        "raw_original_archives_indexed": len(records),
        "raw_original_archives_included": False,
        "scope": "engineering_acceptance_not_scientific_quality_certification",
    }


def _files_to_package(root: Path) -> list[Path]:
    compact = sorted(p.relative_to(root) for p in (root / EVIDENCE).rglob("*") if p.is_file())
    return sorted(set(compact) | set(SUPPORT_FILES), key=lambda p: p.as_posix())


def _archive(output: Path, entries: dict[str, Path | bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_file, mtime=0, compresslevel=9
        ) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as tar:
                for name, source in sorted(entries.items()):
                    info = tarfile.TarInfo(name=f"{SNAPSHOT}/{name}")
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    if isinstance(source, bytes):
                        info.size = len(source)
                        tar.addfile(info, io.BytesIO(source))
                    else:
                        info.size = source.stat().st_size
                        with source.open("rb") as handle:
                            tar.addfile(info, handle)


def _payload_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def source_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha
    except (OSError, subprocess.CalledProcessError):
        pass
    return "unknown"


def build_compact(root: Path, destination: Path, *, commit: str | None = None) -> Path:
    audit = verify_compact(root)
    paths = _files_to_package(root)
    items = {}
    for rel in paths:
        path = _relative_file(root, rel.as_posix())
        items[rel.as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "opensae.stage_d.bundle_manifest.v1",
        "snapshot": SNAPSHOT,
        "source_repo": "OpenSAE/SAEBase",
        "source_commit": commit if commit is not None else source_commit(),
        "audit": audit,
        "files": items,
        "archive_contains_full_raw_runs": False,
        "original_raw_index_present": True,
        "note": "This contains committed compact records, not original M3/M4 run directories.",
    }
    payload = _payload_json(manifest)
    entries: dict[str, Path | bytes] = {p.as_posix(): root / p for p in paths}
    entries[MANIFEST_NAME] = payload
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / f"{SNAPSHOT}-compact.tar.gz"
    _archive(artifact, entries)
    summary_path = destination / f"{SNAPSHOT}-compact-manifest.json"
    summary_path.write_bytes(payload)
    validator = destination / "verify-stage-d.py"
    shutil.copyfile(root / "saebase/tools/stage_d_archive.py", validator)
    sums = [
        f"{digest(destination / name)}  {name}"
        for name in (artifact.name, summary_path.name, validator.name)
    ]
    (destination / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    verify_bundle(artifact)
    return artifact


def _extract_checked(archive: Path, destination: Path) -> Path:
    total = 0
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name
            pure = PurePosixPath(name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or len(pure.parts) < 2
                or pure.parts[0] != SNAPSHOT
                or any(piece in (".", "..", "") for piece in pure.parts)
                or name in seen
                or member.size > 20_000_000
            ):
                raise ValueError(f"unsafe or unexpected archive member: {name}")
            seen.add(name)
            total += member.size
            if total > 100_000_000:
                raise ValueError("bundle too large for compact evidence")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"archive traversal: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            file = tar.extractfile(member)
            if file is None:
                raise ValueError(f"unreadable archive member: {name}")
            with target.open("wb") as dest:
                shutil.copyfileobj(file, dest)
    return destination / SNAPSHOT


def verify_bundle(archive: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="saebase-evidence-") as temp:
        base = _extract_checked(archive, Path(temp))
        manifest = json_obj(base / MANIFEST_NAME)
        if manifest.get("schema") != "opensae.stage_d.bundle_manifest.v1":
            raise ValueError("unexpected bundle manifest version")
        entries = manifest.get("files")
        if not isinstance(entries, dict):
            raise ValueError("bundle manifest has no file listing")
        found = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
        if found != set(entries) | {MANIFEST_NAME}:
            raise ValueError("archive member list differs from manifest")
        for rel, meta in entries.items():
            file = _relative_file(base, rel)
            if file.stat().st_size != meta.get("bytes") or digest(file) != meta.get("sha256"):
                raise ValueError(f"bundle file hash mismatch: {rel}")
        audit = verify_compact(base)
        if (
            audit != manifest.get("audit")
            or manifest.get("archive_contains_full_raw_runs") is not False
        ):
            raise ValueError("bundle falsely claims raw data or has inconsistent audit counts")
        return audit


def verify_original_archives(raw_dir: Path, index: dict) -> dict:
    records = index.get("archives")
    if not isinstance(records, list) or not records:
        raise ValueError("raw archive index is empty")
    if not raw_dir.is_dir():
        raise ValueError(f"original GPU raw archive directory unavailable: {raw_dir}")
    total = 0
    for entry in records:
        name = entry["file"]
        if PurePosixPath(name).name != name or not name.endswith(".tar.gz"):
            raise ValueError(f"unsafe raw archive name: {name}")
        file = _relative_file(raw_dir, name)
        if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
            raise ValueError(f"original raw archive digest/size mismatch: {name}")
        with tarfile.open(file, "r:gz") as archive:
            for member in archive:
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsafe contents in {name}: {member.name}")
        total += file.stat().st_size
    return {
        "verified_archives": len(records),
        "original_bytes": total,
        "status": "byte_hash_verified",
    }


def build_raw(root: Path, raw_dir: Path, destination: Path) -> Path:
    index_path = root / EVIDENCE / "final/raw_run_archive_index.json"
    index = json_obj(index_path)
    audit = verify_original_archives(raw_dir, index)
    entries: dict[str, Path | bytes] = {
        "RAW_ARCHIVE_INDEX.json": index_path,
        "RAW_VERIFICATION.json": _payload_json(audit),
    }
    for item in index["archives"]:
        entries[f"raw/{item['file']}"] = raw_dir / item["file"]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{SNAPSHOT}-original-raw.tar.gz"
    _archive(archive, entries)
    (destination / f"{archive.name}.sha256").write_text(
        f"{digest(archive)}  {archive.name}\n", encoding="ascii"
    )
    verify_raw_bundle(archive)
    return archive


def verify_raw_bundle(archive: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="saebase-raw-") as temp:
        root = _extract_checked(archive, Path(temp))
        index = json_obj(root / "RAW_ARCHIVE_INDEX.json")
        recorded = json_obj(root / "RAW_VERIFICATION.json")
        verified = verify_original_archives(root / "raw", index)
        if recorded != verified:
            raise ValueError("raw bundle counts differ from original archive index")
        expected = {f"raw/{a['file']}" for a in index["archives"]}
        found = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if found != expected | {"RAW_ARCHIVE_INDEX.json", "RAW_VERIFICATION.json"}:
            raise ValueError("raw bundle contains extra or missing files")
        return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify committed compact evidence")
    mode.add_argument("--build", action="store_true", help="build public compact evidence bundle")
    mode.add_argument("--verify-bundle", type=Path, metavar="ARCHIVE")
    mode.add_argument("--verify-raw", type=Path, metavar="RAW_DIR")
    mode.add_argument("--build-raw", type=Path, metavar="RAW_DIR")
    mode.add_argument("--verify-raw-bundle", type=Path, metavar="ARCHIVE")
    parser.add_argument("--output-dir", type=Path, default=Path("dist/stage-d-evidence"))
    args = parser.parse_args()
    try:
        if args.check:
            result = verify_compact(ROOT)
        elif args.build:
            file = build_compact(ROOT, args.output_dir)
            result = {"archive": str(file), "sha256": digest(file), "verified": True}
        elif args.verify_bundle:
            result = verify_bundle(args.verify_bundle)
        elif args.verify_raw:
            result = verify_original_archives(
                args.verify_raw, json_obj(ROOT / EVIDENCE / "final/raw_run_archive_index.json")
            )
        elif args.build_raw:
            file = build_raw(ROOT, args.build_raw, args.output_dir)
            result = {"raw_archive": str(file), "sha256": digest(file), "verified": True}
        else:
            result = verify_raw_bundle(args.verify_raw_bundle)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"Stage D evidence verification failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
