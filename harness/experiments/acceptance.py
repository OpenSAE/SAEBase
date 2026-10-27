"""P8 reproducible ledger, evidence-archive and release-readiness validation.

No feature is promoted because a file was committed or a GPU agent said PASS.
P8 checks source-bound evidence, independent reviewer and requested scope.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

REQUIRED_IDS = {f"{i:02d}" for i in range(1, 51)}
VALID_STATES = {
    "TO_AUDIT",
    "NOT_STARTED",
    "PARTIAL",
    "PASS",
    "BLOCKED",
    "CPU_VERIFIED",
    "GPU_VERIFIED",
    "RELEASED",
}
FIELDS = {
    "id",
    "reference_project",
    "capability",
    "phase",
    "compute_tier",
    "execution_owner",
    "current_status",
    "verified_scope",
    "upstream_reference",
    "open_sae_entrypoint",
    "native_parity_evidence",
    "cpu_evidence",
    "gpu_evidence",
    "artifact_link",
    "reviewer",
    "blocking_reason",
    "core_grade",
    "core_criterion",
    "core_test",
    "remaining_core_work",
    "p8_status",
}


def audit_ledger(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if not rows or set(rows[0]) != FIELDS:
        raise ValueError("Unsupported native capabilities ledger columns")
    ids = [row["id"] for row in rows]
    if set(ids) != REQUIRED_IDS or len(ids) != 50 or len(ids) != len(set(ids)):
        raise ValueError("Require exactly F01–F50, once each")
    counts: dict[str, int] = {status: 0 for status in sorted(VALID_STATES)}
    missing: list[dict[str, Any]] = []
    for row in rows:
        status = row["current_status"]
        if status not in VALID_STATES:
            raise ValueError(f"Invalid state for F{row['id']}")
        counts[status] += 1
        grade = row["core_grade"]
        if not row["core_criterion"].strip() or not row["core_test"].strip():
            raise ValueError(f"F{row['id']} missing core acceptance criterion/test")
        if status == "PASS" and (grade != "3/3" or row["remaining_core_work"].strip()):
            raise ValueError(f"F{row['id']} cannot be core PASS with incomplete work")
        if status == "PARTIAL" and (
            grade not in {"0/3", "1/3", "2/3"} or not row["remaining_core_work"].strip()
        ):
            raise ValueError(f"F{row['id']} needs a graded specific core gap")
        if status == "BLOCKED" and not row["remaining_core_work"].strip():
            raise ValueError(f"F{row['id']} needs an exact external dependency")
        if row["p8_status"] not in {"PENDING", "VERIFIED"}:
            raise ValueError(f"F{row['id']} invalid independent P8 status")
        if row["p8_status"] == "VERIFIED" and status != "RELEASED":
            raise ValueError("P8 VERIFIED requires distinct final RELEASED state")
        if status in {"CPU_VERIFIED", "GPU_VERIFIED", "RELEASED"}:
            needs = ["verified_scope", "cpu_evidence", "reviewer", "artifact_link"]
            if status in {"GPU_VERIFIED", "RELEASED"} and "C0" not in row["compute_tier"]:
                needs += ["gpu_evidence", "native_parity_evidence"]
            absent = [name for name in needs if not row[name].strip()]
            if absent:
                missing.append(
                    {"id": row["id"], "reason": "verified_without_evidence", "fields": absent}
                )
        if status not in {"CPU_VERIFIED", "GPU_VERIFIED", "RELEASED"}:
            missing.append({"id": row["id"], "reason": "not_finally_verified", "status": status})
    released = counts["RELEASED"]
    complete = (
        released == 50 and not missing and all(row["p8_status"] == "VERIFIED" for row in rows)
    )
    return {
        "schema": "opensae.native capabilities.p8.acceptance.v1",
        "ledger_sha256": hashlib.sha256(content).hexdigest(),
        "capabilities": len(rows),
        "status_counts": counts,
        "core_pass_count": counts["PASS"] + counts["RELEASED"],
        "core_partial_count": counts["PARTIAL"],
        "core_blocked_count": counts["BLOCKED"],
        "core_gaps": [
            {"id": row["id"], "grade": row["core_grade"], "missing": row["remaining_core_work"]}
            for row in rows
            if row["current_status"] in {"PARTIAL", "BLOCKED"}
        ],
        "p8_pending_count": sum(row["p8_status"] == "PENDING" for row in rows),
        "release_ready": complete,
        "missing": missing,
        "external_user_tests": "not_inferred_from_ledger",
        "two_gpu_acceptance": (
            "independent_evidence_required"
            if next(r for r in rows if r["id"] == "50")["current_status"] != "RELEASED"
            else "ledger_reports_released_verify_raw_signoff"
        ),
    }


def _safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and not any(part in {".", "..", ""} for part in path.parts)
        and "\\" not in name
    )


def inspect_archive(
    path: Path, *, expected_sha256: str, max_unpacked_bytes: int = 4_000_000_000
) -> dict[str, Any]:
    """Hash the real archive bytes; inspect without executing archived code.

    ZIP CRC, internal SHA256 entries and raw tensor counts are different
    checks. Raw .pt weights/pickles are NEVER deserialized in this module.
    """
    if not path.is_file() or len(expected_sha256) != 64:
        raise ValueError("Actual archive and expected outer SHA256 required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("External archive SHA256 differs from original declaration")
    with zipfile.ZipFile(path) as package:
        infos = package.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if len(names) != len(set(names)) or any(not _safe_zip_name(name) for name in names):
            raise ValueError("Unsafe, duplicated or path-traversing ZIP entries")
        if sum(item.file_size for item in infos) > max_unpacked_bytes:
            raise ValueError("Archive expands beyond reviewed evidence budget")
        # No third-party weight binaries or obvious private environment files.
        forbidden = [
            name
            for name in names
            if name.endswith((".env", ".safetensors", ".ckpt")) or "/.env" in name
        ]
        if forbidden:
            raise ValueError("Archive contains checkpoint/environment paths")
        checksums = [name for name in names if name.endswith("checksums.sha256")]
        if len(checksums) != 1:
            raise ValueError("Exactly one source checksums.sha256 required")
        listing = package.read(checksums[0]).decode("utf-8")
        prefix = str(PurePosixPath(checksums[0]).parent)
        by_name: dict[str, str] = {}
        for line in listing.splitlines():
            if not line.strip():
                continue
            digest_hex, sep, rel = line.partition("  ")
            if not sep or len(digest_hex) != 64 or not _safe_zip_name(rel):
                raise ValueError("Malformed original per-file SHA256 listing")
            candidate = str(PurePosixPath(prefix) / rel) if prefix != "." else rel
            if candidate not in names or candidate in by_name:
                raise ValueError("Checksum path missing or repeated")
            by_name[candidate] = digest_hex
        mismatches = []
        for filename, expected in by_name.items():
            h = hashlib.sha256()
            with package.open(filename) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                mismatches.append(filename)
        if mismatches:
            raise ValueError(f"Evidence checksum mismatch: {mismatches[:3]}")
        failures = [name for name in names if name.endswith("failures.json")]
        return {
            "outer_sha256": digest.hexdigest(),
            "zip_bytes": path.stat().st_size,
            "members": len(names),
            "checked_members": len(by_name),
            "unrounded_pt_files": sum(name.endswith(".pt") for name in names),
            "failed_members": [],
            "has_original_failure_log": len(failures) == 1,
            "raw_tensors_still_require_independent_numeric_checks": True,
        }


def render_progress(report: dict[str, Any]) -> str:
    counts = report["status_counts"]
    lines = [
        "# native capabilities Capability Progress",
        "",
        f"Ledger SHA256: \`{report['ledger_sha256']}\`",
        f"Release ready: **{'YES' if report['release_ready'] else 'NO'}**",
        f"Core behavior implemented: **{report['core_pass_count']}/50**",
        f"P8 independent signoff pending: **{report['p8_pending_count']}/50**",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in counts.items())
    lines.extend(
        [
            "",
            "## Outstanding",
            f"{len(report['missing'])} capability gates have not passed final acceptance.",
        ]
    )
    lines.extend(f"- F{row['id']} {row['reason']}" for row in report["missing"])
    return "\n".join(lines) + "\n"
