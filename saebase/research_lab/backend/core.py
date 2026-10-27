"""Read-only, fail-closed access to SAEBase's actual published evidence.

GPU measurements are never synthesized here.  Historical 64-group engineering
checks are *not* silently relabeled as the new ten-dataset/500-prompt corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path("conf/coverage/stage_d_evidence")
PLATFORM = Path(__file__).resolve().parents[1]
CATALOG = PLATFORM / "data" / "catalog.json"
MAX_PUBLIC_INDEX_BYTES = 64 * 1024 * 1024


class InvalidEvidence(ValueError):
    """Public result or its source identity cannot be trusted."""


def data_root() -> Path:
    return Path(
        os.environ.get("OPENSAE_LAB_DATA", str(REPO_ROOT / "runs" / "research-lab"))
    ).resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def has_complete_identity(site: dict[str, Any]) -> bool:
    """Historical CPU acceptances may lack weight digests; never treat that as a real ID."""
    return all(
        isinstance(site.get(key), str)
        and len(site[key]) == 64
        and all(char in "0123456789abcdef" for char in site[key])
        for key in ("model_sha256", "sae_sha256")
    )


def dataset_catalog() -> dict[str, Any]:
    return read_json(CATALOG)


def load_sites(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Join the frozen 48-site matrix to each site's actual original report."""
    base = (repo_root or REPO_ROOT) / EVIDENCE
    matrix_path = base / "final" / "stage_d_final_matrix.json"
    if not matrix_path.is_file():
        raise InvalidEvidence("Frozen historical 48-site matrix is unavailable")
    matrix = read_json(matrix_path)
    rows = matrix.get("rows", [])
    if not isinstance(rows, list):
        raise InvalidEvidence("Malformed verified-model matrix")
    sites = []
    for row in rows:
        rel = row.get("m3m4_summary")
        if not isinstance(rel, str) or not rel.startswith("reports/") or ".." in Path(rel).parts:
            raise InvalidEvidence("Missing or unsafe report reference")
        report_path = base / rel
        if not report_path.is_file():
            raise InvalidEvidence("Missing original report: " + rel)
        report = read_json(report_path)
        for name in ("model_key", "sae_key", "layer", "hook_name"):
            if report.get(name) != row.get(name):
                raise InvalidEvidence("Report/checkpoint mismatch: " + rel + " / " + name)
        if report.get("status") != "passed":
            raise InvalidEvidence("Unverified result: " + rel)
        if not all(
            row.get(flag) is True
            for flag in ("pair_verified", "benchmark_verified", "reliability_verified")
        ):
            raise InvalidEvidence("Unverified site in published matrix")
        values = []
        for metric in report.get("metric_results", []):
            if not isinstance(metric, dict) or not isinstance(metric.get("metric_id"), str):
                raise InvalidEvidence("Malformed metric row")
            values.append(
                {
                    "id": metric["metric_id"],
                    "value": metric.get("value"),
                    "condition": metric.get("condition"),
                    "n_valid": metric.get("n_valid"),
                }
            )
        sites.append(
            {
                "site_id": f"{row['model_key']}:{row['sae_key']}:L{row['layer']}",
                "model": row["model_key"],
                "sae": row["sae_key"],
                "layer": row["layer"],
                "hook": row["hook_name"],
                "model_sha256": row["model_state_sha256"],
                "sae_sha256": row["sae_state_sha256"],
                "status": "verified_engineering",
                "group_count": report.get("n_independent_evaluation_groups"),
                "protocol": report.get("protocol_id"),
                "evidence_url": "https://github.com/OpenSAE/SAEBase/blob/main/"
                + (EVIDENCE / rel).as_posix(),
                "metrics": values,
            }
        )
    return sites


def historical_metrics(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "site_id": site["site_id"],
            "model": site["model"],
            "sae": site["sae"],
            "layer": site["layer"],
            "hook": site["hook"],
            "metric": metric["id"],
            "value": metric["value"],
            "condition": metric["condition"],
            "n_valid": metric["n_valid"],
            "protocol": site["protocol"],
            "data_class": "historical_engineering",
            "evidence_url": site["evidence_url"],
        }
        for site in sites
        for metric in site["metrics"]
    ]


def load_published_dataset(
    dataset_id: str, root: Path | None = None
) -> list[dict[str, Any]] | None:
    """Return exactly the reviewed, checksum-verified 50-row dataset release.

    A missing or unreviewed release is not made public. An incomplete or
    tampered reviewed release raises rather than returning potentially
    unapproved text.
    """
    catalog = {s["id"]: s for s in dataset_catalog()["sources"]}
    if dataset_id not in catalog:
        raise InvalidEvidence("Unknown curated dataset")
    directory = (root or data_root()) / "datasets" / dataset_id
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    meta = read_json(manifest_path)
    if meta.get("schema") != "opensae.dataset_release.v1":
        raise InvalidEvidence("Unrecognized dataset release schema")
    if not (meta.get("license_reviewed") is True and meta.get("human_reviewed") is True):
        return None
    definition = catalog[dataset_id]
    for field, value in (
        ("dataset_id", dataset_id),
        ("source_repo", definition["repo"]),
        ("source_config", definition["config"]),
        ("source_split", definition["split"]),
        ("sample_count", 50),
        ("discovery", 20),
        ("evaluation", 30),
    ):
        if meta.get(field) != value:
            raise InvalidEvidence("Incorrect reviewed dataset identity: " + field)
    revision = meta.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        raise InvalidEvidence("Reviewed dataset requires exact 40-character source commit")
    file_path = directory / "records.jsonl"
    if not file_path.is_file() or file_path.stat().st_size > 16 * 1024 * 1024:
        raise InvalidEvidence("Missing or oversized reviewed dataset records")
    if digest(file_path) != meta.get("records_sha256"):
        raise InvalidEvidence("Reviewed dataset content differs from checksum")
    with file_path.open("r", encoding="utf-8") as fh:
        try:
            rows = [json.loads(line) for line in fh if line.strip()]
        except json.JSONDecodeError as e:
            raise InvalidEvidence("Malformed curated dataset JSONL") from e
    if len(rows) != 50 or any(not isinstance(row, dict) for row in rows):
        raise InvalidEvidence("Expected exactly 50 reviewed prompt objects")
    ids, groups, counts = set(), set(), {"discovery": 0, "evaluation": 0}
    for row in rows:
        sample = row.get("sample_id")
        group = row.get("prompt_group_id")
        if (
            not isinstance(sample, str)
            or not sample.startswith(dataset_id + ":")
            or sample in ids
            or not isinstance(group, str)
            or not group.startswith(dataset_id + ":")
            or group in groups
            or row.get("source_revision") != revision
            or row.get("source_repo") != definition["repo"]
            or row.get("source_split") != definition["split"]
            or not isinstance(row.get("prompt"), str)
            or not row["prompt"].strip()
            or row.get("usage_split") not in counts
        ):
            raise InvalidEvidence("Invalid prompt identity, source, or group in reviewed release")
        ids.add(sample)
        groups.add(group)
        counts[row["usage_split"]] += 1
    if counts != {"discovery": 20, "evaluation": 30}:
        raise InvalidEvidence("Unexpected reviewed dataset splits")
    return rows


def imported_counts(root: Path | None = None) -> dict[str, int]:
    """Count valid published rows, not unchecked booleans in a manifest."""
    base = (root or data_root()) / "datasets"
    counts: dict[str, int] = {}
    if not base.exists():
        return counts
    for manifest in base.glob("*/manifest.json"):
        try:
            rows = load_published_dataset(manifest.parent.name, root)
            if rows is not None:
                counts[manifest.parent.name] = len(rows)
        except (OSError, InvalidEvidence, ValueError, TypeError):
            continue
    return counts


def _verified_index(folder: Path, key: str, resource: str) -> dict[str, Any] | None:
    manifest_path = folder / "manifest.json"
    resource_path = folder / resource
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("schema") != key:
        raise InvalidEvidence("Unexpected release schema")
    if not resource_path.is_file() or resource_path.stat().st_size > MAX_PUBLIC_INDEX_BYTES:
        raise InvalidEvidence("Missing/oversized " + resource)
    if digest(resource_path) != manifest.get("files", {}).get(resource):
        raise InvalidEvidence("Modified published " + resource)
    return read_json(resource_path)


def feature_index(root: Path | None = None) -> list[dict[str, Any]]:
    document = _verified_index(
        (root or data_root()) / "features", "opensae.feature_index.v1", "index.json"
    )
    if document is None:
        return []
    if not isinstance(document, list):
        raise InvalidEvidence("Feature index must be an array")
    return document


def analyses(root: Path | None = None) -> list[dict[str, Any]]:
    document = _verified_index(
        (root or data_root()) / "analyses", "opensae.analysis_index.v1", "index.json"
    )
    if document is None:
        return []
    if not isinstance(document, list):
        raise InvalidEvidence("Analysis index must be an array")
    return document


def published_reports(root: Path | None = None) -> list[dict[str, Any]]:
    """Only report metadata, no full private prompts or arbitrary file serving."""
    folder = (root or data_root()) / "reports"
    if not folder.is_dir():
        return []
    rows = []
    for subfolder in sorted(folder.iterdir()):
        if not subfolder.is_dir() or not _safe_id(subfolder.name):
            continue
        try:
            manifest = read_json(subfolder / "manifest.json")
            report_path = subfolder / "report.json"
            if manifest.get("schema") != "opensae.public_report.v1":
                continue
            if digest(report_path) != manifest.get("report_sha256"):
                continue
            report = read_json(report_path)
            if report.get("visibility") != "public" or report.get("report_id") != subfolder.name:
                continue
            rows.append(
                {
                    "report_id": report["report_id"],
                    "title": report.get("title", "Untitled experiment"),
                    "kind": report.get("kind", "unknown"),
                    "model": report.get("model"),
                    "source": report.get("source"),
                    "summary": report.get("summary", ""),
                    "url": "/api/reports/" + report["report_id"],
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def _safe_id(value: str) -> bool:
    return len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def public_report(report_id: str, root: Path | None = None) -> dict[str, Any] | None:
    if not _safe_id(report_id):
        return None
    folder = (root or data_root()) / "reports" / report_id
    if not (folder / "manifest.json").is_file():
        return None
    manifest = read_json(folder / "manifest.json")
    path = folder / "report.json"
    if manifest.get("schema") != "opensae.public_report.v1" or digest(path) != manifest.get(
        "report_sha256"
    ):
        raise InvalidEvidence("Report integrity check failed")
    result = read_json(path)
    if result.get("report_id") != report_id or result.get("visibility") != "public":
        return None
    return result
