"""Purpose: Save site attempts, acceptance records and resumable status, then build the final evaluation matrix.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from saebase.api.sae_profiles import get_profile
from tasks.benchmark.results.metrics import atomic_json
from tasks.benchmark.sites.catalog import (
    DEFAULT_MODEL_ORDER,
    HEX64,
    PRIOR_PAIR_VERIFIED,
    PROTOCOL_PAIR,
    PROTOCOL_SMOKE,
    SCHEMA_ACCEPTANCE,
    SCHEMA_FINAL,
    SCHEMA_STATUS,
    SCHEMA_TARGETS,
    CoverageCatalog,
    SiteRecord,
    build_site_records,
    load_coverage,
)


def write_site_evaluation_targets(
    path: Path, catalog: CoverageCatalog | None = None
) -> dict[str, Any]:
    """Materialize the Stage D overlay from the B/C catalog."""
    if catalog is None:
        root = Path(__file__).resolve().parents[3]
        catalog = load_coverage(root / "conf" / "coverage" / "stage_bc.json")
    records = build_site_records(catalog)
    payload = {
        "schema": SCHEMA_TARGETS,
        "source_catalog_schema": "saebase.coverage.stage_bc.v1",
        "join_keys": ["model_key", "sae_key", "layer", "hook_name", "checkpoint_sha"],
        "prior_pair_verified": [
            {"model_key": m, "sae_key": s, "layer": layer}
            for m, s, layer in sorted(PRIOR_PAIR_VERIFIED)
        ],
        "model_order": list(DEFAULT_MODEL_ORDER),
        "n_sites": len(records),
        "sites": [r.to_dict() for r in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def load_site_evaluation_targets(path: Path) -> list[SiteRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_TARGETS:
        raise ValueError("invalid stage_d_targets schema")
    records = [SiteRecord.from_dict(x) for x in raw["sites"]]
    if len(records) != 48:
        raise ValueError(f"stage_d_targets must list 48 sites, got {len(records)}")
    return records


def acceptance_path(output: Path, model_key: str, sae_key: str, layer: int) -> Path:
    return output / "acceptance" / model_key / sae_key / f"L{layer}.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_under(output: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(output.resolve()))
    except Exception:
        return str(p)


def rebuild_canonical_status(
    output: Path, records: list[SiteRecord] | None = None
) -> dict[str, Any]:
    """Rewrite status.json to exactly the 48 catalog keys; archive extras."""
    if records is None:
        root = Path(__file__).resolve().parents[3]
        records = build_site_records(load_coverage(root / "conf" / "coverage" / "stage_bc.json"))
    expected = {r.key.id: r for r in records}
    if len(expected) != 48:
        raise ValueError(f"expected 48 sites, got {len(expected)}")
    old = load_status(output)
    sites: dict[str, Any] = {}
    for site_id, rec in expected.items():
        prev = dict(old.get("sites", {}).get(site_id, {}))
        acceptance = acceptance_path(output, rec.model_key, rec.sae_key, rec.layer)
        pair_ok = False
        if acceptance.exists():
            try:
                raw = read_acceptance(acceptance)
                pair_ok = raw.get("status") == "passed"
            except ValueError:
                pair_ok = False
        if not pair_ok and rec.prior_pair_verified:
            pair_ok = True
            prev.setdefault("status", "skipped_prior")
            prev.setdefault("pair_verified", True)
            prev.setdefault("prior_pair_verified", True)
        if pair_ok:
            prev["pair_verified"] = True
            if prev.get("status") in {None, "registered", "sae_only_verified"}:
                prev["status"] = (
                    "pair_verified"
                    if not rec.prior_pair_verified
                    else prev.get("status", "skipped_prior")
                )
            if rec.prior_pair_verified and prev.get("status") == "registered":
                prev["status"] = "skipped_prior"
        for key in ("acceptance_path", "m3m4_summary"):
            if key in prev:
                prev[key] = relative_under(output, prev[key])
        sites[site_id] = prev
    extras = {k: v for k, v in old.get("sites", {}).items() if k not in expected}
    if extras:
        archive = {
            "schema": "saebase.stage_d.status_archive.v1",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "sites": extras,
        }
        atomic_json(output / "status_archive_extras.json", archive)
        append_attempt(
            output,
            {
                "event": "status_rebuild",
                "archived_extra_keys": sorted(extras),
                "n_canonical": 48,
            },
        )
    payload = {
        "schema": SCHEMA_STATUS,
        "sites": sites,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_sites": 48,
    }
    atomic_json(output / "status.json", payload)
    return payload


def write_checksums_manifest(root: Path, paths: Iterable[Path], dest: Path) -> Path:
    """Write deterministic sha256sum-compatible listing (excludes dest itself)."""
    root = root.resolve()
    dest = dest.resolve()
    rows: list[tuple[str, str]] = []
    for path in paths:
        p = path.resolve()
        if p == dest or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        rows.append((file_sha256(p), rel))
    rows.sort(key=lambda x: x[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{digest}  {rel}\n" for digest, rel in rows)
    dest.write_text(text, encoding="utf-8")
    return dest


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    encoded = json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def redact_secrets(message: str) -> str:
    text = message
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPEN_BUTTON_TOKEN", "WEB_PASSWORD"):
        token = os.environ.get(key)
        if token:
            text = text.replace(token, "[redacted]")
    return text


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in {"GatedRepoError", "RepositoryNotFoundError"} or "401" in msg or "403" in msg:
        return "blocked_access"
    if "out of memory" in msg or "cuda error: out of memory" in msg or name == "OutOfMemoryError":
        return "blocked_compute"
    if name in {"ImportError", "ModuleNotFoundError", "UnsupportedCapability"}:
        return "blocked_loader"
    if (
        "could not load any model candidate" in msg
        or "not found. valid official model names" in msg
    ):
        return "blocked_loader"
    if name in {"ValueError", "AssertionError", "RuntimeError"} and (
        "identity" in msg or "dimension" in msg or "nonfinite" in msg or "feature" in msg
    ):
        return "failed_numeric"
    if "sha" in msg or "integrity" in msg or "manifest" in msg:
        return "failed_integrity"
    if name in {"OSError", "HTTPError", "HfHubHTTPError"}:
        return (
            "blocked_access" if any(x in msg for x in ("401", "403", "gated")) else "blocked_loader"
        )
    return "failed_numeric"


def append_attempt(output: Path, record: dict[str, Any]) -> None:
    path = output / "attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def load_status(output: Path) -> dict[str, Any]:
    path = output / "status.json"
    if not path.exists():
        return {
            "schema": SCHEMA_STATUS,
            "sites": {},
            "updated_at_utc": None,
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_STATUS:
        raise ValueError("corrupt status.json schema")
    return raw


def update_status(output: Path, site_id: str, patch: dict[str, Any]) -> None:
    status = load_status(output)
    current = dict(status["sites"].get(site_id, {}))
    current.update(patch)
    status["sites"][site_id] = current
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "status.json", status)


def read_acceptance(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_ACCEPTANCE:
        raise ValueError(f"invalid acceptance schema at {path}")
    digest = raw.get("artifact_sha256")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        raise ValueError("acceptance artifact missing SHA-256")
    expected = canonical_payload_sha256(raw)
    if digest != expected:
        raise ValueError("acceptance artifact checksum mismatch")
    return raw


def resume_fingerprint_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match for a source-aware resume skip."""
    return {
        "model_key": record.get("model_key"),
        "evaluation_host_model_key": record.get("evaluation_host_model_key")
        or record.get("model_key"),
        "sae_key": record.get("sae_key"),
        "layer": record.get("layer"),
        "hook_name": record.get("hook_name"),
        "model_state_sha256": record.get("model_state_sha256"),
        "sae_state_sha256": record.get("sae_state_sha256"),
        "model_weight_dtype": record.get("model_weight_dtype") or record.get("precision"),
        "sae_weight_dtype": record.get("sae_weight_dtype") or record.get("precision"),
        "protocol": record.get("protocol") or PROTOCOL_PAIR,
        "dataset_sha256": record.get("dataset_sha256"),
        "git_commit": record.get("git_commit"),
        "loaded_model_name": record.get("loaded_model_name"),
    }


def should_skip_resume(
    output: Path,
    *,
    model_key: str,
    sae_key: str,
    layer: int,
    resume: bool,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return existing acceptance record when resume fingerprint still matches."""
    if not resume:
        return None
    path = acceptance_path(output, model_key, sae_key, layer)
    if not path.exists():
        return None
    try:
        record = read_acceptance(path)
    except ValueError:
        return None
    if record.get("status") != "passed":
        return None
    if (
        record.get("model_key") != model_key
        or record.get("sae_key") != sae_key
        or record.get("layer") != layer
    ):
        return None
    if expected:
        got = resume_fingerprint_fields(record)
        for key, want in expected.items():
            if want is None:
                continue
            if got.get(key) != want:
                append_attempt(
                    output,
                    {
                        "event": "resume_fingerprint_mismatch",
                        "site_id": f"{model_key}|{sae_key}|L{layer}",
                        "field": key,
                        "expected": want,
                        "found": got.get(key),
                    },
                )
                return None
    return record


def write_acceptance(output: Path, record: dict[str, Any]) -> Path:
    path = acceptance_path(
        output,
        record["model_key"],
        record["sae_key"],
        int(record["layer"]),
    )
    if "artifact_sha256" not in record:
        record = dict(record)
        record["artifact_sha256"] = canonical_payload_sha256(record)
    atomic_json(path, record)
    # Re-read validates checksum contract.
    read_acceptance(path)
    return path


def failure_record(
    *,
    sae_key: str,
    layer: int,
    device: str,
    exc: BaseException,
    model_key: str | None = None,
) -> dict[str, Any]:
    profile = get_profile(sae_key)
    status = classify_exception(exc)
    payload = {
        "schema": SCHEMA_ACCEPTANCE,
        "status": status,
        "model_key": model_key or profile.model_key,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": profile.hook_name(layer) if layer in profile.layers else None,
        "device": device,
        "error_type": type(exc).__name__,
        "error": redact_secrets(str(exc))[:1500],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_weights_validated": False,
    }
    payload["artifact_sha256"] = canonical_payload_sha256(payload)
    return payload


def _pair_verified_from_evidence(
    output: Path,
    rec: SiteRecord,
    site_status: dict[str, Any],
) -> bool:
    """Derive pair_verified only from acceptance artifacts or prior CPU proof."""
    if rec.prior_pair_verified:
        return True
    path_raw = site_status.get("acceptance_path")
    candidates: list[Path] = []
    if path_raw:
        p = Path(path_raw)
        candidates.append(p if p.is_absolute() else output / p)
    candidates.append(acceptance_path(output, rec.model_key, rec.sae_key, rec.layer))
    for path in candidates:
        if path.exists():
            try:
                raw = read_acceptance(path)
            except ValueError:
                continue
            if raw.get("status") == "passed" and raw.get("model_key") == rec.model_key:
                return True
    return False


def build_final_matrix(output: Path, records: list[SiteRecord]) -> dict[str, Any]:
    status = load_status(output)
    if set(status.get("sites", {})) - {r.key.id for r in records}:
        raise ValueError(
            "status.json contains non-canonical site keys; run rebuild_canonical_status"
        )
    if len(records) != 48:
        raise ValueError(f"final matrix requires 48 site records, got {len(records)}")
    rows = []
    counts: dict[str, int] = {}
    for rec in records:
        site_status = status["sites"].get(rec.key.id, {})
        state = site_status.get("status", rec.status)
        pair_ok = _pair_verified_from_evidence(output, rec, site_status)
        counts[state] = counts.get(state, 0) + 1
        rows.append(
            {
                "model_key": rec.model_key,
                "sae_key": rec.sae_key,
                "layer": rec.layer,
                "hook_name": rec.hook_name,
                "access": rec.access,
                "execution_wave": rec.execution_wave,
                "prior_pair_verified": rec.prior_pair_verified,
                "status": state,
                "pair_verified": pair_ok,
                "benchmark_verified": bool(site_status.get("benchmark_verified")),
                "reliability_verified": bool(site_status.get("reliability_verified")),
                "protocol": site_status.get("protocol"),
                "acceptance_path": relative_under(output, site_status.get("acceptance_path")),
                "m3m4_summary": relative_under(output, site_status.get("m3m4_summary")),
                "error_type": site_status.get("error_type"),
                "error": site_status.get("error"),
                "sae_state_sha256": site_status.get("sae_state_sha256"),
                "model_state_sha256": site_status.get("model_state_sha256"),
            }
        )
    n_pair = sum(1 for r in rows if r["pair_verified"])
    n_pop = sum(1 for r in rows if r.get("protocol") == "population" and r["reliability_verified"])
    n_smoke = sum(
        1
        for r in rows
        if r.get("protocol") in {None, "smoke_only", PROTOCOL_SMOKE} and r["reliability_verified"]
    )
    complete = n_pair == 48 and n_pop == 48
    return {
        "schema": SCHEMA_FINAL,
        "n_sites": len(rows),
        "counts": counts,
        "n_pair_verified": n_pair,
        "n_population_m3m4": n_pop,
        "n_smoke_only_m3m4": n_smoke,
        "stage_d_complete": complete,
        "rows": rows,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
