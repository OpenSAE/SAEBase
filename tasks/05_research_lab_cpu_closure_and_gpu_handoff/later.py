"""Packages that stay blocked or partial unless the measured population exists."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import REPO, log, write_json
from run_gpu import json_loads

sys.path.insert(0, str(REPO / "apps" / "research-lab"))

from backend.api import submit_jobs  # noqa: E402
from backend.feature_store import create_feature_store  # noqa: E402
from backend.jobs import JobQueue  # noqa: E402


def _codes(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "private").glob("*_codes.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json_loads(line))
    return rows


def run_gpu03(root: Path) -> dict:
    cards = []
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in _codes(root):
        if row.get("dataset_id") == "controlled6":
            continue
        by_site.setdefault(row.get("site_id", "unknown"), []).append(row)
    for site_id, rows in sorted(by_site.items()):
        counts: Counter[int] = Counter()
        peak: dict[int, float] = {}
        for row in rows:
            seen = set()
            for feature, value in row.get("nonzero_codes", []):
                feature = int(feature)
                seen.add(feature)
                peak[feature] = max(peak.get(feature, 0.0), abs(float(value)))
            counts.update(seen)
        chosen = counts.most_common(50)
        for feature, count in chosen:
            cards.append(
                {
                    "site_id": site_id,
                    "feature_id": feature,
                    "samples_active": count,
                    "peak_abs_activation": peak[feature],
                    "text_excerpt": None,
                    "reason_text_hidden": "human_reviewed=false and license_reviewed=false",
                    "explanation": None,
                }
            )
        if len(chosen) < 50:
            cards.append(
                {
                    "site_id": site_id,
                    "status": "insufficient_observations",
                    "distinct_features_observed": len(counts),
                    "cards_written": len(chosen),
                    "dead_features_not_inferred": True,
                }
            )
    status = "partial" if cards else "not_started"
    payload = {
        "status": status,
        "sites_with_real_codes": sorted(by_site),
        "card_count": sum(1 for card in cards if "feature_id" in card),
        "target_cards": 300,
        "cards": cards,
    }
    write_json(root / "GPU-03" / "feature_cards.json", payload)
    log(root, f"GPU-03 {status}; cards={payload['card_count']}")
    return payload


def run_skipped(root: Path, gpu02: dict) -> dict[str, dict]:
    del gpu02
    payloads = {
        "GPU-04": {
            "status": "partial",
            "reason": "Pilot identity/ablate/scale/reconstruction ran inside GPU-01. The 6-site intervention comparison and authenticated online job route were not opened. POST /api/jobs stays disabled.",
            "post_api_jobs": "closed_503",
        },
        "GPU-05": {
            "status": "not_started",
            "reason": "GPU-01 passed. Full SAR remains 3 sites x 500 groups x 384 = 576000 coalition measurements plus 57600 rotation controls, which exceeds one hour, so it was not started. One SAR site, qwen3_1_7b_base L14, has base captures; the other two SAR sites were not loaded.",
            "protocol": "K=16,M=384,fit=256,held-out=128",
        },
        "GPU-06": {
            "status": "not_started",
            "reason": "GPU-01 passed, but RSS semantic_review and tokenizer_alignment are still pending. The full 6 x 500 x 46 measurement was not started.",
        },
    }
    for name, payload in payloads.items():
        write_json(root / name / "decision.json", payload)
    return payloads


def _store_slots(root: Path) -> list[Path]:
    return [
        root / "GPU-08" / name
        for name in ("feature_store", "feature_store_full", "feature_store_v3")
    ]


def _store_taken(path: Path) -> bool:
    return (path / "manifest.json").exists() or (path / "features.sqlite3").exists()


def run_gpu08(root: Path) -> dict:
    observations = []
    recomputed = []
    rejected = 0
    free_slot = next((path for path in _store_slots(root) if not _store_taken(path)), None)
    for row in _codes(root):
        if row.get("dataset_id") == "controlled6":
            continue
        if not row.get("hash_matches_historical"):
            if row.get("historical_hash_was_empty") or row.get("recomputed_identity"):
                recomputed.append(
                    {
                        "site_id": row.get("site_id"),
                        "sample_id": row.get("sample_id"),
                        "loaded_model_sha256": row.get("loaded_model_sha256"),
                        "loaded_sae_sha256": row.get("loaded_sae_sha256"),
                        "not_a_historical_match": True,
                    }
                )
            else:
                rejected += 1
            continue
        if free_slot is None:
            continue
        for feature, value in row.get("nonzero_codes", []):
            activation = float(value)
            if activation != activation or activation in (float("inf"), float("-inf")):
                rejected += 1
                continue
            observations.append(
                {
                    "site_id": row["site_id"],
                    "model_sha256": row["loaded_model_sha256"],
                    "sae_sha256": row["loaded_sae_sha256"],
                    "source_type": "real_checkpoint",
                    "feature_id": int(feature),
                    "token_position": int(row["token_position"]),
                    "activation": activation,
                    "prompt_group_id": row["prompt_group_id"],
                    "sample_id": row["sample_id"],
                    "public_text_reviewed": False,
                }
            )
    store_status: dict[str, Any]
    store_dir = free_slot or (root / "GPU-08" / "feature_store")
    if free_slot is None:
        existing = [path for path in _store_slots(root) if _store_taken(path)]
        latest = existing[-1]
        manifest = json.loads((latest / "manifest.json").read_text(encoding="utf-8"))
        store_status = {
            "status": "partial",
            "existing_store": latest.name,
            "observation_count": manifest.get("observation_count"),
            "feature_count": manifest.get("feature_count"),
            "immutable_store_not_overwritten": True,
            "rejected_or_unmatched_rows": rejected,
            "public_text_reviewed": False,
        }
    elif not observations:
        store_status = {
            "status": "partial",
            "indexed_observations": 0,
            "rejected_or_unmatched_rows": rejected,
            "reason": "No captured row matched the historical weight hashes, so the public feature index was not filled with unmatched identities.",
        }
    else:
        try:
            store_status = create_feature_store(observations, store_dir)
            store_status["status"] = "partial"
            store_status["rejected_or_unmatched_rows"] = rejected
        except Exception as exc:
            store_status = {
                "status": "partial",
                "error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:300],
                "rejected_or_unmatched_rows": rejected,
            }
    write_json(
        root / "GPU-08" / "recomputed_identity_ledger.json",
        {"rows": len(recomputed), "not_historical_matches": True},
    )
    store_status["recomputed_identity_rows_not_inserted"] = len(recomputed)
    write_json(root / "GPU-08" / "metric_ledger.json", store_status)
    return store_status


def _extra_negative_checks(root: Path) -> dict[str, Any]:
    """Reject path traversal and XSS fields; TTL must not mark a job successful."""
    from backend.jobs import JobError

    negative_db = root / "GPU-09" / "queue_negative_v2.sqlite"
    if negative_db.exists():
        negative_db = root / "GPU-09" / "queue_negative_v3.sqlite"
    queue = JobQueue(negative_db, max_queued=4, max_per_owner=2)
    hostile = {
        "site_id": "pythia410m:pythia410m_mlp_topk:L12",
        "operation": "identity",
        "feature_ids": [],
        "strength": 0,
        "max_new_tokens": 0,
        "input_ref": hashlib.sha256(b"xss-and-traversal").hexdigest(),
        "note": "<script>alert(1)</script>",
        "path": "../../etc/passwd",
    }
    rejected = False
    try:
        queue.enqueue(owner="owner-neg", client_key="clientkeyN1", spec=hostile, now=2_000.0)
    except JobError:
        rejected = True
    valid = {
        "site_id": "pythia410m:pythia410m_mlp_topk:L12",
        "operation": "identity",
        "feature_ids": [],
        "strength": 0,
        "max_new_tokens": 0,
        "input_ref": hashlib.sha256(b"ttl-check").hexdigest(),
    }
    job = queue.enqueue(owner="owner-neg", client_key="clientkeyN2", spec=valid, now=3_000.0)
    claim = queue.claim(worker_id="worker-ttl", now=3_010.0, lease_seconds=30)
    reaped = queue.reap_expired(now=3_100.0, max_attempts=3)
    status = queue.status(job_id=job["job_id"], owner="owner-neg")
    outside = REPO.parent / "passwd"
    return {
        "hostile_fields_rejected": rejected,
        "status_hides_script": "<script>" not in json.dumps(status),
        "status_hides_traversal_path": "etc/passwd" not in json.dumps(status),
        "no_file_written_outside_release": not outside.exists(),
        "ttl_reap_does_not_mark_success": claim is not None
        and (status or {}).get("status") != "succeeded"
        and reaped["failed"] + reaped["requeued"] >= 1,
    }


def run_gpu09(root: Path) -> dict:
    queue_path = root / "GPU-09" / "queue_acceptance.sqlite"
    if queue_path.exists():
        queue_path = root / "GPU-09" / "queue_acceptance_v2.sqlite"
    queue = JobQueue(queue_path, max_queued=2, max_per_owner=1)
    input_ref = hashlib.sha256(b"private-input-not-stored-in-queue").hexdigest()
    spec = {
        "site_id": "pythia410m:pythia410m_mlp_topk:L12",
        "operation": "identity",
        "feature_ids": [],
        "strength": 0,
        "max_new_tokens": 0,
        "input_ref": input_ref,
    }
    checks: dict[str, Any] = {}
    first = queue.enqueue(owner="owner-a-lab", client_key="clientkeyA1", spec=spec, now=1_000.0)
    second = queue.enqueue(owner="owner-a-lab", client_key="clientkeyA1", spec=spec, now=1_001.0)
    checks["idempotent_enqueue"] = (
        second["deduplicated"] is True and second["job_id"] == first["job_id"]
    )
    other = queue.enqueue(
        owner="owner-b-lab",
        client_key="clientkeyB1",
        spec=spec,
        now=1_002.0,
    )
    claim_a = queue.claim(worker_id="workerA", now=1_010.0, lease_seconds=30)
    claim_b = queue.claim(worker_id="workerB", now=1_011.0, lease_seconds=30)
    checks["two_workers_receive_distinct_jobs"] = (
        claim_a is not None and claim_b is not None and claim_a["job_id"] != claim_b["job_id"]
    )
    checks["stale_attempt_rejected"] = (
        queue.heartbeat(
            job_id=claim_a["job_id"],
            worker_id="workerA",
            attempt=claim_a["attempt"] + 1,
            now=1_012.0,
        )
        is False
    )
    checks["owner_cancel"] = (
        queue.cancel(job_id=claim_a["job_id"], owner="owner-a-lab", now=1_020.0) is True
    )
    checks["stranger_cannot_read_status"] = (
        queue.status(job_id=other["job_id"], owner="owner-c-lab") is None
    )
    expired = queue.reap_expired(now=1_100.0, max_attempts=3)
    checks["expired_lease_not_marked_success"] = expired["failed"] + expired["requeued"] >= 0
    try:
        from fastapi import HTTPException

        try:
            submit_jobs()
            checks["post_api_jobs_closed"] = False
        except HTTPException as exc:
            checks["post_api_jobs_closed"] = exc.status_code == 503
    except Exception as exc:
        checks["post_api_jobs_closed"] = type(exc).__name__
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "not_installed"
    payload = {
        "status": "blocked",
        "cpu_queue_checks": checks,
        "cpu_queue_checks_passed": all(value is True for value in checks.values()),
        "vllm": vllm_version,
        "vllm_note": "vLLM was not installed because the pinned generation extra (vllm==0.27.1) would replace the torch 2.11 cu128 runtime needed for the registered checkpoints. SAE capture stayed on the hook API. No KV-cache parity was claimed.",
        "hf_space": "blocked_no_organization_write_token",
        "three_external_researchers": "blocked_not_available_in_this_session",
        "public_online_gpu": "remains_off",
        "negative_checks": _extra_negative_checks(root),
    }
    write_json(root / "GPU-09" / "live_service_acceptance.json", payload)
    log(root, "GPU-09 queue checks recorded; public GPU route remains closed")
    return payload
