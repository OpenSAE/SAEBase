"""Opt-in private job API on top of SAEBase's existing durable SQLite queue.

Disabled on the public Research Lab. Uses separate operator/worker bearer
secrets and a durable private data directory. No user text enters the queue.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .jobs import JobError, JobQueue

router = APIRouter(prefix="/api/private", tags=["private-compute"])


def authorize(role: str, bearer: str | None) -> Path:
    if os.environ.get("OPENSAE_ENABLE_PRIVATE_JOBS") != "1":
        raise HTTPException(503, "Private worker disabled")
    root = os.environ.get("OPENSAE_PRIVATE_DATA")
    operator = os.environ.get("OPENSAE_PRIVATE_OPERATOR_TOKEN")
    worker = os.environ.get("OPENSAE_PRIVATE_WORKER_TOKEN")
    if (
        not root
        or not Path(root).is_absolute()
        or not operator
        or len(operator) < 32
        or not worker
        or len(worker) < 32
        or operator == worker
    ):
        raise HTTPException(503, "Durable storage and distinct strong secrets required")
    expected = operator if role == "operator" else worker
    if (
        not isinstance(bearer, str)
        or not bearer.startswith("Bearer ")
        or not secrets.compare_digest(bearer[7:], expected)
    ):
        raise HTTPException(401, "Unauthorized")
    directory = Path(root)
    if not directory.is_dir() or directory.is_symlink():
        raise HTTPException(503, "Private durable storage is unavailable")
    return directory


def get_queue(root: Path) -> JobQueue:
    return JobQueue(root / "queue" / "jobs.sqlite3", max_queued=16, max_per_owner=2)


def input_path(root: Path, digest: str) -> Path:
    if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
        raise ValueError("Invalid private input digest")
    return root / "inputs" / (digest + ".json")


def load_input(root: Path, digest: str) -> str:
    file = input_path(root, digest)
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 65536:
        raise FileNotFoundError("Unreachable private input")
    data = json.loads(file.read_text(encoding="utf-8"))["text"]
    if not isinstance(data, str) or hashlib.sha256(data.encode()).hexdigest() != digest:
        raise ValueError("Private prompt checksum mismatch")
    return data


class PrivateInput(BaseModel):
    text: str = Field(min_length=1, max_length=16000)


class PrivateJob(BaseModel):
    client_key: str
    site_id: str
    operation: str
    feature_ids: list[int]
    strength: float = 1.0
    max_new_tokens: int = Field(default=0, ge=0, le=64)
    input_ref: str


class Completion(BaseModel):
    job_id: str
    worker_id: str
    attempt: int
    result_sha256: str | None = None
    error_code: str | None = None


@router.post("/inputs")
def store_input(
    body: PrivateInput, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    root = authorize("operator", authorization)
    digest = hashlib.sha256(body.text.encode()).hexdigest()
    file = input_path(root, digest)
    file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not file.exists():
        with file.open("x", encoding="utf-8") as stream:
            json.dump({"text": body.text}, stream)
        os.chmod(file, 0o600)
    return {"input_ref": digest, "private": True}


@router.post("/jobs")
def enqueue(body: PrivateJob, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    root = authorize("operator", authorization)
    if body.max_new_tokens and not (
        body.operation in {"identity", "ablate", "scale"} and 16 <= body.max_new_tokens <= 32
    ):
        raise HTTPException(
            422, "Generation jobs use 16–32 new tokens with identity, ablate, or scale"
        )
    try:
        load_input(root, body.input_ref)
        return get_queue(root).enqueue(
            owner="single-operator",
            client_key=body.client_key,
            spec=body.model_dump(exclude={"client_key"}),
        )
    except (JobError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/jobs/{job_id}")
def status(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    root = authorize("operator", authorization)
    result = get_queue(root).status(job_id=job_id, owner="single-operator")
    if result is None:
        raise HTTPException(404, "Unknown private job")
    return result


@router.get("/jobs/{job_id}/result")
def private_result(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Return only the authenticated operator's hash-verified native output."""
    root = authorize("operator", authorization)
    status = get_queue(root).status(job_id=job_id, owner="single-operator")
    if status is None:
        raise HTTPException(404, "Unknown private job")
    if status["status"] != "succeeded":
        raise HTTPException(409, "Private job has not produced a committed result")
    if len(job_id) != 40 or any(ch not in "0123456789abcdef" for ch in job_id):
        raise HTTPException(404, "Invalid private job identity")
    file = root / "results" / (job_id + ".json")
    if not file.is_file() or file.is_symlink() or file.stat().st_size > 10 * 1024 * 1024:
        raise HTTPException(503, "Private result file is missing or oversized")
    payload = file.read_bytes()
    if hashlib.sha256(payload).hexdigest() != status["result_sha"]:
        raise HTTPException(503, "Private result checksum mismatch")
    try:
        record = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(503, "Private result is not valid JSON") from exc
    if not isinstance(record, dict) or record.get("job_id") != job_id:
        raise HTTPException(503, "Private result belongs to a different job")
    return record


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    root = authorize("operator", authorization)
    return {"cancelled": get_queue(root).cancel(job_id=job_id, owner="single-operator")}


@router.post("/workers/claim")
def claim(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    root = authorize("worker", authorization)
    return {"job": get_queue(root).claim(worker_id="opensaebase-worker", lease_seconds=600)}


@router.post("/workers/finish")
def finish(body: Completion, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    root = authorize("worker", authorization)
    try:
        accepted = get_queue(root).finish(**body.model_dump())
    except JobError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"accepted": accepted}
