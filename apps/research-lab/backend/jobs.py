"""Durable CPU job state machine for a future authenticated GPU worker.

This module deliberately stores no prompt text, only an opaque SHA-256
reference to a separately stored private input. It has no public HTTP
endpoint and will not execute model code. Authentication, privacy controls
and GPU queue activation are independent, mandatory deployment gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .core import canonical_json, has_complete_identity, load_sites


class JobError(ValueError):
    pass


def _sha256(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _owner_digest(owner: str) -> str:
    if not isinstance(owner, str) or not 3 <= len(owner) <= 128:
        raise JobError("A verified internal owner identity is required")
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()


class JobQueue:
    """SQLite-backed lifecycle; caller must supply authenticated owner."""

    def __init__(self, path: Path, *, max_queued: int = 16, max_per_owner: int = 1):
        if not 1 <= max_queued <= 1000 or not 1 <= max_per_owner <= 10:
            raise JobError("Queue budget outside supported bounds")
        self.path = Path(path)
        self.max_queued = max_queued
        self.max_per_owner = max_per_owner
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.path.exists():
            self.path.touch(mode=0o600)
        os.chmod(self.path, 0o600)
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_sha TEXT NOT NULL,
                    client_key TEXT NOT NULL,
                    request_sha TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN
                        ('queued','running','succeeded','failed','cancelled')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_until REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    result_sha TEXT,
                    error_code TEXT,
                    UNIQUE(owner_sha, client_key)
                );
                CREATE INDEX IF NOT EXISTS jobs_status_created
                  ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS jobs_owner_status
                  ON jobs(owner_sha, status);
                """
            )
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _validated_spec(spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise JobError("Expected explicit job specification")
        expected = {
            "site_id",
            "operation",
            "feature_ids",
            "strength",
            "max_new_tokens",
            "input_ref",
        }
        if set(spec) != expected:
            raise JobError("Unexpected job fields; raw prompt text is forbidden")
        if not _sha256(spec["input_ref"]):
            raise JobError("Input must be a pinned private content hash")
        sites = {site["site_id"]: site for site in load_sites()}
        site = sites.get(spec["site_id"])
        if site is None or not has_complete_identity(site):
            raise JobError("Worker site must have complete original weight hashes")
        if spec["operation"] not in {
            "identity",
            "ablate",
            "scale",
            "clamp",
            "add",
            "coalition",
            "reconstruction",
            "inspect",
        }:
            raise JobError("Unknown feature intervention")
        features = spec["feature_ids"]
        if (
            not isinstance(features, list)
            or len(features) > 8
            or any(type(item) is not int or item < 0 for item in features)
            or len(features) != len(set(features))
        ):
            raise JobError("Invalid feature selection")
        if spec["operation"] in ("identity", "reconstruction", "inspect"):
            if features:
                raise JobError("Identity/reconstruction conditions select no feature IDs")
        elif not features:
            raise JobError("Feature editing requires at least one feature ID")
        strength = spec["strength"]
        if (
            type(strength) not in (float, int)
            or not math.isfinite(strength)
            or not -10 <= strength <= 10
        ):
            raise JobError("Invalid edit strength")
        tokens = spec["max_new_tokens"]
        if spec["operation"] == "inspect" and (features or tokens != 0 or strength != 0):
            raise JobError("Recorded prompt inspection has no feature edits or generation")
        if type(tokens) is not int or not 0 <= tokens <= 64:
            raise JobError("Invalid token budget")
        return spec

    def enqueue(
        self,
        *,
        owner: str,
        client_key: str,
        spec: dict[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]:
        """Store private input reference only; do not load or run the model."""
        if not isinstance(client_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", client_key):
            raise JobError("Invalid idempotency key")
        payload = self._validated_spec(spec)
        owner_sha = _owner_digest(owner)
        request_sha = hashlib.sha256(canonical_json(payload)).hexdigest()
        job_id = hashlib.sha256(
            canonical_json({"owner": owner_sha, "client": client_key, "request": request_sha})
        ).hexdigest()[:40]
        clock = time.time() if now is None else float(now)
        with self._transaction() as con:
            duplicate = con.execute(
                "SELECT job_id,request_sha,status FROM jobs WHERE owner_sha=? AND client_key=?",
                (owner_sha, client_key),
            ).fetchone()
            if duplicate:
                if duplicate["request_sha"] != request_sha:
                    raise JobError("Idempotency key was already used with different inputs")
                return {
                    "job_id": duplicate["job_id"],
                    "status": duplicate["status"],
                    "deduplicated": True,
                }
            busy = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            mine = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE owner_sha=? AND status IN ('queued','running')",
                (owner_sha,),
            ).fetchone()[0]
            if busy >= self.max_queued or mine >= self.max_per_owner:
                raise JobError("Active job quota reached")
            con.execute(
                """INSERT INTO jobs
                (job_id,owner_sha,client_key,request_sha,spec_json,status,created_at,updated_at)
                VALUES (?,?,?,?,?,'queued',?,?)""",
                (
                    job_id,
                    owner_sha,
                    client_key,
                    request_sha,
                    canonical_json(payload).decode(),
                    clock,
                    clock,
                ),
            )
        return {"job_id": job_id, "status": "queued", "deduplicated": False}

    def claim(
        self, *, worker_id: str, now: float | None = None, lease_seconds: int = 300
    ) -> dict[str, Any] | None:
        """Atomic worker claim, never called by an unauthenticated HTTP route."""
        if not isinstance(worker_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", worker_id):
            raise JobError("Invalid worker identity")
        if not 30 <= lease_seconds <= 3600:
            raise JobError("Invalid worker lease duration")
        clock = time.time() if now is None else float(now)
        with self._transaction() as con:
            row = con.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            attempts = row["attempts"] + 1
            con.execute(
                """UPDATE jobs SET status='running',attempts=?,worker_id=?,
                lease_until=?,updated_at=? WHERE job_id=?""",
                (attempts, worker_id, clock + lease_seconds, clock, row["job_id"]),
            )
            return {
                "job_id": row["job_id"],
                "attempt": attempts,
                "request_sha256": row["request_sha"],
                "spec": json.loads(row["spec_json"]),
                "lease_until": clock + lease_seconds,
            }

    def heartbeat(
        self,
        *,
        job_id: str,
        worker_id: str,
        attempt: int,
        now: float | None = None,
        lease_seconds: int = 300,
    ) -> bool:
        if type(attempt) is not int or attempt < 1:
            raise JobError("Invalid worker attempt fencing token")
        if not 30 <= lease_seconds <= 3600:
            raise JobError("Invalid lease duration")
        clock = time.time() if now is None else float(now)
        with self._transaction() as con:
            row = con.execute(
                "SELECT status,worker_id,lease_until,attempts FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["worker_id"] != worker_id
                or row["attempts"] != attempt
                or row["lease_until"] < clock
            ):
                return False
            con.execute(
                "UPDATE jobs SET lease_until=?,updated_at=? WHERE job_id=?",
                (clock + lease_seconds, clock, job_id),
            )
            return True

    def finish(
        self,
        *,
        job_id: str,
        worker_id: str,
        attempt: int,
        result_sha256: str | None = None,
        error_code: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Worker reports a committed result digest, never arbitrary output text."""
        if type(attempt) is not int or attempt < 1:
            raise JobError("Invalid worker attempt fencing token")
        if (result_sha256 is None) == (error_code is None):
            raise JobError("Provide exactly one of result_sha256 or a sanitized error code")
        if result_sha256 is not None and not _sha256(result_sha256):
            raise JobError("Expected exact 64-character committed result hash")
        if error_code is not None and not re.fullmatch(r"[A-Z_]{3,40}", error_code):
            raise JobError("Error code must be a fixed sanitized category")
        clock = time.time() if now is None else float(now)
        with self._transaction() as con:
            row = con.execute(
                "SELECT status,worker_id,lease_until,attempts FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["worker_id"] != worker_id
                or row["attempts"] != attempt
                or row["lease_until"] < clock
            ):
                return False
            con.execute(
                """UPDATE jobs SET status=?,worker_id=NULL,lease_until=NULL,
                result_sha=?,error_code=?,updated_at=? WHERE job_id=?""",
                (
                    "succeeded" if result_sha256 is not None else "failed",
                    result_sha256,
                    error_code,
                    clock,
                    job_id,
                ),
            )
            return True

    def reap_expired(self, *, now: float | None = None, max_attempts: int = 3) -> dict[str, int]:
        """Retry jobs whose worker lease expired, but never mark them successful."""
        if not 1 <= max_attempts <= 10:
            raise JobError("Invalid retry budget")
        clock = time.time() if now is None else float(now)
        recovered, failed = 0, 0
        with self._transaction() as con:
            rows = con.execute(
                """SELECT job_id,attempts FROM jobs WHERE status='running'
                AND lease_until < ? ORDER BY job_id""",
                (clock,),
            ).fetchall()
            for row in rows:
                exhausted = row["attempts"] >= max_attempts
                con.execute(
                    """UPDATE jobs SET status=?,worker_id=NULL,lease_until=NULL,
                    error_code=?,updated_at=? WHERE job_id=?""",
                    (
                        "failed" if exhausted else "queued",
                        "WORKER_LEASE_EXPIRED" if exhausted else None,
                        clock,
                        row["job_id"],
                    ),
                )
                recovered += 0 if exhausted else 1
                failed += 1 if exhausted else 0
        return {"requeued": recovered, "failed": failed}

    def cancel(self, *, job_id: str, owner: str, now: float | None = None) -> bool:
        """Cancellation changes the durable state; worker must stop on check."""
        clock = time.time() if now is None else float(now)
        owner_sha = _owner_digest(owner)
        with self._transaction() as con:
            row = con.execute(
                "SELECT status FROM jobs WHERE job_id=? AND owner_sha=?", (job_id, owner_sha)
            ).fetchone()
            if row is None or row["status"] not in ("queued", "running"):
                return False
            con.execute(
                """UPDATE jobs SET status='cancelled',worker_id=NULL,
                lease_until=NULL,updated_at=? WHERE job_id=?""",
                (clock, job_id),
            )
            return True

    def status(self, *, job_id: str, owner: str) -> dict[str, Any] | None:
        """Only the owner can inspect status; no prompt content is stored."""
        owner_sha = _owner_digest(owner)
        with self._transaction() as con:
            row = con.execute(
                "SELECT job_id,status,attempts,result_sha,error_code FROM jobs WHERE job_id=? AND owner_sha=?",
                (job_id, owner_sha),
            ).fetchone()
            return dict(row) if row is not None else None
