"""Append-only, crash-consistent measurement journal for batch experiments.

Files are committed by an atomic manifest replacement. Every committed JSONL
shard is SHA-256 checked during resume; uncommitted files can be safely ignored.
This is an exclusively locked local filesystem writer, not a distributed DB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import fcntl
import hashlib
import json
import os
import tempfile

from saebase.store.identity import fingerprint, stable_json
from .specs import ExperimentIdentity

SCHEMA = "saebase.m2.results.v1"


class MeasurementIntegrityError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".m2-partial-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


class ResultWriter:
    """One active writer per directory, with manifest-based resume."""

    def __init__(
        self,
        root: str | Path,
        identity: ExperimentIdentity,
        *,
        shard_rows: int = 128,
        resume: bool = True,
    ) -> None:
        if type(shard_rows) is not int or shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.shard_rows = shard_rows
        self._lock = (self.root / ".writer.lock").open("a+")
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._closed = False
        self._pending: list[dict[str, Any]] = []
        self._index: set[str] = set()
        try:
            path = self.root / "manifest.json"
            if path.exists():
                if not resume:
                    raise FileExistsError(f"existing experiment {self.root}")
                self.manifest = self._read_json(path)
                if (
                    self.manifest.get("schema") != SCHEMA
                    or self.manifest.get("identity") != identity.to_dict()
                ):
                    raise MeasurementIntegrityError(
                        "experiment identity/schema changed; cannot resume"
                    )
                if self.manifest.get("identity_key") != identity.key:
                    raise MeasurementIntegrityError("experiment digest changed")
                for shard in self.manifest["shards"]:
                    name = shard["filename"]
                    if not isinstance(name, str) or name != f"shard-{shard['index']:06d}.jsonl":
                        raise MeasurementIntegrityError("unsafe shard filename")
                    buf = (self.root / name).read_bytes()
                    if _sha(buf) != shard["sha256"]:
                        raise MeasurementIntegrityError(f"committed result shard corrupted: {name}")
                    rows = buf.splitlines()
                    if len(rows) != shard["rows"]:
                        raise MeasurementIntegrityError(f"row count mismatch: {name}")
                    for line in rows:
                        item = json.loads(line)
                        key = item.get("measurement_id")
                        if not isinstance(key, str) or key != fingerprint(
                            item["measurement_identity"]
                        ):
                            raise MeasurementIntegrityError("measurement identity tampered")
                        if key in self._index:
                            raise MeasurementIntegrityError("duplicate measurement identity")
                        self._index.add(key)
            else:
                self.manifest = {
                    "schema": SCHEMA,
                    "identity": identity.to_dict(),
                    "identity_key": identity.key,
                    "shards": [],
                    "rows": 0,
                }
                _atomic(path, (stable_json(self.manifest) + "\n").encode())
        except Exception:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise MeasurementIntegrityError("invalid results manifest") from e

    def has(self, key: str) -> bool:
        if self._closed:
            raise RuntimeError("writer is closed")
        return key in self._index

    def append(self, row: Mapping[str, Any]) -> bool:
        if self._closed:
            raise RuntimeError("writer is closed")
        data = dict(row)
        key = data.get("measurement_id")
        if not isinstance(key, str) or key != fingerprint(data.get("measurement_identity")):
            raise MeasurementIntegrityError("row has no valid measurement identity")
        stable_json(data)
        if key in self._index:
            return False
        self._pending.append(data)
        self._index.add(key)
        if len(self._pending) >= self.shard_rows:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._pending:
            return
        index = len(self.manifest["shards"])
        name = f"shard-{index:06d}.jsonl"
        data = ("\n".join(stable_json(row) for row in self._pending) + "\n").encode()
        # Replace an uncommitted shard after a simulated crash. Never replace a
        # committed shard because the next index is exactly manifest shard count.
        _atomic(self.root / name, data)
        updated = dict(self.manifest)
        updated["shards"] = [
            *self.manifest["shards"],
            {"index": index, "filename": name, "sha256": _sha(data), "rows": len(self._pending)},
        ]
        updated["rows"] = self.manifest["rows"] + len(self._pending)
        _atomic(self.root / "manifest.json", (stable_json(updated) + "\n").encode())
        self.manifest = updated
        self._pending.clear()

    def close(self) -> None:
        if not self._closed:
            try:
                self.flush()
            finally:
                self._closed = True
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
                self._lock.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def committed(self) -> int:
        return int(self.manifest["rows"])


def read_results(root: str | Path, identity: ExperimentIdentity) -> list[dict[str, Any]]:
    """Verified snapshot of a committed result set."""
    with ResultWriter(root, identity) as writer:
        rows = []
        for shard in writer.manifest["shards"]:
            for line in (writer.root / shard["filename"]).read_text().splitlines():
                rows.append(json.loads(line))
    return rows
