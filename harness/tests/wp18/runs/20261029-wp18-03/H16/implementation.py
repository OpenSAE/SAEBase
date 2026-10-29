"""Check a synthetic file manifest without opening the named files."""

from __future__ import annotations

import re

_SHA = re.compile(r"[0-9a-f]{64}")


class InputError(ValueError):
    """Raised for a bad path, digest, size, or duplicate manifest row."""


def solve(payload: dict) -> dict:
    """Return manifest rows sorted by relative path."""
    if not isinstance(payload, dict) or set(payload) != {"files"}:
        raise InputError("schema")
    rows = payload["files"]
    if not isinstance(rows, list):
        raise InputError("files")
    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        item = _file_row(row)
        path = item["path"]
        if not isinstance(path, str) or path in seen:
            raise InputError("duplicate")
        seen.add(path)
        accepted.append(item)
    accepted.sort(key=lambda item: str(item["path"]))
    return {"files": accepted}


def _file_row(row: object) -> dict[str, object]:
    """Validate one relative path, byte count, and lowercase sha256."""
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise InputError("file")
    path = row["path"]
    size = row["bytes"]
    digest = row["sha256"]
    if not isinstance(path, str) or "\\" in path or path.startswith("/"):
        raise InputError("path")
    parts = path.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InputError("path")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise InputError("bytes")
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        raise InputError("sha256")
    return {"path": path, "bytes": size, "sha256": digest}
