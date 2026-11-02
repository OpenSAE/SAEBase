"""Read only explicitly authorized SAEBase Context sources for agent dialogue.

This is source-access evidence, not validation of the agent's prose or a grant of
permission to edit, execute, network, download models, or disclose private data.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

MAX_BYTES = 2_000_000
SOURCES = {
    "knowledge": ("doc/knowledge/context_graph.json", "nodes"),
    "core": ("harness/core/project_graph.json", "nodes"),
    "term": ("doc/knowledge/glossary.json", "terms"),
    "atlas_term": ("doc/knowledge/atlas_terms.json", "terms"),
}


def _safe_path(root: Path, name: str) -> Path:
    """Require a repository-relative *file*, not a directory or escaping symlink."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("unsafe source path")
    base = root.resolve(strict=True)
    target = (base / name).resolve(strict=False)
    if not target.is_relative_to(base):
        raise ValueError("source escapes repository")
    if not target.is_file():
        raise FileNotFoundError("source does not exist")
    if target.stat().st_size > MAX_BYTES:
        raise ValueError("source exceeds per-file read budget")
    return target


def _read(root: Path, name: str, allowed: frozenset[str]) -> tuple[dict[str, Any] | None, str | None, str]:
    if name not in allowed:
        return None, None, "BLOCKED_SCOPE"
    try:
        data = _safe_path(root, name).read_bytes()
        return {"bytes": data}, hashlib.sha256(data).hexdigest(), "READ"
    except (OSError, ValueError):
        return None, None, "UNRESOLVED"


def read_project_context(
    root: Path,
    requests: list[dict[str, str]],
    allowed_paths: list[str],
    *,
    max_requests: int = 12,
) -> dict[str, Any]:
    """Return small grounded projections and per-source access receipts.

    The caller must supply an exact-file allowlist granted for *this task*.
    No wildcard, directory-wide walk or automatic authorization is supported.
    """
    if not isinstance(requests, list) or len(requests) > max_requests:
        raise ValueError("too many context requests")
    if not isinstance(allowed_paths, list) or any(not isinstance(p, str) for p in allowed_paths):
        raise ValueError("allowed_paths must contain exact relative files")
    allowed = frozenset(allowed_paths)
    rows: list[dict[str, Any]] = []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("each request must be a mapping")
        kind, identifier = request.get("kind"), request.get("id")
        row: dict[str, Any] = {"kind": kind, "id": identifier, "status": "UNRESOLVED"}
        if not isinstance(identifier, str) or not identifier or not isinstance(kind, str):
            rows.append(row)
            continue
        if kind in SOURCES:
            name, collection = SOURCES[kind]
            data, digest, state = _read(root, name, allowed)
            row.update(source=name, source_sha256=digest, status=state)
            if data is not None:
                try:
                    payload = json.loads(data["bytes"])
                    found = next((x for x in payload[collection] if x.get("id") == identifier), None)
                    if found is not None:
                        row["status"] = "READ"
                        row["source_revision"] = payload.get("schema_version")
                        row["value"] = {
                            k: found[k] for k in (
                                "id", "title", "type", "description", "term", "definition",
                                "properties", "status", "source_evidence", "evidence",
                                "context_term_ids", "term_ids",
                            ) if k in found
                        }
                    else:
                        row["status"] = "UNRESOLVED_ID"
                except (ValueError, KeyError, TypeError):
                    row["status"] = "UNRESOLVED_SOURCE"
        elif kind == "code":
            name = identifier
            if not name.endswith(".py"):
                row["status"] = "UNRESOLVED_ID"
            else:
                data, digest, state = _read(root, name, allowed)
                row.update(source=name, source_sha256=digest, status=state)
                if data is not None:
                    try:
                        tree = ast.parse(data["bytes"].decode("utf-8"), filename=name)
                        row["value"] = {
                            "functions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][:30],
                            "classes": [n.name for n in tree.body if isinstance(n, ast.ClassDef)][:30],
                        }
                    except (ValueError, SyntaxError, UnicodeDecodeError):
                        row["status"] = "UNRESOLVED_SOURCE"
        elif kind == "environment":
            # Only an explicitly allowed snapshot can be read; never probe automatically.
            name = identifier
            if not name.endswith(".json") or not name.startswith("harness/environment/"):
                row["status"] = "UNRESOLVED_ID"
            else:
                data, digest, state = _read(root, name, allowed)
                row.update(source=name, source_sha256=digest, status=state)
                if data is not None:
                    try:
                        obj = json.loads(data["bytes"])
                        if not isinstance(obj, dict):
                            raise ValueError("invalid environment profile")
                        row["status"] = "SYNTHETIC" if obj.get("synthetic") is True else (
                            "OBSERVED_SNAPSHOT" if obj.get("captured_at_utc") else "UNRESOLVED_SOURCE"
                        )
                        row["value"] = {
                            "synthetic": obj.get("synthetic"),
                            "captured_at_utc": obj.get("captured_at_utc"),
                            "gpu_status": (obj.get("gpu") or {}).get("status"),
                        }
                    except (ValueError, AttributeError, TypeError):
                        row["status"] = "UNRESOLVED_SOURCE"
        else:
            row["status"] = "UNSUPPORTED_KIND"
        rows.append(row)
    return {
        "schema": "saebase.agent_context_receipt.v1",
        "permission": "READ_ONLY_EXACT_PATHS",
        "scientific_validation": "NOT_COVERED",
        "response_validation": "NOT_COVERED",
        "results": rows,
    }
