"""Reusable reference validators for universal Harness invariants.

Governance-owned: task agents must not modify these functions to make an exam
pass. They are usable for independent negative tests, never a replacement for
human approval or a full security sandbox.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any


class InvariantError(ValueError):
    """Input violates a task-independent invariant."""


_REFERENCE = re.compile(r"^([^:\n\r]+):([1-9][0-9]*)$")
_SECRET_KEY = re.compile(
    r"(?i)(?:token|secret|password|api.?key|authorization|cookie|credential|private.?key)"
)
_PRIVATE_PATH_KEY = re.compile(r"(?i)^(?:home|cwd|hostname|user_path|path)$")
_PRIVATE_VALUE = re.compile(
    r"(?i)(?:[A-Za-z]:[\\/]|/(?:home|Users|root|private|run/secrets)/|"
    r"(?:(?:token|password|secret|api_key)\s*[:=]\s*\S+))"
)
_LOCALE_FIELDS = ("title", "summary", "description")
_PLACEHOLDERS = ("english translation unavailable", "中文翻译不可用")


def reference_location(text: str, root: Path, *, resolve: bool = True) -> dict[str, Any]:
    """Separate reference shape, path containment and actual line availability.

    If resolve=False, only syntax and containment are confirmed. An unresolved
    source is never represented as verified provenance.
    """
    if not isinstance(text, str):
        raise InvariantError("Source reference is not text")
    match = _REFERENCE.fullmatch(text)
    if not match:
        raise InvariantError("Expected canonical source path:positive_line")
    relative, line_text = match.groups()
    path = PurePosixPath(relative)
    if (
        path.is_absolute() or "\\" in relative or relative.startswith("/")
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or path.as_posix() != relative
    ):
        raise InvariantError("Source path is not canonical repository-relative")
    root = Path(root).resolve(strict=True)
    absolute = (root / relative).resolve(strict=False)
    if not absolute.is_relative_to(root):
        raise InvariantError("Source symlink escapes authority root")
    record: dict[str, Any] = {
        "path": relative,
        "line": int(line_text),
        "syntax_valid": True,
        "boundary_valid": True,
        "source_resolved": False,
    }
    if resolve:
        if not absolute.is_file():
            raise InvariantError("Source does not exist")
        try:
            line_count = sum(1 for _ in absolute.open(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            raise InvariantError("Source cannot be decoded or read") from exc
        if record["line"] > line_count:
            raise InvariantError("Source line does not exist")
        record["source_resolved"] = True
    return record


def redact_structure(value: Any, *, max_depth: int = 24) -> tuple[Any, list[str]]:
    """Redact sensitive keys/values throughout admitted dict/list structures.

    Unsupported structures and excess depth fail closed instead of appearing
    unredacted in a result. Does not log original secrets.
    """
    redactions: list[str] = []

    def clean(current: Any, key_path: str, depth: int) -> Any:
        if depth > max_depth:
            raise InvariantError("Structured output exceeds supported depth")
        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                raise InvariantError("Structured output keys must be strings")
            result = {}
            for key in sorted(current):
                location = key_path + "." + key if key_path else key
                if _SECRET_KEY.search(key) or _PRIVATE_PATH_KEY.fullmatch(key):
                    redactions.append(location)
                    result[key] = "[REDACTED]"
                else:
                    result[key] = clean(current[key], location, depth + 1)
            return result
        if isinstance(current, (list, tuple)):
            return [clean(item, f"{key_path}[{i}]", depth + 1)
                    for i, item in enumerate(current)]
        if isinstance(current, str):
            if _PRIVATE_VALUE.search(current) or current.startswith(("/", "~")):
                redactions.append(key_path)
                return "[REDACTED]"
            return current
        if current is None or isinstance(current, (bool, int, float)):
            return current
        raise InvariantError("Unsupported structure type")

    return clean(value, "", 0), sorted(set(redactions))


def validated_dfs(graph: dict, *, scope: str = "all", target: str | None = None) -> list[str]:
    """DFS over a complete valid parent graph; never materialize missing targets."""
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise InvariantError("Graph requires nodes list")
    rows = graph["nodes"]
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
           or not row["id"] for row in rows):
        raise InvariantError("Invalid graph node")
    nodes = {row["id"]: row for row in rows}
    if len(nodes) != len(rows):
        raise InvariantError("Duplicate graph node")
    for row in rows:
        parent = row.get("parent_id")
        if parent is not None and parent not in nodes:
            raise InvariantError("Unknown parent node")
        if parent is not None and row.get("tree_id") != nodes[parent].get("tree_id"):
            raise InvariantError("Cross-tree parent link")
    for start in nodes:
        seen: set[str] = set()
        current = start
        while current is not None:
            if current in seen:
                raise InvariantError("Parent cycle")
            seen.add(current)
            current = nodes[current].get("parent_id")
    if scope not in ("all", "tree", "parent", "node"):
        raise InvariantError("Unknown review scope")
    if scope in ("node", "parent") and target not in nodes:
        raise InvariantError("Review target does not exist")
    if scope == "tree":
        if not isinstance(target, str) or not any(
            row.get("tree_id") == target for row in rows
        ):
            raise InvariantError("Unknown tree")
        selected = {id_ for id_, row in nodes.items() if row.get("tree_id") == target}
    elif scope in ("parent", "node"):
        selected = {target} if scope == "node" else set(nodes)
    else:
        selected = set(nodes)

    children = {id_: [] for id_ in nodes}
    for row in rows:
        parent = row.get("parent_id")
        if parent in nodes:
            children[parent].append(row["id"])
    for group in children.values():
        group.sort(key=lambda id_: (nodes[id_].get("order", 0), id_))

    result: list[str] = []

    def visit(id_: str) -> None:
        if id_ not in selected:
            return
        result.append(id_)
        if scope != "node":
            for child in children[id_]:
                visit(child)

    if scope in ("node", "parent"):
        visit(target)
    else:
        roots = [id_ for id_ in selected
                 if nodes[id_].get("parent_id") not in selected]
        roots.sort(key=lambda id_: (nodes[id_].get("order", 0), id_))
        for root in roots:
            visit(root)
    return result


def localization_defects(record: dict, *, required=("zh", "en")) -> list[str]:
    """Report missing language fields and placeholder/corruption without rewriting."""
    if not isinstance(record, dict):
        raise InvariantError("Localization record must be a mapping")
    defects: list[str] = []
    for field in _LOCALE_FIELDS:
        value = record.get(field)
        if not isinstance(value, dict):
            defects.extend(f"{field}.{lang}:missing" for lang in required)
            continue
        for lang in required:
            text = value.get(lang)
            if not isinstance(text, str) or not text.strip():
                defects.append(f"{field}.{lang}:missing")
            elif "\ufffd" in text or any(p in text.lower() for p in _PLACEHOLDERS):
                defects.append(f"{field}.{lang}:untranslated")
    for lang in required:
        title = (record.get("title") or {}).get(lang, "")
        summary = (record.get("summary") or {}).get(lang, "")
        if isinstance(title, str) and isinstance(summary, str) and title and summary:
            norm = lambda value: "".join(c.casefold() for c in value if c.isalnum())
            if norm(title) == norm(summary):
                defects.append(f"summary.{lang}:tautology")
    if "localized_signature" in record and record.get("localized_signature") != record.get("signature"):
        defects.append("signature:changed")
    return defects
