from __future__ import annotations

from pathlib import Path

from src.schema import digest, require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"allowed", "operations"})
    if payload["allowed"] != "workspace":
        raise ValueError("schema")
    workspace = Path(workdir).resolve()
    planned = []
    for operation in payload["operations"]:
        rel = operation.get("path")
        if (
            not isinstance(rel, str)
            or not rel
            or rel.startswith(("/", "~"))
            or "\\" in rel
            or ":" in rel
            or any(part in ("", ".", "..") for part in rel.split("/"))
            or operation.get("op") != "write"
            or not isinstance(operation.get("text"), str)
        ):
            raise ValueError("path or operation")
        target = (workspace / rel).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError("escape")
        planned.append((rel, target, operation["text"]))
    written = []
    for rel, target, text in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append({"path": rel, "sha256": digest(text)})
    return {
        "written": written,
        "files": sorted(
            str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()
        ),
        "count": len(written),
        "scope": "workspace",
    }
