"""对点名文件做稳定摘要。这是 WP18 测验代码，不会扫描仓库或上传结果。"""

from __future__ import annotations

import hashlib
from pathlib import Path


class EvidenceIndexError(ValueError):
    """路径不在允许的显式清单内。"""


def index_files(root: str | Path, relative_paths: list[str]) -> list[dict[str, object]]:
    """计算给定相对路径的 sha256 与字节数，并按路径稳定排序。

    拒绝重复路径、缺失文件、父目录跳转，以及指向根目录之外的软链接。
    不会遍历目录，也不会读取未点名的文件。
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise EvidenceIndexError("root is not a directory")
    if not isinstance(relative_paths, list):
        raise EvidenceIndexError("relative_paths must be a list")
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for raw in relative_paths:
        rel = _require_relative(raw)
        if rel in seen:
            raise EvidenceIndexError(f"duplicate path: {rel}")
        seen.add(rel)
        target = _resolve_inside(base, rel)
        if not target.is_file():
            raise EvidenceIndexError(f"missing file: {rel}")
        data = target.read_bytes()
        rows.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _require_relative(raw: object) -> str:
    """只接受不含绝对前缀、空段或父目录跳转的相对路径。"""
    if not isinstance(raw, str) or raw != raw.strip() or not raw:
        raise EvidenceIndexError("path must be a nonempty relative string")
    if raw.startswith("/") or "\\" in raw:
        raise EvidenceIndexError(f"absolute path rejected: {raw}")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise EvidenceIndexError(f"path escape rejected: {raw}")
    return raw


def _resolve_inside(base: Path, rel: str) -> Path:
    """逐段解析路径；软链接的真实目标必须仍在根目录内。"""
    current = base
    for part in rel.split("/"):
        current = current / part
        if not current.is_symlink():
            continue
        resolved = current.resolve()
        if not resolved.is_relative_to(base):
            raise EvidenceIndexError(f"symlink escapes root: {rel}")
        current = resolved
    if current.exists() and not current.resolve().is_relative_to(base):
        raise EvidenceIndexError(f"path escapes root: {rel}")
    return current
