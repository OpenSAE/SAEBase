"""对点名文件做稳定摘要。只读调用方给出的路径，不扫描目录。"""

from __future__ import annotations

import hashlib
from pathlib import Path


class IndexRefusal(ValueError):
    """清单中的某一项不能成为文件证据。"""


def index_named_files(root: str | Path, names: list[str]) -> list[dict[str, object]]:
    """返回按路径排序的 path、sha256 和 size。

    输入必须是字符串列表。重复项、缺失文件、父目录跳转和指向根外的软链接都会拒绝。
    空列表是合法输入，结果仍是空列表。
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise IndexRefusal("root is not a directory")
    if not isinstance(names, list):
        raise IndexRefusal("names must be a list")
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for raw in names:
        rel = _relative_name(raw)
        if rel in seen:
            raise IndexRefusal(f"duplicate path: {rel}")
        seen.add(rel)
        target = _inside_root(base, rel)
        if not target.is_file():
            raise IndexRefusal(f"missing file: {rel}")
        payload = target.read_bytes()
        rows.append(
            {"path": rel, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _relative_name(raw: object) -> str:
    """在打开文件之前拒绝绝对路径、空段和父目录跳转。"""
    if not isinstance(raw, str) or raw != raw.strip() or not raw:
        raise IndexRefusal("path must be a nonempty relative string")
    if raw.startswith("/") or "\\" in raw:
        raise IndexRefusal(f"absolute path rejected: {raw}")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise IndexRefusal(f"path escape rejected: {raw}")
    return raw


def _inside_root(base: Path, rel: str) -> Path:
    """逐段解析；软链接的真实位置必须仍在根目录内。"""
    current = base
    for part in rel.split("/"):
        current = current / part
        if not current.is_symlink():
            continue
        resolved = current.resolve()
        if not resolved.is_relative_to(base):
            raise IndexRefusal(f"symlink escapes root: {rel}")
        current = resolved
    if current.exists() and not current.resolve().is_relative_to(base):
        raise IndexRefusal(f"path escapes root: {rel}")
    return current
