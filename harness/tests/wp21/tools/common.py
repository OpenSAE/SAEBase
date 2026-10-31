"""WP21 preparation and evidence utilities. No model execution or grading here."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath

BASE_COMMIT = "41a54cbfdee5cce24d3d8f7517aadef0c752c5ac"
ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "harness/tests/wp21"
ALWAYS = {"CT-BASE", "CT-SCOPE", "CT-TRUTH", "CT-CHECK", "CT-TRACE"}
PROTECTED = (".cursor/", ".github/", "harness/", "saebase/runtime/", "doc/")
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_json(path: Path):
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON field: " + key)
            obj[key] = value
        return obj

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=finite_float,
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON: " + x)),
    )


def write_json(path: Path, value, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def relative_name(value: str, *, pattern: bool = False) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("nonempty canonical relative path required")
    if "\\" in value or ":" in value or "\x00" in value or value.startswith("/"):
        raise ValueError("absolute, Windows, or NUL path forbidden")
    parts = value.split("/")
    if any(x in {"", ".", ".."} for x in parts) or PurePosixPath(value).as_posix() != value:
        raise ValueError("noncanonical path")
    if not pattern and any(c in value for c in "*?["):
        raise ValueError("wildcard forbidden in actual path")
    return value


def contained(root: Path, relative: str, *, exists: bool = False) -> Path:
    """Reject all symlink components, including symlinks that stay inside root."""
    relative_name(relative)
    root = root.resolve(strict=True)
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink is not an evidence/source file: " + relative)
    if not path.resolve(strict=exists).is_relative_to(root):
        raise ValueError("path escapes its declared root")
    return path


def snapshot(root: Path, *, skip_outputs: bool = True) -> dict[str, str]:
    """Hashes source, config and governance; model outputs stay in separate manifests."""
    result = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        rel_dir = Path(folder).relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        if skip_outputs and rel_dir == Path("."):
            dirs[:] = [d for d in dirs if d != "experiment"]
        if skip_outputs and rel_dir.as_posix() == "harness/tests/wp21":
            dirs[:] = [d for d in dirs if d not in {"runs", "acceptance"}]
        for directory in dirs:
            if (Path(folder) / directory).is_symlink():
                raise ValueError("source directory symlink forbidden")
        for name in sorted(files):
            path = Path(folder) / name
            rel = path.relative_to(root).as_posix()
            if path.suffix in IGNORED_SUFFIXES or rel == ".cursor/wp21_assignment.json":
                continue
            contained(root, rel, exists=True)
            result[rel] = sha256(path)
    return result


def load_spec(root: Path = ROOT):
    return read_json(root / "harness/tests/wp21/cases.json")


def get_case(case_id: str, root: Path = ROOT):
    if re.fullmatch(r"R(?:0[1-9]|1[0-9]|20)", case_id) is None:
        raise ValueError("case must be R01..R20")
    for case in load_spec(root)["cases"]:
        if case["id"] == case_id:
            return case
    raise ValueError("case not found")


def allowed_source(path: str, case: dict) -> bool:
    relative_name(path)
    if path.startswith(PROTECTED) or "/" not in path:
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in case["allowed_paths"])


def source_changes(before: dict, after: dict) -> list[dict]:
    result = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            result.append(
                {
                    "path": path,
                    "before_sha256": before.get(path),
                    "after_sha256": after.get(path),
                    "operation": "add"
                    if path not in before
                    else "delete"
                    if path not in after
                    else "modify",
                }
            )
    return result


# The examiner controls this manifest. Students must never regenerate it.
LOCK_PATH = "harness/tests/wp21/suite_lock.json"


def validate_suite_lock(root: Path = ROOT) -> dict:
    """Validate byte identities of the whole frozen exam, including card prose.

    This guards accidental drift and candidate edits. An independent package
    digest is still required to authenticate the lock itself across machines.
    """
    lock = read_json(contained(root, LOCK_PATH, exists=True))
    if lock.get("schema") != "saebase.wp21.suite_lock.v1" or lock.get("base_commit") != BASE_COMMIT:
        raise ValueError("wrong suite lock schema/base")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("suite lock has no file table")
    expected = {
        "harness/tests/wp21/cases.json",
        "harness/contracts/catalog.json",
        "harness/contracts/REAL_WORKFLOW_PROTOCOL.md",
        "harness/tests/wp21/fixtures/common/prompts.jsonl",
    }
    expected.update(f"harness/tests/wp21/cases/R{i:02d}.md" for i in range(1, 21))
    expected.update(f"harness/tests/wp21/prompts/R{i:02d}.md" for i in range(1, 21))
    fixtures = root / "harness/tests/wp21/fixtures"
    if fixtures.exists():
        for item in fixtures.rglob("*"):
            if item.is_file():
                expected.add(item.relative_to(root).as_posix())
    if set(files) != expected:
        raise ValueError(
            "frozen suite file list changed: added/missing="
            + repr(sorted(expected ^ set(files))[:12])
        )
    for relative, digest in sorted(files.items()):
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("bad suite lock checksum: " + relative)
        path = contained(root, relative, exists=True)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("frozen exam content changed: " + relative)
    if lock.get("files_sha256") != canonical_hash(files):
        raise ValueError("frozen exam manifest checksum differs")
    return {"file_count": len(files), "files_sha256": lock["files_sha256"]}
