"""Versioned GitHub Harness archive registry and pre-release preparation.

Historical contents are preserved byte-for-byte; this tool NEVER treats archive
records as scientific validation. Ordinary checks perform no GitHub writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
INDEX = Path("harness/github/archive_index.json")
SCHEMA = "saebase.harness.archive.v1"
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHORT_NAME = re.compile(r"^[a-z]+_[a-z]+$")
DOC_NAME = re.compile(
    r"^(?P<version>\d+\.\d+\.\d+)_(?P<kind>dev|wp)(?P<seq>\d{3})_"
    r"(?P<slug>[a-z]+_[a-z]+)(?P<suffix>\.en\.md|\.md|\.json)$"
)
TEST_DIR = re.compile(
    r"^(?P<version>\d+\.\d+\.\d+)_wt(?P<seq>\d{3})_"
    r"(?P<slug>[a-z]+_[a-z]+)$"
)
OWNERS = {"dev": "development_doc", "wp": "work_plan", "wt": "work_test"}


class ArchiveError(ValueError):
    """Refuse undocumented, unsafe or non-reproducible archival changes."""


def version_tuple(version: str) -> tuple[int, int, int]:
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise ArchiveError("Expected exact semantic version M.m.p")
    return tuple(map(int, version.split(".")))


def safe(path: str) -> Path:
    if not isinstance(path, str) or not path.startswith("harness/github/"):
        raise ArchiveError("Archive path must remain under harness/github/")
    p = Path(path)
    if p.is_absolute() or "\\" in path or ".." in p.parts:
        raise ArchiveError("Unsafe archive path")
    return p


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def path_blob(root: Path, name: str) -> str:
    path = root / safe(name)
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("Missing or unsafe archived source: " + name)
    return git_blob(path.read_bytes())


def read_index(root: Path = ROOT) -> dict:
    index = json.loads((root / INDEX).read_text(encoding="utf-8"))
    if index.get("schema") != SCHEMA or not isinstance(index.get("entries"), list):
        raise ArchiveError("Invalid archive index")
    return index


def archive_identity(path: str) -> tuple[str, str, int]:
    relative = safe(path).parts
    if len(relative) < 7:
        raise ArchiveError("Invalid version archive depth: " + path)
    # harness/github/<owner>/archive/<version>/<name>[/<file>]
    _, _, owner, archive, version = relative[:5]
    if archive != "archive" or owner not in OWNERS.values():
        raise ArchiveError("Invalid archive owner: " + path)
    version_tuple(version)
    if owner == "work_test":
        if len(relative) < 7:
            raise ArchiveError("Versioned WT must be a directory containing tests")
        m = TEST_DIR.fullmatch(relative[5])
        kind = "wt"
    else:
        if len(relative) != 6:
            raise ArchiveError("Versioned Dev/WP must be direct archive files")
        m = DOC_NAME.fullmatch(relative[5])
        kind = "dev" if owner == "development_doc" else "wp"
    if not m or m.group("kind") != kind if kind != "wt" else not m:
        raise ArchiveError("Noncanonical archive filename: " + path)
    if m.group("version") != version:
        raise ArchiveError("Archive filename and folder versions differ")
    return version, kind, int(m.group("seq"))


def check(index: dict, root: Path = ROOT, *, release_version: str | None = None) -> dict:
    if index.get("schema") != SCHEMA:
        raise ArchiveError("Wrong archive schema")
    targets: set[str] = set()
    numbers: dict[tuple[str, str, int], str] = {}
    for row in index["entries"]:
        source, target = row["original_path"], row["archived_path"]
        safe(source)
        identity = archive_identity(target)
        if row.get("version") != identity[0]:
            raise ArchiveError("Archive index version mismatch: " + target)
        if target in targets:
            raise ArchiveError("Duplicate archival target " + target)
        targets.add(target)
        # A WT folder contains multiple files sharing the same sequence.
        if identity in numbers and (
            identity[1] != "wt" or numbers[identity] != str(Path(target).parent)
        ):
            raise ArchiveError("Duplicate archive numbering: " + target)
        numbers[identity] = str(Path(target).parent)
        if row.get("mode") not in ("move", "copy_compat"):
            raise ArchiveError("Invalid archive migration mode")
        if row.get("status") != "archived":
            raise ArchiveError("Index cannot claim unfinished archival change")
        observed = path_blob(root, target)
        if observed != row["git_blob_sha"]:
            raise ArchiveError("Historical bytes changed: " + target)
        old = root / safe(source)
        if row["mode"] == "move" and old.exists():
            raise ArchiveError("Moved historical path still exists: " + source)
        if row["mode"] == "copy_compat" and path_blob(root, source) != observed:
            raise ArchiveError("Legacy compatibility source differs: " + source)
    if release_version:
        v = version_tuple(release_version)
        highest = max((version_tuple(e["version"]) for e in index["entries"]), default=(0, 0, 0))
        if v > highest and release_version not in index.get("prepared_for_versions", []):
            raise ArchiveError(
                "Full Release Validation requires archive preparation for "
                + release_version
            )
    return {"status": "PASS", "records": len(targets),
            "versions": sorted({e["version"] for e in index["entries"]}, key=version_tuple),
            "release_ready": release_version or "not_requested"}


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()


def prepare(closing: str, next_version: str, root: Path = ROOT) -> dict:
    """Read-only, deterministic candidate plan; no commits, PR or authority edits."""
    if version_tuple(next_version) <= version_tuple(closing):
        raise ArchiveError("Next release must be newer than the version being archived")
    index = read_index(root)
    check(index, root)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if str(project["version"]) != next_version:
        raise ArchiveError("Next version must match pyproject.toml before archive preparation")
    found = []
    for kind, owner in OWNERS.items():
        folder = root / "harness/github" / owner
        if kind == "wt":
            for child in sorted(folder.iterdir()):
                m = TEST_DIR.fullmatch(child.name)
                if not m or m.group("version") != closing or not child.is_dir():
                    continue
                if not (child / ".archive-ready").is_file():
                    # Protect live tests. Retirement must be deliberate.
                    continue
                for source in sorted(child.rglob("*")):
                    if source.is_file():
                        relative = source.relative_to(root).as_posix()
                        target = ("harness/github/" + owner + "/archive/"
                                  + closing + "/" + child.name + "/"
                                  + source.relative_to(child).as_posix())
                        found.append({"original_path": relative, "archived_path": target,
                                      "git_blob_sha": path_blob(root, relative), "mode": "move",
                                      "version": closing})
        else:
            for source in sorted(folder.iterdir()):
                m = DOC_NAME.fullmatch(source.name)
                if not m or m.group("version") != closing or m.group("kind") != kind:
                    continue
                relative = source.relative_to(root).as_posix()
                target = ("harness/github/" + owner + "/archive/"
                          + closing + "/" + source.name)
                found.append({"original_path": relative, "archived_path": target,
                              "git_blob_sha": path_blob(root, relative), "mode": "move",
                              "version": closing})
    existing = {x["original_path"] for x in index["entries"]}
    candidates = [x for x in found if x["original_path"] not in existing]
    for row in candidates:
        archive_identity(row["archived_path"])
        if (root / safe(row["archived_path"])).exists():
            raise ArchiveError("Archive target already exists: " + row["archived_path"])
    return {
        "schema": "saebase.harness.archive_preparation.v1",
        "source_sha": git_head(root), "closing_version": closing,
        "next_version": next_version, "status": "HUMAN_CHECK_REQUIRED",
        "candidates": candidates,
        "note": "Prepare is read-only. Review all affected Atlas locations before implementation.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("check")
    verify.add_argument("--release-version")
    plan = sub.add_parser("prepare")
    plan.add_argument("--closing-version", required=True)
    plan.add_argument("--next-version", required=True)
    plan.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        if args.command == "check":
            result = check(read_index(), release_version=args.release_version)
        else:
            result = prepare(args.closing_version, args.next_version)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ArchiveError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        p.exit(2, "ARCHIVE BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
