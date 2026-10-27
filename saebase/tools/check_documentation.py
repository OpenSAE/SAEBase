#!/usr/bin/env python3
"""Check local Markdown links in SAEBase's developer-facing documentation.

No network access or third-party Markdown package is required. External URLs
and same-page fragments are not checked by this script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"!?\[[^\]\n]+\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]+\")?\)")
DOCS = (
    "README.md",
    "README_cn.md",
    ".github/CONTRIBUTING.md",
)


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        [
            *(root / p for p in DOCS),
            *(root / "harness/github_development_doc").rglob("*.md"),
            *(root / "saebase/examples").rglob("*.md"),
        ]
    )


def missing_links(path: Path, root: Path) -> list[str]:
    """Check that each relative Markdown destination exists inside root."""
    root = root.resolve()
    missing = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for match in LINK.finditer(line):
            target = match.group(1).strip("<>")
            if not target or target.startswith("#"):
                continue
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            local = unquote(parsed.path)
            if not local:
                continue
            dest = (path.parent / local).resolve()
            if not dest.is_relative_to(root) or not dest.exists():
                missing.append(f"{path.relative_to(root)}:{index}: {target}")
    return missing


def validate(root: Path = ROOT) -> tuple[int, list[str]]:
    files = markdown_files(root)
    absent = [str(p.relative_to(root)) for p in files if not p.is_file()]
    if absent:
        return len(files), [f"missing Markdown page: {p}" for p in absent]
    broken = []
    for file in files:
        broken.extend(missing_links(file, root))
    return len(files), broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on broken relative links")
    args = parser.parse_args()
    if not args.check:
        parser.error("pass --check")
    n, broken = validate()
    if broken:
        for item in broken:
            print("Broken documentation link:", item)
        return 1
    print(f"Verified local links in {n} Markdown documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
