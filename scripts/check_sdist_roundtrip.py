#!/usr/bin/env python3
"""Rebuild an SAEBase wheel from its sdist in a separate temporary directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def check(dist_dir: Path) -> None:
    sources = list(dist_dir.glob("opensae_saebase-*.tar.gz"))
    if len(sources) != 1:
        raise ValueError("expected exactly one SAEBase source archive")
    with tempfile.TemporaryDirectory(prefix="saebase-sdist-check-") as work:
        root = Path(work)
        with tarfile.open(sources[0], "r:gz") as archive:
            archive.extractall(root, filter="data")
        packages = [p for p in root.iterdir() if p.is_dir()]
        if len(packages) != 1 or not (packages[0] / "pyproject.toml").is_file():
            raise ValueError("source archive has an unexpected package layout")
        output = root / "rebuilt-dist"
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(output), str(packages[0])],
            check=True,
            cwd=root,
        )
        shutil.copy2(sources[0], output / sources[0].name)
        script = Path(__file__).resolve().parent / "verify_distribution.py"
        subprocess.run([sys.executable, str(script), str(output)], check=True)
        print("Clean sdist -> wheel roundtrip OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", default="dist")
    args = parser.parse_args()
    check(Path(args.dist_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
