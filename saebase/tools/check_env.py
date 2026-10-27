#!/usr/bin/env python3
"""Explain dependency availability without loading a GPU runtime or a model."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from saebase.runtime import missing_packages, print_python_identity, required_packages


def main():
    p = argparse.ArgumentParser(description="Inspect SAEBase dependencies")
    p.add_argument("--mode", choices=("core", "analysis", "generation"), default="analysis")
    a = p.parse_args()
    print_python_identity()
    missing = missing_packages(a.mode)
    for mod, package in required_packages(a.mode):
        print(f"[{'missing' if any(x[0] == mod for x in missing) else 'ok'}] {mod} ({package})")
    if missing:
        extra = {"core": ".", "analysis": ".[analysis]", "generation": ".[generation]"}[a.mode]
        print(f'[env] Install from the repository: python -m pip install -e "{extra}"')
        return 1
    print(
        f"[env] {a.mode} dependency probes passed (checkpoint download and GPU runtime not tested)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
