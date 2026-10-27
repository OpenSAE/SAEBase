"""Small, read-only term projections from doc/knowledge/glossary.json."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
GLOSSARY = ROOT / "doc/knowledge/glossary.json"


def select_terms(glossary: dict, ids: list[str]) -> dict:
    by_id = {term["id"]: term for term in glossary["terms"]}
    missing = [k for k in ids if k not in by_id]
    if missing:
        raise ValueError("Unknown glossary terms: " + ", ".join(missing))
    return {
        "kind": "glossary_projection",
        "schema_version": 1,
        "terms": [by_id[k] for k in dict.fromkeys(ids)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", help="Stable term ID, e.g. term.sae")
    args = parser.parse_args()
    try:
        glossary = json.loads(GLOSSARY.read_text(encoding="utf-8"))
        output = select_terms(glossary, args.ids)
    except (OSError, ValueError, KeyError) as exc:
        print("INVALID glossary selection:", exc, file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
