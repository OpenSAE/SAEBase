"""Validate optional scientific Context attributes and their existing source evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "doc/knowledge/context_attributes.json"
GLOSSARY = ROOT / "doc/knowledge/glossary.json"
TYPES = {"text", "formula", "diagram", "code", "reference"}
DIAGRAMS = {"residual", "cub_split"}


class ContextAttributeError(ValueError):
    pass


def source_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ContextAttributeError("Source paths must be repository-relative")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ContextAttributeError("Unsafe source path")
    path = (root / value).resolve(strict=False)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContextAttributeError(f"Referenced source file is unavailable: {value}")
    return path


def validate_attributes(data: dict, term_ids: set[str], root: Path = ROOT) -> dict:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ContextAttributeError("Unrecognized Context attributes schema")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ContextAttributeError("Context attributes require entries list")
    seen = set()
    block_count = 0
    for item in entries:
        if not isinstance(item, dict):
            raise ContextAttributeError("Context entry must be an object")
        term_id = item.get("term_id")
        if not isinstance(term_id, str) or term_id not in term_ids or term_id in seen:
            raise ContextAttributeError("Duplicate or undefined Context ID: " + str(term_id))
        seen.add(term_id)
        blocks = item.get("blocks")
        if not isinstance(blocks, list) or not blocks or len(blocks) > 8:
            raise ContextAttributeError("Each Context entry needs 1-8 blocks")
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") not in TYPES:
                raise ContextAttributeError("Unsupported Context attribute type")
            if not isinstance(block.get("label"), str) or not block["label"].strip():
                raise ContextAttributeError("Every Context block needs a human-readable label")
            kind = block["type"]
            location = block.get("source") if kind != "reference" else block.get("path")
            origin = source_path(location, root)
            if kind == "reference":
                if "value" in block:
                    raise ContextAttributeError("References use path, not text values")
            else:
                value = block.get("value") if kind != "diagram" else block.get("caption")
                if not isinstance(value, str) or not value.strip() or len(value) > 1300:
                    raise ContextAttributeError("Missing or oversized Context attribute body")
            if kind == "diagram" and block.get("diagram") not in DIAGRAMS:
                raise ContextAttributeError("Only reviewed schematic types are allowed")
            if kind == "code":
                if origin.suffix != ".py":
                    raise ContextAttributeError("Source code excerpts must reference Python files")
                if block["value"] not in origin.read_text(encoding="utf-8"):
                    raise ContextAttributeError("Code snippet does not match its cited source")
                if len(block["value"]) > 500:
                    raise ContextAttributeError("Code excerpt too long for the public view")
            if kind == "formula" and len(block["value"]) > 180:
                raise ContextAttributeError("Formula too long for a compact node")
            block_count += 1
    return {"terms_with_attributes": len(seen), "attribute_blocks": block_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", default=str(DEFAULT))
    args = parser.parse_args()
    try:
        data = json.loads(Path(args.attributes).read_text(encoding="utf-8"))
        glossary = json.loads(GLOSSARY.read_text(encoding="utf-8"))
        result = validate_attributes(data, {x["id"] for x in glossary["terms"]})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("INVALID Context attribute definition:", exc, file=sys.stderr)
        return 1
    print("VALID Context attributes:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
