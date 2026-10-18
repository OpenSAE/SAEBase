"""Publish immutable, consented result summaries; never execute user code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import canonical_json, data_root
from .ingest import _atomic_bytes


def publish_report(source: dict[str, Any], *, output: Path, public_consent: bool) -> dict[str, Any]:
    if not public_consent:
        raise PermissionError("Explicit public-text/result publication approval required")
    if source.get("visibility") != "public":
        raise ValueError("Source must explicitly request public visibility")
    if source.get("source_type") not in ("real_checkpoint","synthetic_demo","external_verified"):
        raise ValueError("Source type must be explicit")
    if source.get("kind") not in ("intervention","sar","rss","localization","benchmark"):
        raise ValueError("Unsupported report kind")
    if not isinstance(source.get("protocol_id"),str) or not source["protocol_id"]:
        raise ValueError("Missing protocol identity")
    if not isinstance(source.get("source"),dict) or not source["source"].get("commit"):
        raise ValueError("Missing source commit provenance")
    if not isinstance(source.get("title"),str) or not source["title"].strip():
        raise ValueError("Missing report title")
    if not isinstance(source.get("results"),list):
        raise ValueError("Missing measured results")
    cleaned=dict(source)
    cleaned.pop("report_id",None)
    for result in cleaned["results"]:
        if not isinstance(result,dict) or not {"metric","status"} <= result.keys():
            raise ValueError("Malformed result row")
        if result["status"] not in ("completed","blocked","unsupported","failed","undefined"):
            raise ValueError("Unsupported result status")
        if result["status"] != "completed" and result.get("value") is not None:
            raise ValueError("Uncomputed result cannot have a numerical value")
    report_id=hashlib.sha256(canonical_json(cleaned)).hexdigest()[:40]
    cleaned["report_id"]=report_id
    folder=output / report_id
    if folder.exists():
        raise FileExistsError("This immutable report already exists")
    payload=canonical_json(cleaned)
    manifest={
        "schema":"opensae.public_report.v1",
        "report_id":report_id,
        "report_sha256":hashlib.sha256(payload).hexdigest(),
        "source_type":cleaned["source_type"],
        "public_consent":True,
    }
    _atomic_bytes(folder/"report.json",payload)
    _atomic_bytes(folder/"manifest.json",canonical_json(manifest))
    return {"report_id":report_id,"path":str(folder),"report_sha256":manifest["report_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--public-consent",action="store_true")
    args=parser.parse_args(argv)
    with args.input.open("r",encoding="utf-8") as f:
        report=json.load(f)
    print(json.dumps(publish_report(report,output=args.output or data_root()/"reports",
        public_consent=args.public_consent),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
