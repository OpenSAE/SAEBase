"""Validate and aggregate authentic Stage B/C CPU SAE-only evidence artifacts.

Input directories contain extracted JSON files from GitHub Actions artifacts.
A corrected checkpoint may supersede exactly one previously BLOCKED observation,
without re-running unrelated successful SAE downloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from saebase.sae.coverage import load_coverage, verify_cpu_sae_evidence

CATALOG = Path(__file__).resolve().parents[1] / "conf" / "coverage" / "stage_bc.json"
SCHEMA = "saebase.coverage.cpu_matrix_result.v2"


def _records(folder: Path, catalog) -> dict[tuple[str, int], dict[str, Any]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing CPU artifact directory: {folder}")
    files = sorted(folder.rglob("*.json"))
    if not files:
        raise ValueError(f"no evidence JSON artifacts found in {folder}")
    entries = {}
    for path in files:
        raw = path.read_bytes()
        data = json.loads(raw)
        accepted = verify_cpu_sae_evidence(catalog, data)
        key = (accepted["sae_key"], accepted["layer"])
        if key in entries:
            raise ValueError(f"duplicated evidence for {key}")
        entries[key] = {
            **accepted,
            "evidence_file": path.name,
            "artifact_json_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return entries


def aggregate(
    source_dir: Path,
    catalog_path: Path = CATALOG,
    replacements_dir: Path | None = None,
) -> dict[str, Any]:
    catalog = load_coverage(catalog_path)
    expected = {
        (item.sae_key, layer) for item in catalog.entries for layer in item.cpu_probe_layers
    }
    base = _records(source_dir, catalog)
    if set(base) - expected:
        raise ValueError("baseline artifacts include undeclared SAE checkpoint")
    if replacements_dir is not None:
        repaired = _records(replacements_dir, catalog)
        if set(repaired) - expected:
            raise ValueError("replacement artifacts include undeclared checkpoint")
        for key, replacement in repaired.items():
            existing = base.get(key)
            if existing and existing["status"] == "sae_checkpoint_verified_only":
                if existing["sae_state_sha256"] != replacement.get("sae_state_sha256"):
                    raise ValueError("cannot silently replace a previously verified SAE hash")
                continue
            if replacement["status"] != "sae_checkpoint_verified_only":
                raise ValueError("replacement must be a successful genuine checkpoint artifact")
            base[key] = {**replacement, "repaired_after_block": existing is not None}
    missing = sorted(expected - set(base))
    passed = sorted(x for x, v in base.items() if v["status"] == "sae_checkpoint_verified_only")
    blocked = sorted(x for x, v in base.items() if v["status"] != "sae_checkpoint_verified_only")
    results = [{**base[key]} for key in sorted(base)]
    return {
        "schema": SCHEMA,
        "kind": "original_SAE_weights_only_not_model_SAE_pair_verification",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "expected": len(expected),
        "with_artifact": len(base),
        "sae_only_verified": len(passed),
        "failed_or_blocked": len(blocked),
        "missing": len(missing),
        "blocked_checkpoints": [f"{k[0]} L{k[1]}" for k in blocked],
        "missing_checkpoints": [f"{k[0]} L{k[1]}" for k in missing],
        "entries": results,
        "all_checkpoints_verified": not missing and not blocked,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--replacements-dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=Path("stage_bc_cpu_result.json"))
    args = parser.parse_args(argv)
    summary = aggregate(args.source_dir, args.catalog, args.replacements_dir)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"SAE-only {summary['sae_only_verified']}/{summary['expected']} verified; "
        f"{summary['failed_or_blocked']} blocked; {summary['missing']} missing."
    )
    if summary["blocked_checkpoints"]:
        print("Blocked:", *summary["blocked_checkpoints"], sep="\n- ")
    if summary["missing_checkpoints"]:
        print("Missing:", *summary["missing_checkpoints"], sep="\n- ")
    return 0 if summary["all_checkpoints_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
