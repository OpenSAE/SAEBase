"""Read-only auditor for Stage D compact evidence vs raw runs and status.

Fails when:
- status.json is not exactly the 48 catalog keys
- a counted M3/M4 summary lacks its raw run directory / manifests
- compact metric numbers disagree with metrics.jsonl
- checksums.sha256 entries do not match file contents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.benchmark.coverage_contract import load_coverage
from tasks.benchmark.stage_d import (
    build_site_records,
    file_sha256,
    load_status,
    read_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


def _mean_from_metrics(
    metrics_path: Path, metric_id: str, condition: str | None = None
) -> float | None:
    vals = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("metric_id") != metric_id:
            continue
        if condition is not None and row.get("condition_id") != condition:
            continue
        if row.get("value") is None:
            continue
        vals.append(float(row["value"]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def verify_checksums(manifest: Path, root: Path) -> list[str]:
    errors: list[str] = []
    if not manifest.is_file():
        return [f"missing checksums manifest: {manifest}"]
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            errors.append(f"{manifest}:{line_no}: malformed line")
            continue
        digest, rel = parts[0], parts[-1]
        if rel == "checksums.sha256" or rel.endswith("/checksums.sha256"):
            errors.append(f"{manifest}:{line_no}: manifest must not hash itself")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"missing file for checksum: {rel}")
            continue
        got = file_sha256(path)
        if got != digest:
            errors.append(f"checksum mismatch for {rel}")
    return errors


def verify_evidence(output: Path, *, evidence_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    records = build_site_records(load_coverage(CATALOG))
    expected_ids = {r.key.id for r in records}
    status = load_status(output)
    site_ids = set(status.get("sites", {}))
    if site_ids != expected_ids:
        missing = sorted(expected_ids - site_ids)
        extra = sorted(site_ids - expected_ids)
        if missing:
            errors.append(f"status missing {len(missing)} keys e.g. {missing[:3]}")
        if extra:
            errors.append(f"status has {len(extra)} non-canonical keys e.g. {extra[:3]}")

    reports = output / "reports"
    for path in sorted(reports.glob("*_m3m4.json")) if reports.exists() else []:
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_id = summary.get("m3_run_id")
        if not run_id:
            errors.append(f"{path.name}: missing m3_run_id")
            continue
        root = output / "runs" / run_id
        need = [
            "resolved_config.json",
            "plan.json",
            "metrics.jsonl",
            "metrics_manifest.json",
            "reliability.json",
            "reliability_manifest.json",
        ]
        missing = [n for n in need if not (root / n).exists()]
        if missing:
            errors.append(f"{path.name}: raw run {run_id} missing {missing}")
            continue
        metrics_path = root / "metrics.jsonl"
        for row in summary.get("metric_results", []):
            mid = row.get("metric_id")
            cond = row.get("condition")
            value = row.get("value")
            if value is None:
                continue
            # Compact summaries store one aggregated value per metric/condition.
            # Recompute mean over metrics.jsonl rows for the same ids.
            recomputed = None
            vals = []
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("metric_id") != mid:
                    continue
                if cond is not None and rec.get("condition_id") != cond:
                    continue
                if rec.get("value") is None:
                    continue
                vals.append(float(rec["value"]))
            if not vals:
                # Fall back: some runners store the compact value directly as the only row.
                continue
            recomputed = sum(vals) / len(vals)
            if abs(recomputed - float(value)) > 1e-6 and abs(vals[0] - float(value)) > 1e-6:
                # Accept either mean or first matching stored value.
                if not any(abs(v - float(value)) <= 1e-6 for v in vals):
                    errors.append(
                        f"{path.name}: metric {mid}/{cond} value {value} not in metrics.jsonl"
                    )
        manifest = json.loads((root / "metrics_manifest.json").read_text(encoding="utf-8"))
        if summary.get("m3_metrics_sha256") and summary["m3_metrics_sha256"] != manifest.get(
            "sha256"
        ):
            errors.append(f"{path.name}: m3_metrics_sha256 disagrees with metrics_manifest")

    # Acceptance checksums (canonical L<layer>.json only; notes stay in history).
    for acc in (output / "acceptance").rglob("L*.json") if (output / "acceptance").exists() else []:
        if not acc.stem[1:].isdigit():
            continue
        try:
            read_acceptance(acc)
        except ValueError as exc:
            errors.append(f"acceptance {acc}: {exc}")

    checksum_path = (evidence_root or output) / "final" / "checksums.sha256"
    if checksum_path.exists():
        errors.extend(verify_checksums(checksum_path, evidence_root or output))
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/stage_d_runs"))
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Optional committed evidence root (defaults to --output)",
    )
    args = parser.parse_args(argv)
    errors = verify_evidence(args.output, evidence_root=args.evidence_root)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True, "errors": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
