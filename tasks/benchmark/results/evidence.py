"""Purpose: Read site evaluation evidence, compare numeric records and summarize support from actual completed measurements.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]

EVIDENCE = Path("conf/coverage/stage_d_evidence")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_obj(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe evidence file: {relative}")
    return path


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


def verify_site_checksums(manifest: Path, root: Path) -> list[str]:
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
        expected_digest, rel = parts[0], parts[-1]
        if rel == "checksums.sha256" or rel.endswith("/checksums.sha256"):
            errors.append(f"{manifest}:{line_no}: manifest must not hash itself")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"missing file for checksum: {rel}")
            continue
        got = digest(path)
        if got != expected_digest:
            errors.append(f"checksum mismatch for {rel}")
    return errors


def verify_evidence(output: Path, *, evidence_root: Path | None = None) -> list[str]:
    from tasks.benchmark.sites.catalog import build_site_records, load_coverage
    from tasks.benchmark.sites.records import load_status, read_acceptance

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
        errors.extend(verify_site_checksums(checksum_path, evidence_root or output))
    return errors


def site_main(argv=None) -> int:

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


EVIDENCE_REL = Path("conf/coverage/stage_d_evidence")


def _evidence_file(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError(f"invalid evidence path: {rel!r}")
    base = (root / EVIDENCE_REL).resolve()
    candidate = (base / rel).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"missing or unsafe evidence file: {rel!r}")
    return candidate


def validate_data(catalog: dict, final: dict, status: dict, root: Path) -> dict:
    """Validate identity, evidence reports, ledgers, and reported totals."""
    if catalog.get("schema") != "saebase.coverage.stage_bc.v1":
        raise ValueError("unexpected catalog schema")
    if final.get("schema") != "saebase.stage_d.final_matrix.v1":
        raise ValueError("unexpected final matrix schema")
    if status.get("schema") != "saebase.stage_d.status.v1":
        raise ValueError("unexpected status schema")
    targets, rows = catalog.get("targets"), final.get("rows")
    if not isinstance(targets, list) or not isinstance(rows, list):
        raise ValueError("missing catalog targets or final matrix rows")
    expected = {}
    for target in targets:
        for layer in target["selected_layers"]:
            key = (target["model_key"], target["sae_key"], layer)
            if key in expected:
                raise ValueError(f"duplicate catalog site: {key}")
            expected[key] = target
    found, ledger_keys = {}, set()
    counts = dict(pair=0, benchmark=0, reliability=0, population=0, smoke=0)
    states = {}
    for row in rows:
        key = (row["model_key"], row["sae_key"], row["layer"])
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate final matrix row: {key}")
        found[key] = row
        hook = row["hook_name"]
        state = row["status"]
        states[state] = states.get(state, 0) + 1
        ledger_key = f"{key[0]}|{key[1]}|L{key[2]}|{hook}"
        ledger_keys.add(ledger_key)
        ledger = status.get("sites", {}).get(ledger_key)
        if not isinstance(ledger, dict) or ledger.get("status") != state:
            raise ValueError(f"status mismatch at {key}")
        p = row.get("pair_verified") is True
        b = row.get("benchmark_verified") is True
        a = row.get("reliability_verified") is True
        if (a and not b) or (b and not p):
            raise ValueError(f"verification hierarchy mismatch at {key}")
        counts["pair"] += int(p)
        counts["benchmark"] += int(b)
        counts["reliability"] += int(a)
        counts["population"] += int(a and b and row.get("protocol") == "population")
        counts["smoke"] += int(row.get("protocol") == "smoke_only")
        if p:
            # Older accepted CPU pairs have separate pre-Stage-D provenance.
            # Never manufacture a model SHA when the current matrix omits it.
            for name in ("model_state_sha256", "sae_state_sha256"):
                value = row.get(name)
                if value is not None and re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
                    raise ValueError(f"malformed weight hash at {key}: {name}")
                if value is None and not row.get("prior_pair_verified"):
                    raise ValueError(f"missing new-pair weight hash at {key}: {name}")
            if row.get("acceptance_path"):
                _evidence_file(root, row["acceptance_path"])
            elif not row.get("prior_pair_verified"):
                raise ValueError(f"missing pair acceptance at {key}")
        if b or a:
            report = json_obj(_evidence_file(root, row.get("m3m4_summary")))
            for field, wanted in zip(("model_key", "sae_key", "layer", "hook_name"), (*key, hook)):
                if report.get(field) != wanted:
                    raise ValueError(f"{field} mismatch in report at {key}")
            if report.get("status") != "passed" or report.get("protocol") != row.get("protocol"):
                raise ValueError(f"non-passing or wrong-protocol report at {key}")
            if row.get("protocol") == "population":
                if report.get("n_independent_evaluation_groups", 0) < 64:
                    raise ValueError(f"insufficient independent groups at {key}")
                if report.get("feature_selection_split") != "selection":
                    raise ValueError(f"selection/evaluation split mismatch at {key}")
            if a and report.get("reliability_metric_audits", 0) < 1:
                raise ValueError(f"missing reliability audit at {key}")
    if set(found) != set(expected):
        raise ValueError(f"missing catalog layer sites: {set(expected) - set(found)}")
    if set(status.get("sites", {})) != ledger_keys or status.get("n_sites") != len(rows):
        raise ValueError("status ledger coverage differs from final matrix")
    for field, computed in (
        ("n_sites", len(rows)),
        ("n_pair_verified", counts["pair"]),
        ("n_population_m3m4", counts["population"]),
        ("n_smoke_only_m3m4", counts["smoke"]),
    ):
        if final.get(field) != computed:
            raise ValueError(f"{field}: expected {computed}, found {final.get(field)}")
    if final.get("counts") != states:
        raise ValueError("final matrix state counts do not match rows")
    return dict(
        catalog=targets,
        rows=found,
        total=len(rows),
        profiles=len(targets),
        date=final["updated_at_utc"].split("T")[0],
        **counts,
    )


def load_support(root: Path = ROOT) -> dict:
    return validate_data(
        json_obj(root / "conf/coverage/stage_bc.json"),
        json_obj(root / EVIDENCE_REL / "final/stage_d_final_matrix.json"),
        json_obj(root / EVIDENCE_REL / "status.json"),
        root,
    )


if __name__ == "__main__":
    raise SystemExit(site_main())
