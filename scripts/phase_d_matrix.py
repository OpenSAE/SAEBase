"""Resumable Stage D orchestrator over the 48-site catalog.

Schedules one model family at a time, skips proven passes under ``--resume``,
and writes environment / status / attempt ledgers under ``--output``.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from saebase.benchmarks.results import atomic_json
from saebase.sae.coverage import load_coverage
from saebase.sae.stage_d import (
    DEFAULT_MODEL_ORDER,
    build_final_matrix,
    build_site_records,
    load_stage_d_targets,
    schedule_sites,
    write_stage_d_targets,
)
from scripts.phase_d_verify import run_one

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_environment(output: Path) -> dict:
    info = {
        "schema": "saebase.stage_d.environment.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    try:
        import shutil
        usage = shutil.disk_usage("/workspace")
        info["disk_total_bytes"] = usage.total
        info["disk_free_bytes"] = usage.free
    except Exception:
        pass
    atomic_json(output / "environment.json", info)
    return info


def ensure_targets(output: Path) -> Path:
    targets = ROOT / "conf" / "coverage" / "stage_d_targets.json"
    if not targets.exists():
        write_stage_d_targets(targets)
    # Mirror into the run directory for provenance.
    payload = json.loads(targets.read_text(encoding="utf-8"))
    atomic_json(output / "stage_d_targets.json", payload)
    plan = load_coverage(CATALOG).plan()
    atomic_json(output / "catalog_snapshot.json", plan)
    return targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/stage_d_runs"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--model", default=None, help="Restrict to one model_key")
    parser.add_argument("--sae-key", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--include-prior",
        action="store_true",
        help="Also schedule the four previously CPU-verified pair sites",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Write final/stage_d_final_matrix.json from current status",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    targets_path = ensure_targets(args.output)
    write_environment(args.output)
    records = load_stage_d_targets(targets_path)
    # Refresh skip flags from the live catalog/prior set.
    live = build_site_records(load_coverage(CATALOG))
    by_id = {r.key.id: r for r in live}
    records = [by_id.get(r.key.id, r) for r in records]

    queued = schedule_sites(
        records,
        model=args.model,
        sae_key=args.sae_key,
        layer=args.layer,
        include_prior=args.include_prior,
        model_order=DEFAULT_MODEL_ORDER,
    )
    # Explicit layer/sae/model filters should still allow a prior site when asked.
    if (args.layer is not None or args.sae_key is not None) and not queued:
        queued = schedule_sites(
            records,
            model=args.model,
            sae_key=args.sae_key,
            layer=args.layer,
            include_prior=True,
            model_order=DEFAULT_MODEL_ORDER,
        )

    summary = {
        "n_queued": len(queued),
        "device": args.device,
        "dry_run": args.dry_run,
        "sites": [
            {
                "model_key": r.model_key,
                "sae_key": r.sae_key,
                "layer": r.layer,
                "hook_name": r.hook_name,
                "wave": r.execution_wave,
                "prior": r.prior_pair_verified,
            }
            for r in queued
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        if args.finalize:
            from saebase.sae.stage_d import rebuild_canonical_status
            rebuild_canonical_status(args.output, records)
            matrix = build_final_matrix(args.output, records)
            final_dir = args.output / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(final_dir / "stage_d_final_matrix.json", matrix)
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    failures = 0
    current_model = None
    for rec in queued:
        if current_model is not None and rec.model_key != current_model:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({
                "event": "model_family_complete",
                "model_key": current_model,
            }), flush=True)
        current_model = rec.model_key
        print(json.dumps({
            "event": "verify_start",
            "model_key": rec.model_key,
            "sae_key": rec.sae_key,
            "layer": rec.layer,
        }), flush=True)
        result = run_one(
            sae_key=rec.sae_key,
            layer=rec.layer,
            device=args.device,
            output=args.output,
            resume=args.resume,
        )
        ok = result.get("status") == "passed"
        if not ok:
            failures += 1
        print(json.dumps({
            "event": "verify_end",
            "status": result.get("status"),
            "model_key": rec.model_key,
            "sae_key": rec.sae_key,
            "layer": rec.layer,
            "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
        }), flush=True)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            print(json.dumps({
                "event": "resource_snapshot",
                "cuda_allocated_bytes": allocated,
                "cuda_reserved_bytes": reserved,
            }), flush=True)

    if args.finalize:
        from saebase.sae.stage_d import rebuild_canonical_status
        rebuild_canonical_status(args.output, records)
        matrix = build_final_matrix(args.output, records)
        final_dir = args.output / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if not matrix.get("stage_d_complete"):
            # Still write a progress matrix, but refuse a "complete" claim.
            matrix = dict(matrix)
            matrix["publication"] = "incomplete"
            matrix["publication_note"] = (
                "Gates not met: require 48/48 pair_verified and 48/48 population M3/M4. "
                f"pair={matrix.get('n_pair_verified')} population={matrix.get('n_population_m3m4')}"
            )
        else:
            matrix = dict(matrix)
            matrix["publication"] = "complete"
        atomic_json(final_dir / "stage_d_final_matrix.json", matrix)
        summary_md = [
            "# Stage D final matrix",
            "",
            f"Updated: {matrix['updated_at_utc']}",
            f"Sites: {matrix['n_sites']}",
            f"Publication: {matrix.get('publication')}",
            f"Pair verified: {matrix.get('n_pair_verified')}/48",
            f"Population M3/M4: {matrix.get('n_population_m3m4')}/48",
            f"Smoke-only M3/M4 (historical): {matrix.get('n_smoke_only_m3m4')}",
            "",
            "## Counts",
            "",
        ]
        for key, value in sorted(matrix["counts"].items()):
            summary_md.append(f"- `{key}`: {value}")
        summary_md.append("")
        if matrix.get("publication_note"):
            summary_md.extend(["## Note", "", matrix["publication_note"], ""])
        (final_dir / "stage_d_summary.md").write_text(
            "\n".join(summary_md) + "\n", encoding="utf-8",
        )
        if not matrix.get("stage_d_complete"):
            return 1 if failures else 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
