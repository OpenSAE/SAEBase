#!/usr/bin/env python3
"""Six-model Stage A coverage preflight and opt-in real checkpoint acceptance.

Examples:
    saebase-stage-a --dry-run
    saebase-stage-a --real --target qwen35_2b --device cpu --output runs/stage-a
    saebase-stage-a --real --target qwen3_8b --target llama31_8b --device cuda
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from saebase.coverage.stage_a import STAGE_A, get_pair, plan_stage_a, run_real_pair, write_evidence


def _blocked_reason(error: Exception) -> str:
    """Report actionable categories without putting HF tokens/URLs in artifacts."""
    name = type(error).__name__
    detail = str(error).lower()
    if "gated" in detail or "restricted" in detail or "access to model" in detail or "401" in detail or "403" in detail:
        return "model_or_checkpoint_access_not_granted"
    if "out of memory" in detail or "cuda memory" in detail:
        return "hardware_memory_insufficient"
    if name in {"ImportError", "ModuleNotFoundError"} or "requires transformers" in detail:
        return "runtime_dependencies_unavailable"
    if "not found" in detail or "repository not found" in detail:
        return "upstream_checkpoint_not_found"
    return "verification_error"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--target", action="append", default=None,
                        help="Model key (repeat for multiple; omit to select all)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("stage-a-results"))
    args = parser.parse_args(argv)

    requested = tuple(args.target) if args.target else tuple(x.model_key for x in STAGE_A)
    plan = plan_stage_a(requested)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0

    if args.device == "cpu" and any(get_pair(k).scale == "8B" for k in requested):
        parser.error("8B acceptance requires --device cuda and suitable memory")
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    failed = []
    for model_key in requested:
        item = get_pair(model_key)
        try:
            evidence = run_real_pair(item, device=args.device)
            print(f"{model_key}: real_weights_verified", flush=True)
        except Exception as exc:
            reason = _blocked_reason(exc)
            evidence = {
                "schema": "saebase.stage_a.coverage.v1",
                "status": "blocked",
                "model_key": model_key,
                "sae_key": item.sae_key,
                "reason": reason,
                "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                "not_verified": True,
            }
            failed.append(model_key)
            print(f"{model_key}: blocked ({reason}; {type(exc).__name__})",
                  file=sys.stderr, flush=True)
            # Diagnostics are opt-in and reserved for publicly accessible
            # checkpoints: traceback strings can contain private source paths.
            import os
            if os.environ.get("SAEBASE_STAGE_A_DEBUG") == "1":
                import traceback
                traceback.print_exc(file=sys.stderr)
        path = folder / f"{model_key}.json"
        write_evidence(path, evidence)
    summary = {
        "schema": "saebase.stage_a.run_summary.v1",
        "requested": list(requested),
        "real_verified": [x for x in requested if x not in failed],
        "blocked": failed,
        "evidence_dir": str(folder),
        "complete": not failed,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
