"""Stage D single-site GPU/CPU model–SAE pair acceptance.

Supports every catalog profile and selected layer. Writes checksum-bound
acceptance JSON under ``<output>/acceptance/<model>/<sae>/L<layer>.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from saebase.sae.coverage import load_coverage
from saebase.sae.registry import get_profile
from saebase.sae.stage_d import (
    PROMPTS,
    append_attempt,
    failure_record,
    should_skip_resume,
    update_status,
    verify_pair_site,
    write_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_one(
    *,
    sae_key: str,
    layer: int,
    device: str,
    output: Path,
    resume: bool,
    model_key_override: str | None = None,
) -> dict:
    catalog = load_coverage(CATALOG)
    entry = catalog.find(sae_key)
    if layer not in entry.selected_layers:
        raise ValueError(f"layer {layer} is not a Stage B/C selected layer for {sae_key}")
    profile = get_profile(sae_key)
    site_id = f"{entry.model_key}|{sae_key}|L{layer}|{profile.hook_name(layer)}"
    started = datetime.now(timezone.utc).isoformat()
    skipped = should_skip_resume(
        output, model_key=entry.model_key, sae_key=sae_key, layer=layer, resume=resume,
    )
    if skipped is not None:
        update_status(output, site_id, {
            "status": "pair_verified",
            "pair_verified": True,
            "acceptance_path": str(
                output / "acceptance" / entry.model_key / sae_key / f"L{layer}.json"
            ),
            "resumed": True,
            "sae_state_sha256": skipped.get("sae_state_sha256"),
            "model_state_sha256": skipped.get("model_state_sha256"),
        })
        append_attempt(output, {
            "site_id": site_id, "event": "resume_skip", "started_at_utc": started,
            "status": "pair_verified",
        })
        return skipped

    append_attempt(output, {
        "site_id": site_id, "event": "start", "device": device,
        "started_at_utc": started,
    })
    try:
        record = verify_pair_site(
            sae_key=sae_key,
            layer=layer,
            device=device,
            prompts=PROMPTS,
            model_key_override=model_key_override,
        )
        record["git_commit"] = _git_commit()
        record["python"] = platform.python_version()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        if torch.cuda.is_available() and device.startswith("cuda"):
            record["gpu_name"] = torch.cuda.get_device_name(0)
        # Recompute checksum after enriching metadata.
        from saebase.sae.stage_d import canonical_payload_sha256
        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(output, site_id, {
            "status": "pair_verified",
            "pair_verified": True,
            "acceptance_path": str(path),
            "sae_state_sha256": record["sae_state_sha256"],
            "model_state_sha256": record["model_state_sha256"],
            "error_type": None,
            "error": None,
        })
        append_attempt(output, {
            "site_id": site_id, "event": "end", "status": "passed",
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "peak_cuda_allocated_bytes": record.get("peak_cuda_allocated_bytes"),
        })
        return record
    except Exception as exc:
        record = failure_record(
            sae_key=sae_key, layer=layer, device=device, exc=exc,
            model_key=entry.model_key,
        )
        record["git_commit"] = _git_commit()
        from saebase.sae.stage_d import canonical_payload_sha256
        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(output, site_id, {
            "status": record["status"],
            "pair_verified": False,
            "acceptance_path": str(path),
            "error_type": record["error_type"],
            "error": record["error"],
        })
        append_attempt(output, {
            "site_id": site_id, "event": "end", "status": record["status"],
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": record["error_type"],
        })
        return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-key", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/stage_d_runs"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--model-key-override", default=None)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    record = run_one(
        sae_key=args.sae_key,
        layer=args.layer,
        device=args.device,
        output=args.output,
        resume=args.resume,
        model_key_override=args.model_key_override,
    )
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
