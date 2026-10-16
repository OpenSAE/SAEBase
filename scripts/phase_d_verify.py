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
    PROTOCOL_PAIR,
    append_attempt,
    failure_record,
    relative_under,
    resolve_evaluation_host,
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
    evaluation_host, auto_override = resolve_evaluation_host(
        catalog_model_key=entry.model_key,
        sae_key=sae_key,
        model_key_override=model_key_override,
    )
    site_id = f"{evaluation_host}|{sae_key}|L{layer}|{profile.hook_name(layer)}"
    started = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    # Fingerprint gates science-invalidating drift. Legacy pair artifacts without
    # git_commit/dataset fields still resume when host/hook/protocol match;
    # Mistral LN and Gemma IT host fixes are re-run with --no-resume.
    expected = {
        "model_key": evaluation_host,
        "evaluation_host_model_key": evaluation_host,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": profile.hook_name(layer),
        "protocol": PROTOCOL_PAIR,
    }
    skipped = should_skip_resume(
        output,
        model_key=evaluation_host,
        sae_key=sae_key,
        layer=layer,
        resume=resume,
        expected=expected,
    )
    if skipped is not None:
        update_status(output, site_id, {
            "status": "pair_verified",
            "pair_verified": True,
            "acceptance_path": relative_under(
                output,
                output / "acceptance" / evaluation_host / sae_key / f"L{layer}.json",
            ),
            "resumed": True,
            "protocol": PROTOCOL_PAIR,
            "sae_state_sha256": skipped.get("sae_state_sha256"),
            "model_state_sha256": skipped.get("model_state_sha256"),
            "evaluation_host_model_key": evaluation_host,
            "loaded_model_name": skipped.get("loaded_model_name"),
        })
        append_attempt(output, {
            "site_id": site_id, "event": "resume_skip", "started_at_utc": started,
            "status": "pair_verified",
        })
        return skipped

    append_attempt(output, {
        "site_id": site_id, "event": "start", "device": device,
        "started_at_utc": started,
        "evaluation_host_model_key": evaluation_host,
        "model_key_override": auto_override,
    })
    try:
        record = verify_pair_site(
            sae_key=sae_key,
            layer=layer,
            device=device,
            prompts=PROMPTS,
            model_key_override=auto_override,
            catalog_model_key=entry.model_key,
            git_commit=git_commit,
        )
        record["git_commit"] = git_commit
        record["python"] = platform.python_version()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        if torch.cuda.is_available() and device.startswith("cuda"):
            record["gpu_name"] = torch.cuda.get_device_name(0)
        from saebase.sae.stage_d import canonical_payload_sha256
        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(output, site_id, {
            "status": "pair_verified",
            "pair_verified": True,
            "acceptance_path": relative_under(output, path),
            "protocol": PROTOCOL_PAIR,
            "sae_state_sha256": record["sae_state_sha256"],
            "model_state_sha256": record["model_state_sha256"],
            "evaluation_host_model_key": evaluation_host,
            "trained_on_model_key": record.get("trained_on_model_key"),
            "transfer_kind": record.get("transfer_kind"),
            "loaded_model_name": record.get("loaded_model_name"),
            "error_type": None,
            "error": None,
        })
        append_attempt(output, {
            "site_id": site_id, "event": "end", "status": "passed",
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "peak_cuda_allocated_bytes": record.get("peak_cuda_allocated_bytes"),
            "supersedes_prior": True,
        })
        return record
    except Exception as exc:
        record = failure_record(
            sae_key=sae_key, layer=layer, device=device, exc=exc,
            model_key=evaluation_host,
        )
        record["git_commit"] = git_commit
        from saebase.sae.stage_d import canonical_payload_sha256
        record.pop("artifact_sha256", None)
        record["artifact_sha256"] = canonical_payload_sha256(record)
        path = write_acceptance(output, record)
        update_status(output, site_id, {
            "status": record["status"],
            "pair_verified": False,
            "acceptance_path": relative_under(output, path),
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
        raise SystemExit("CUDA requested but unavailable")
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
