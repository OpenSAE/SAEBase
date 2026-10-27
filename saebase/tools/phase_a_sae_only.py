"""Download and check genuine Phase A pretrained SAE weights without the base LLM.

SAE-only success is not model-SAE pair acceptance. No model is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import torch

from saebase.sae.loading import load_sae_for_profile
from saebase.sae.registry import get_profile
from saebase.store import sha256_state_dict
from saebase.tools.phase_a_verify import TARGETS


def verify_sae_only(name: str) -> dict:
    sae_key, layer, expected_width = TARGETS[name]
    profile = get_profile(sae_key)
    sae, meta = load_sae_for_profile(sae_key, layer, device="cpu")
    if int(sae.cfg.d_in) != expected_width or int(sae.cfg.d_sae) < 1:
        raise ValueError("SAE dimensions differ from registered model width")
    if meta["hook_name"] != profile.hook_name(layer):
        raise ValueError("downloaded SAE hook differs from declared checkpoint")
    sae.eval()
    with torch.inference_mode():
        ref = next(sae.parameters())
        activations = torch.zeros((1, expected_width), dtype=ref.dtype, device=ref.device)
        codes = sae.encode(activations)
        reconstruction = sae.decode(codes)
    if codes.shape != (1, int(sae.cfg.d_sae)) or reconstruction.shape != activations.shape:
        raise ValueError("SAE encoder/decoder shapes are invalid")
    if not torch.isfinite(codes).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("SAE encoder/decoder produced NaNs or infinities")
    return {
        "schema": "saebase.phase_a.sae_only_acceptance.v1",
        "status": "sae_checkpoint_verified_only",
        "model": name,
        "model_evaluated": False,
        "model_sae_pair_verified": False,
        "sae_profile": sae_key,
        "sae_repo": profile.repo_id or profile.sae_release,
        "sae_id": profile.sae_id(layer),
        "sae_state_sha256": sha256_state_dict(sae),
        "d_in": expected_width,
        "d_sae": int(sae.cfg.d_sae),
        "probe_feature_l0": int(torch.count_nonzero(codes).item()),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=TARGETS.keys(), required=True)
    p.add_argument("--output", type=Path, default=Path("phase-a-sae-only"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = verify_sae_only(args.model)
    except Exception as exc:
        error = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            error = error.replace(token, "[redacted]")
        result = {
            "schema": "saebase.phase_a.sae_only_acceptance.v1",
            "status": "blocked",
            "model": args.model,
            "model_evaluated": False,
            "model_sae_pair_verified": False,
            "error_type": type(exc).__name__,
            "error": error[:1500],
        }
    path = args.output / f"{args.model}.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "sae_checkpoint_verified_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
