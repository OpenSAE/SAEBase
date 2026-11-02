"""Real CPU model/SAE acceptance for Stage B's three small ungated models.

Test actual checkpoint capture, reconstruction, identity hook patch and
one non-trivial feature intervention. Do not load larger Phase D models here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from tasks.benchmark.coverage_contract import load_coverage
from saebase.api.sae_loading import load_sae_for_profile, load_model_for_profile
from saebase.api.sae_profiles import get_profile
from saebase.store import sha256_state_dict
from models.hf_hooks_backend import load_hf_hooks_model

CATALOG = Path(__file__).resolve().parents[2] / "conf" / "coverage" / "stage_bc.json"
PROMPTS = ("The capital of France is", "The scientist explained the experiment.")


def load_cpu_pair(sae_key: str, *, device: str = "cpu"):
    """Return one real CPU ModelAdapter + SAE using the registered native site."""
    if device != "cpu":
        raise ValueError("Stage B/C acceptance is intentionally CPU-only")
    entry = load_coverage(CATALOG).find(sae_key)
    if not entry.model_pair_cpu_candidate:
        raise ValueError("model requires Phase D hardware or access")
    layer = entry.cpu_probe_layers[0]
    p = get_profile(sae_key)
    if p.model_key == "r1_distill_qwen15b":
        # Native HF decoder hook is the unprocessed resid_post used to train
        # the reasoning SAE. Avoid a duplicate HF -> TL model conversion.
        handle = load_hf_hooks_model(
            p.model_name,
            key=p.model_key,
            family=p.base.family,
            device="cpu",
            dtype=torch.bfloat16,
        )
        model_raw = handle.model
        model_name = handle.loaded_name
    else:
        model_raw, model_name = load_model_for_profile(sae_key, device="cpu")
    model = ModelAdapter(model_raw)
    sae, meta = load_sae_for_profile(sae_key, layer, device="cpu")
    site = ActivationSite(layer, p.hook_name(layer), position=-1)
    if meta["hook_name"] != site.hook_name:
        raise ValueError("SAE native checkpoint hook mismatch")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    return entry, p, model, sae, meta, site, model_name, model_raw


def verify_pair(sae_key: str, *, device: str = "cpu") -> dict:
    entry, p, model, sae, meta, site, model_name, model_raw = load_cpu_pair(sae_key, device=device)
    layer = site.layer
    observations = []
    for prompt in PROMPTS:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        if snap.activation.numel() != sae.cfg.d_in or not torch.isfinite(snap.codes).all():
            raise ValueError("loaded SAE incompatible with actual model activation")
        if torch.count_nonzero(snap.codes).item() == 0:
            raise ValueError("no active features on real test input")
        base = model.forward_logits(tokens=snap.tokens).float()
        identity = model.patch_logits(
            tokens=snap.tokens,
            site=site,
            activation=snap.activation,
        ).float()
        max_err = float(torch.max(torch.abs(base - identity)).item())
        if not 0 <= max_err <= 0.01:
            raise ValueError(f"identity patch mismatch: {max_err}")
        fid = int(torch.argmax(snap.codes.abs()).item())
        patched = edit_features(sae, snap, replacements={fid: 0.0})
        changed = model.patch_logits(
            tokens=snap.tokens,
            site=site,
            activation=patched.activation,
        ).float()
        delta = float(torch.linalg.vector_norm(base - changed).item())
        if not (delta > 1e-8 and torch.isfinite(torch.tensor(delta))):
            raise ValueError("SAE feature removal did not change real logits")
        observations.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "l0": int(torch.count_nonzero(snap.codes).item()),
                "reconstruction_mse": float(
                    torch.mean((snap.activation.float() - snap.reconstruction.float()) ** 2).item()
                ),
                "identity_max_abs_logit_error": max_err,
                "ablated_feature": fid,
                "feature_delta_logit_l2": delta,
            }
        )
    return {
        "schema": "saebase.coverage.cpu_pair_acceptance.v1",
        "status": "model_and_sae_verified",
        "sae_key": sae_key,
        "model_key": entry.model_key,
        "model_repo": model_name,
        "sae_repo": p.repo_id or p.sae_release,
        "layer": layer,
        "hook": site.hook_name,
        "model_state_sha256": sha256_state_dict(model_raw),
        "sae_state_sha256": sha256_state_dict(sae),
        "sample_count": len(observations),
        "observations": observations,
        "device": device,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-key", required=True)
    p.add_argument("--output", type=Path, default=Path("stage-bc-evidence"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        record = verify_pair(args.sae_key)
    except Exception as exc:
        err = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            err = err.replace(token, "[redacted]")
        record = {
            "schema": "saebase.coverage.cpu_pair_acceptance.v1",
            "status": "blocked",
            "sae_key": args.sae_key,
            "error_type": type(exc).__name__,
            "error": err[:1400],
            "model_and_sae_verified": False,
        }
    file = args.output / f"{args.sae_key}_pair.json"
    file.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record["status"] == "model_and_sae_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
