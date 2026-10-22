#!/usr/bin/env python3
"""Bounded P1 real-checkpoint smoke, opt-in only; no benchmark expansion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from saebase.api import capture_features, evaluate_edit, load_registered
from saebase.sae.registry import get_profile
from saebase.store.identity import sha256_state_dict


PROMPTS = (
    "The capital of France is",
    "A scientist explained the results carefully.",
    "Small birds fly across the morning sky.",
)


@torch.no_grad()
def smoke(*, sae_key: str, layer: int, device: str) -> dict:
    profile = get_profile(sae_key)
    bundle = load_registered(sae_key=sae_key, layer=layer, device=device)
    model, sae = bundle.model, bundle.sae
    site = bundle.site
    if site.hook_name != profile.hook_name(layer):
        raise ValueError("checkpoint site disagrees with registered hook")
    results = []
    for text in PROMPTS:
        tokens = model.tokens(prompt=text)
        snapshot = capture_features(model, sae, site=site, tokens=tokens)
        model.assert_identity_patch(tokens=tokens, site=site, atol=1e-4, rtol=1e-4)
        if not bool(torch.isfinite(snapshot.codes).all()):
            raise ValueError("SAE codes must be finite")
        selected = int(torch.argmax(snapshot.codes.abs()).item())
        if float(snapshot.codes[selected].abs().item()) <= 0:
            raise ValueError("no active SAE feature for this prompt")
        edited = evaluate_edit(
            model, sae, site=site, tokens=tokens,
            replacements={selected: 0.0}, verify_identity=True,
        )
        results.append({
            "text_token_count": int(tokens.shape[1]),
            "selected_feature": selected,
            "code_nonzeros": int(torch.count_nonzero(snapshot.codes).item()),
            "reconstruction_mse": float(
                ((snapshot.activation.float() - snapshot.reconstruction.float()) ** 2).mean()
            ),
            "edited_logit_delta_l2": float(edited.logit_delta.float().norm().item()),
        })
    return {
        "schema": "opensae.wp12.p1.smoke.v1",
        "status": "small_real_smoke_pass",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": profile.model_key,
        "sae_key": sae_key,
        "site": site.hook_name,
        "model_loaded_name": bundle.identity.loaded_model_name,
        "backend": model.backend,
        "engine": getattr(model.model, "_saebase_transformerlens_engine", None),
        "processing": getattr(model.model, "_saebase_weight_processing", None),
        "model_state_sha256": sha256_state_dict(model.model),
        "sae_state_sha256": sha256_state_dict(sae),
        "hook_count": len(model.available_hooks(layer=layer)),
        "precision": str(next(sae.parameters()).dtype),
        "results": results,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-key", default="pythia70m_resid_sm")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested, but no GPU is available")
    result = smoke(sae_key=args.sae_key, layer=args.layer, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"], "engine": result["engine"],
        "site": result["site"], "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
