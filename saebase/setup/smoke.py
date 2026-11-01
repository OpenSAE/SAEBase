"""An opt-in real model/SAE smoke; only a real GPU run can produce a PASS.

Use one fresh worker per selected Model/SAE site. No source modifications,
checkpoint substitutions or simulated numerical outcomes are permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any


def smoke(model_key: str, sae_key: str, layer: int, prompt: str) -> dict[str, Any]:
    """Load exact registered assets and exercise the public capture interface."""
    import torch
    from saebase.api import load_registered, capture_features

    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_ENV_NO_CUDA")
    bundle = load_registered(model_key=model_key, sae_key=sae_key, layer=layer, device="cuda")
    snapshot = capture_features(bundle.model, bundle.sae, site=bundle.site, prompt=prompt)
    codes = snapshot.codes
    if codes.numel() < 1 or not torch.isfinite(codes).all().item():
        raise RuntimeError("Real SAE capture is missing or non-finite")
    identity: dict[str, Any] = bundle.identity.to_dict()
    return {
        "status": "PASS",
        "model_key": model_key,
        "sae_key": sae_key,
        "layer": layer,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "shape": list(codes.shape),
        "identity": identity,
        "scope": "one actual prompt and one registered site only",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--allow-gpu", action="store_true")
    args = p.parse_args()
    if not (args.allow_network and args.allow_gpu):
        p.error("Both network checkpoint access and real GPU execution require explicit approval")
    print(json.dumps(smoke(args.model, args.sae, args.layer, args.prompt), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
