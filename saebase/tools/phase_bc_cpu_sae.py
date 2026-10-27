"""Check a genuinely downloaded Stage B/C SAE checkpoint on CPU.

No language-model weights are loaded. A passed SAE-only record must not be
presented as successful end-to-end model/intervention verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import torch

from saebase.sae.coverage import load_coverage
from saebase.sae.loading import load_sae_for_profile
from saebase.sae.registry import get_profile
from saebase.store import sha256_state_dict

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "conf" / "coverage" / "stage_bc.json"
SCHEMA = "saebase.coverage.cpu_sae_acceptance.v1"


def verify_cpu_sae(sae_key: str, layer: int, catalog_path: str | Path = DEFAULT_CATALOG) -> dict:
    entry = load_coverage(catalog_path).find(sae_key)
    if layer not in entry.cpu_probe_layers:
        raise ValueError("requested layer not selected for CPU SAE-only verification")
    source = entry.source(layer)
    sae, metadata = load_sae_for_profile(sae_key, layer, device="cpu")
    profile = get_profile(sae_key)
    if metadata["hook_name"] != profile.hook_name(layer):
        raise ValueError("native SAE hook disagrees with selected registered hook")
    d_in = int(sae.cfg.d_in)
    d_sae = int(sae.cfg.d_sae)
    if d_in <= 0 or d_sae <= 0 or d_sae <= d_in:
        raise ValueError("unexpected SAE dimensions")
    sae.eval()
    try:
        ref = next(sae.parameters())
    except StopIteration:
        raise ValueError("SAE has no loaded parameter weights") from None
    if ref.device.type != "cpu":
        raise ValueError("SAE-only acceptance must run on CPU")
    generator = torch.Generator(device="cpu").manual_seed(2026 + int(layer))
    x = torch.randn((2, d_in), generator=generator, dtype=ref.dtype)
    with torch.no_grad():
        z = sae.encode(x)
        recon = sae.decode(z)
        repeated = sae.encode(x)
    if z.shape != (2, d_sae) or recon.shape != x.shape:
        raise ValueError("actual encoder or decoder output dimensions invalid")
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(recon).all()):
        raise ValueError("original weights produced nonfinite activations")
    if not torch.equal(z, repeated):
        raise ValueError("SAE encoding is not deterministic")
    record = {
        "schema": SCHEMA,
        "status": "sae_checkpoint_verified_only",
        "model_sae_pair_verified": False,
        "model_loaded": False,
        "sae_key": sae_key,
        "model_key": profile.model_key,
        "layer": layer,
        "hook_name": profile.hook_name(layer),
        "source": source,
        "sae_state_sha256": sha256_state_dict(sae),
        "d_in": d_in,
        "d_sae": d_sae,
        "synthetic_probe_l0": [int(torch.count_nonzero(z[i]).item()) for i in range(2)],
        "synthetic_probe_reconstruction_mse": float(
            torch.mean((x.float() - recon.float()) ** 2).item()
        ),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-key", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--output", type=Path, default=Path("stage-bc-evidence"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f"{args.sae_key}_l{args.layer}.json"
    try:
        record = verify_cpu_sae(args.sae_key, args.layer, args.catalog)
    except Exception as exc:
        message = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            message = message.replace(token, "[redacted]")
        record = {
            "schema": SCHEMA,
            "status": "blocked",
            "model_sae_pair_verified": False,
            "model_loaded": False,
            "sae_key": args.sae_key,
            "layer": args.layer,
            "error_type": type(exc).__name__,
            "error": message[:1000],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    dest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record["status"] == "sae_checkpoint_verified_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
