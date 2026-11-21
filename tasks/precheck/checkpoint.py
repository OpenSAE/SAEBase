"""SAE-only resource preparation and native encode/decode probe.

Owner: precheck. Reuses the registered SAE loader; never loads a language model.
The synthetic probe establishes software usability, not SAE research quality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from conf import load_group


def probe_checkpoint(sae_key: str, layer: int, *, device: str = "auto") -> dict:
    import torch

    from saebase.api.sae_loading import load_sae_for_profile
    from saebase.api.sae_profiles import get_profile
    from saebase.store import sha256_state_dict

    profile = get_profile(sae_key)
    sae, metadata = load_sae_for_profile(sae_key, layer, device=device)
    ref = next(sae.parameters())
    generator = torch.Generator(device=ref.device).manual_seed(2026 + layer)
    x = torch.randn((2, int(sae.cfg.d_in)), generator=generator, dtype=ref.dtype, device=ref.device)
    with torch.inference_mode():
        z = sae.encode(x)
        rebuilt = sae.decode(z)
    if z.shape != (2, int(sae.cfg.d_sae)) or rebuilt.shape != x.shape:
        raise ValueError("Native SAE encode/decode shapes differ from its configuration")
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(rebuilt).all()):
        raise ValueError("Native SAE produced non-finite values")
    return {
        "schema": "saebase.precheck.sae_checkpoint.v1",
        "status": "PASS",
        "model_loaded": False,
        "model_sae_pair_verified": False,
        "scope": "SAE-only native encode/decode with synthetic inputs",
        "sae_key": sae_key,
        "model_key": profile.model_key,
        "layer": layer,
        "hook_name": metadata["hook_name"],
        "d_in": int(sae.cfg.d_in),
        "d_sae": int(sae.cfg.d_sae),
        "device": str(ref.device),
        "source": {
            "repo_id": profile.repo_id or profile.sae_release,
            "sae_id": profile.sae_id(layer),
            "folder": profile.sae_folder(layer),
        },
        "sae_state_sha256": sha256_state_dict(sae),
        "synthetic_probe_l0": torch.count_nonzero(z, dim=1).cpu().tolist(),
        "synthetic_probe_reconstruction_mse": ((x.float() - rebuilt.float()) ** 2)
        .mean(1)
        .cpu()
        .tolist(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    import argparse

    from saebase.api.sae_profiles import get_profile

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--sae-key")
    choice.add_argument("--model", help="Registered model/profile alias")
    parser.add_argument(
        "--coverage-catalog",
        type=Path,
        help="Optional coverage protocol whose source fields to retain",
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = get_profile(args.sae_key or args.model)
    layer = args.layer
    if layer is None:
        config = load_group("model", profile.model_key).model
        from saebase.runtime.sae_operations import resolve_sae

        layer = resolve_sae(config, profile.key).layer
    try:
        result = probe_checkpoint(profile.key, layer, device=args.device)
        result.update(
            schema="saebase.coverage.cpu_sae_acceptance.v1"
            if args.sae_key
            else "saebase.phase_a.sae_only_acceptance.v1",
            status="sae_checkpoint_verified_only",
        )
        if args.coverage_catalog:
            from tasks.benchmark.sites.catalog import load_coverage

            result["source"] = load_coverage(args.coverage_catalog).find(profile.key).source(layer)
    except Exception as exc:
        from tasks.precheck.report import error_result

        result = error_result(
            exc, stage="sae_checkpoint", attempted=["load SAE and native encode/decode"]
        )
        result.update(model_loaded=False, model_sae_pair_verified=False)
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{profile.key}_l{layer}" if args.sae_key else args.model
    from saebase.runtime.experiment_results import write_json

    write_json(args.output / f"{name}.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "sae_checkpoint_verified_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
