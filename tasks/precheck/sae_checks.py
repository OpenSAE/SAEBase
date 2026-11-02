"""One actual SAE capture/encode/decode check at an explicitly selected site."""

from __future__ import annotations

import hashlib
from typing import Any


def select_sites(
    model_cfg: Any, selection: Any, *, layer: int | None = None
) -> list[dict[str, Any]]:
    """Model-owned lists, never a second global SAE Registry.

    A free-text SAE key must still refer to a Model-owned profile. The legacy
    profile table is used only to resolve current checkpoint and hook parameters.
    """
    from saebase.api.sae_profiles import get_profile

    model = str(model_cfg.name)
    available = list(model_cfg.get("sae_profiles", []))
    defaults = list(model_cfg.get("default_saes", []))
    if selection in ("default", None):
        chosen = defaults
    elif selection == "all":
        chosen = available
    elif isinstance(selection, str):
        chosen = [selection]
    else:
        chosen = list(selection)
    if not chosen:
        raise ValueError(
            f"Model {model!r} has no default SAE; set model.default_saes or task.sae_list explicitly"
        )
    rows = []
    seen = set()
    for entry in chosen:
        key = str(entry.get("sae", entry.get("key"))) if hasattr(entry, "get") else str(entry)
        if key not in available:
            raise ValueError(
                f"SAE {key!r} is not in model {model!r} sae_profiles; configure that Model first"
            )
        profile = get_profile(key)
        if model not in profile.allowed_model_keys():
            raise ValueError(f"SAE {key!r} does not support model {model!r}")
        requested = entry.get("layer", layer) if hasattr(entry, "get") else layer
        site_layer = int(requested) if requested is not None else int(profile.layers[-1])
        if site_layer not in profile.layers:
            raise ValueError(
                f"SAE {key!r} supports layers {list(profile.layers)}, not {site_layer}"
            )
        identity = (key, site_layer)
        if identity in seen:
            raise ValueError(f"Duplicate SAE check {identity}")
        seen.add(identity)
        rows.append({"sae_key": key, "layer": site_layer, "hook": profile.hook_name(site_layer)})
    return rows


def check_real_sae(
    *, model_key: str, sae_key: str, layer: int, prompt: str, device: str
) -> dict[str, Any]:
    """Use real checkpoints and real tokens; neither fake activations nor random smoke."""
    import torch
    from saebase.api import capture_features, load_registered

    if not prompt or not prompt.strip():
        raise ValueError("Precheck requires a nonempty real prompt for SAE capture")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_RESOURCE: no CUDA device available")
    pair = load_registered(
        sae_key=sae_key, layer=layer, model_key=model_key, device=device, require_patch=False
    )
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    h, z, rebuilt = snapshot.activation, snapshot.codes, snapshot.reconstruction
    if h.ndim != 1 or z.ndim != 1 or rebuilt.shape != h.shape:
        raise RuntimeError("SAE capture/encode/decode returned invalid shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (h, z, rebuilt)):
        raise RuntimeError("SAE activation, latent, or reconstruction is non-finite")
    if h.numel() != pair.identity.d_in or z.numel() != pair.identity.d_sae:
        raise RuntimeError("SAE checkpoint dimensions do not match the captured activation")
    return {
        "status": "PASS",
        "scope": "single real prompt / single SAE site; not scientific SAE quality acceptance",
        "model_key": model_key,
        "sae_key": sae_key,
        "layer": layer,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "identity": pair.identity.to_dict(),
        "activation_shape": list(h.shape),
        "latent_shape": list(z.shape),
        "l0": int(torch.count_nonzero(z).item()),
        "reconstruction_mse": float(torch.mean((h.float() - rebuilt.float()) ** 2).item()),
    }
