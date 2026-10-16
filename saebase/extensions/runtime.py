"""Explicit third-party model/SAE loading with shape and site validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity
from .registry import external_model, external_sae


@dataclass(frozen=True)
class LoadedExternalPair:
    model: ModelAdapter
    sae: Any
    identity: SAEIdentity
    verification: str = "loaded_unverified"

    @property
    def site(self) -> ActivationSite:
        return self.identity.site


def validate_external_target(target: Any) -> None:
    """Offline preflight for an external benchmark target, before weights."""
    model = external_model(target.model_key)
    third_party = any(key.startswith("ext.") for key in (target.model_key, *target.sae_keys))
    if not third_party:
        return
    if model is None:
        raise ValueError(f"external target model not registered: {target.model_key}")
    for key in target.sae_keys:
        sae = external_sae(key)
        if sae is None:
            raise ValueError(f"external target SAE not registered: {key}")
        if sae.model_key != target.model_key:
            raise ValueError(f"SAE {key} targets {sae.model_key}, not {target.model_key}")
        if (sae.site.layer, sae.site.hook_name) != (target.site.layer, target.site.hook_name):
            raise ValueError("benchmark hook and layer differ from external SAE registration")
        expected = next(
            (
                site.width
                for site in model.hooks
                if site.site.layer == target.site.layer
                and site.site.hook_name == target.site.hook_name
            ),
            None,
        )
        if expected != sae.d_in:
            raise ValueError("benchmark activation width differs from registered SAE input")


def load_external_pair(
    *,
    model_key: str,
    sae_key: str,
    layer: int,
    device: str = "cpu",
    probe_tokens: torch.Tensor | None = None,
) -> LoadedExternalPair:
    """Load explicitly registered providers. A token probe is optional.

    A successful probe tests site/SAE mechanics on these token IDs only; it
    never certifies checkpoint provenance, scientific quality or population
    reliability. Full model/SAE acceptance remains a separate protocol.
    """
    model_spec = external_model(model_key)
    sae_spec = external_sae(sae_key)
    if model_spec is None or sae_spec is None:
        raise ValueError("external model and SAE must both be explicitly registered")
    if sae_spec.model_key != model_key or sae_spec.site.layer != layer:
        raise ValueError("external model/SAE identity or layer mismatch")
    model_hook = next(
        (
            h
            for h in model_spec.hooks
            if h.site.layer == layer and h.site.hook_name == sae_spec.site.hook_name
        ),
        None,
    )
    if model_hook is None or model_hook.width != sae_spec.d_in:
        raise ValueError("external model hook/width differs from SAE specification")
    loaded_model = model_spec.factory(device)
    adapter = (
        loaded_model
        if isinstance(loaded_model, ModelAdapter)
        else ModelAdapter(loaded_model, backend=model_spec.backend)
    )
    adapter.capabilities.require("capture")
    adapter.capabilities.require("patch")

    sae = sae_spec.factory(device)
    config = getattr(sae, "cfg", None)
    if config is None or not hasattr(config, "d_in") or not hasattr(config, "d_sae"):
        raise ValueError("external SAE must expose cfg.d_in and cfg.d_sae")
    if int(config.d_in) != sae_spec.d_in or int(config.d_sae) != sae_spec.d_sae:
        raise ValueError("loaded SAE dimensions differ from registered dimensions")
    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise TypeError("external SAE must implement encode and decode")
    identity = SAEIdentity.from_loaded(
        sae=sae,
        model_key=model_key,
        model_revision=model_spec.revision,
        revision=sae_spec.revision,
        metadata={
            "model_key": model_key,
            "sae_key": sae_key,
            "sae_release": sae_spec.source,
            "sae_id": sae_spec.sae_id,
            "hook_name": sae_spec.site.hook_name,
            "layer": sae_spec.site.layer,
            "loader": "external_plugin",
            "normalization": sae_spec.normalization,
            "d_in": sae_spec.d_in,
            "d_sae": sae_spec.d_sae,
            "model_revision": model_spec.revision,
        },
    )
    if identity.dtype is not None and identity.dtype != f"torch.{sae_spec.dtype}":
        raise ValueError("loaded SAE dtype differs from declared dtype")
    if probe_tokens is None:
        return LoadedExternalPair(adapter, sae, identity)

    with torch.no_grad():
        activations = adapter.capture(tokens=probe_tokens, site=sae_spec.site)
        if activations.numel() != sae_spec.d_in:
            raise ValueError("actual captured model hook dimension differs from SAE input")
        expected_dtype = getattr(torch, sae_spec.dtype)
        if activations.dtype != expected_dtype:
            raise ValueError("captured hook dtype differs from declared SAE dtype")
        z = sae.encode(activations.unsqueeze(0))
        if (
            not isinstance(z, torch.Tensor)
            or z.shape != (1, sae_spec.d_sae)
            or not bool(torch.isfinite(z).all())
        ):
            raise ValueError("SAE encode must return finite [1, d_sae] features")
        decoded = sae.decode(z)
        if (
            not isinstance(decoded, torch.Tensor)
            or decoded.shape != (1, sae_spec.d_in)
            or not bool(torch.isfinite(decoded).all())
        ):
            raise ValueError("SAE decode must return finite [1, d_in] activations")
        adapter.assert_identity_patch(tokens=probe_tokens, site=sae_spec.site)
    return LoadedExternalPair(adapter, sae, identity, "probe_passed_not_checkpoint_certified")
