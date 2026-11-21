"""Stable entrypoints for SAEBase Phase B public research API.

For actual SAE/model checkpoint loading use ``load_registered()``. The API
also accepts custom model/SAE objects and needs no downloads for small tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathlib import Path as _Path

# Keep existing module names after grouping their source files by purpose.
__path__.append(str(_Path(__file__).parent / "sae_adapters"))

from .features import (
    FeaturePatch,
    FeatureSnapshot,
    InterventionResult,
    capture_features,
    encode_activations,
    coalition_replacements,
    edit_features,
    evaluate_edit,
)
from .components import BlockComponents, capture_block_components
from .nnsight_local import NNsightLocal, TraceLocation
from .metadata import SAEIdentity
from .model import ActivationSite, ModelAdapter, ModelCapabilities, UnsupportedCapability


@dataclass(frozen=True)
class LoadedResearchPair:
    model: ModelAdapter
    sae: Any
    identity: SAEIdentity

    @property
    def site(self) -> ActivationSite:
        return self.identity.site


def load_registered(
    *,
    sae_key: str,
    layer: int,
    device: str = "auto",
    model_key: str | None = None,
    require_patch: bool = True,
) -> LoadedResearchPair:
    """Load a bundled model/profile pair, with explicit site/provenance validation.

    Loads public weights on demand and requires model dependencies. Registration
    metadata does not by itself prove real checkpoint compatibility.
    """
    from saebase.api.sae_loading import load_model_and_sae

    model, sae, metadata = load_model_and_sae(
        sae_key, layer, device=device, model_key_override=model_key
    )
    identity = SAEIdentity.from_loaded(
        sae=sae,
        metadata=metadata,
        model_key=model_key,
    )
    from saebase.api.sae_profiles import get_profile

    expected_hook = get_profile(sae_key).hook_name(layer)
    if identity.hook_name != expected_hook:
        raise ValueError(
            f"Loaded SAE hook {identity.hook_name!r} differs from registered site {expected_hook!r}"
        )
    adapter = ModelAdapter(model)
    adapter.capabilities.require("capture")
    if require_patch:
        adapter.capabilities.require("patch")
    if adapter.backend == "hf_hooks":
        if identity.hook_name != f"blocks.{identity.layer}.hook_resid_post":
            raise UnsupportedCapability(
                "requested SAE hook is incompatible with HF layer-output patch backend"
            )
    return LoadedResearchPair(model=adapter, sae=sae, identity=identity)


__all__ = [
    "ActivationSite",
    "ModelAdapter",
    "ModelCapabilities",
    "UnsupportedCapability",
    "SAEIdentity",
    "LoadedResearchPair",
    "load_registered",
    "BlockComponents",
    "capture_block_components",
    "NNsightLocal",
    "TraceLocation",
    "FeatureSnapshot",
    "FeaturePatch",
    "InterventionResult",
    "capture_features",
    "encode_activations",
    "edit_features",
    "coalition_replacements",
    "evaluate_edit",
]
