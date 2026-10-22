"""CPU-only index of registered and officially published Gemma Scope 2 assets.

Listing an asset never downloads model weights, and a registered SAE profile
is not proof of a successfully loaded checkpoint. Upstream manifest discovery
is opt-in and records the exact returned Hugging Face model revision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from saebase.sae.registry import list_profiles


_VARIANT = re.compile(r"^layer_(\d+)_width_([A-Za-z0-9]+)_l0_([A-Za-z0-9]+)$")


@dataclass(frozen=True)
class ScopeAsset:
    model_key: str
    repo_id: str
    revision: str | None
    source: str
    sae_id: str
    layer: int
    hook: str
    width: str | None
    l0_variant: str | None
    loader: str
    registered_profile: str | None
    status: str
    activation_family: str = "resid_post"


def _parts(sae_id: str) -> tuple[int | None, str | None, str | None]:
    for segment in sae_id.split("/"):
        match = _VARIANT.fullmatch(segment)
        if match is not None:
            layer, width, l0 = match.groups()
            return int(layer), width, l0
    return None, None, None


def registered_scope_assets(*, model_key: str | None = None) -> list[ScopeAsset]:
    """List all Gemma Scope 2 checkpoint sites registered in SAEBase."""
    output: list[ScopeAsset] = []
    for profile in list_profiles():
        repo = profile.repo_id or ""
        if not repo.startswith("google/gemma-scope-2-"):
            continue
        if model_key is not None and profile.model_key != model_key:
            continue
        for layer in profile.layers:
            sae_id = profile.sae_id(layer)
            parsed_layer, width, l0 = _parts(sae_id)
            if parsed_layer is not None and parsed_layer != layer:
                raise ValueError(f"SAE profile {profile.key} has inconsistent layer")
            output.append(
                ScopeAsset(
                    model_key=profile.model_key,
                    repo_id=repo,
                    revision=None,
                    source="registered_profile",
                    sae_id=sae_id,
                    layer=layer,
                    hook=profile.hook_name(layer),
                    width=width,
                    l0_variant=l0,
                    loader=profile.loader,
                    registered_profile=profile.key,
                    status="registered_not_live_verified",
                )
            )
    return sorted(output, key=lambda e: (e.model_key, e.layer, e.repo_id, e.sae_id))


def discover_official_scope_assets(
    repo_id: str,
    *,
    revision: str | None = None,
    info: Any | None = None,
) -> list[ScopeAsset]:
    """List real width/L0 variants from an official HF repo's file manifest.

    `info` is injectable for offline tests; otherwise this makes a
    metadata-only HF Hub call. Returned variants are indexed only when a
    repository file actually includes their precise SAE directory name.
    """
    registered = registered_scope_assets()
    valid = [a for a in registered if a.repo_id == repo_id]
    if not valid:
        raise ValueError("unregistered Gemma Scope 2 repository; refusing arbitrary discovery")
    if info is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Install huggingface-hub to discover official variants") from exc
        info = HfApi().model_info(repo_id, revision=revision)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("HF asset listing lacks an immutable source SHA")
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        raise ValueError("HF asset listing has no file manifest")
    # Scope 2 publishes several hook families. P1 enumerates SAE families
    # only; transcoders, crosscoders and CLTs need typed P6 loaders.
    sae_families = {"resid_post", "attn_out", "mlp_out"}
    by_key: dict[tuple[int, str, str, str], ScopeAsset] = {}
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str):
            continue
        path_parts = filename.split("/")
        kind = path_parts[0]
        family = kind.removesuffix("_all")
        if family not in sae_families:
            continue
        for idx, part in enumerate(path_parts):
            layer, width, l0 = _parts(part)
            if layer is None or width is None or l0 is None:
                continue
            key = (layer, width, l0, kind)
            if key in by_key:
                continue
            # The entry came from an actual published path; do not generate
            # combinations absent from the manifest, even at new layers.
            identifier = "/".join(path_parts[: idx + 1])
            match = [
                a
                for a in valid
                if a.layer == layer and a.sae_id in {part, identifier}
                and family == "resid_post"
            ]
            exemplar = valid[0]
            # The official attention/MLP extraction location differs from
            # the residual hook, so those sites require an explicit mapping.
            hook = (
                f"blocks.{layer}.hook_resid_post"
                if family == "resid_post"
                else "requires_native_site_mapping"
            )
            by_key[key] = ScopeAsset(
                model_key=exemplar.model_key,
                repo_id=repo_id,
                revision=sha,
                source="official_file_manifest",
                sae_id=identifier,
                layer=layer,
                hook=hook,
                width=width,
                l0_variant=l0,
                loader=exemplar.loader,
                registered_profile=match[0].registered_profile if match else None,
                status="official_manifest_unloaded" if match else "official_manifest_unmapped",
                activation_family=family,
            )
    return sorted(by_key.values(), key=lambda a: (a.layer, a.width or "", a.l0_variant or ""))


def registered_catalog_json(*, model_key: str | None = None) -> list[dict[str, Any]]:
    return [asdict(asset) for asset in registered_scope_assets(model_key=model_key)]
