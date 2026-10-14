"""Stable, serializable provenance of an SAE evaluation site."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model import ActivationSite


@dataclass(frozen=True)
class SAEIdentity:
    """Selected model/SAE identities; unknown revision and scaling remain explicit.

    A named profile alone is not evidence that a checkpoint was downloaded or
    tested. `revision=None` means the exact commit was not resolved.
    """

    model_key: str
    sae_key: str
    checkpoint: str
    sae_id: str
    hook_name: str
    layer: int
    loader: str
    d_in: int
    d_sae: int
    revision: str | None = None
    model_revision: str | None = None
    loaded_model_name: str | None = None
    trained_on_model_key: str | None = None
    target_model_key: str | None = None
    transfer_kind: str = "none"
    normalization: str = "unspecified"
    dtype: str | None = None

    @property
    def site(self) -> ActivationSite:
        return ActivationSite(layer=self.layer, hook_name=self.hook_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_loaded(
        cls, *, sae: Any, metadata: dict[str, Any], model_key: str | None = None,
        revision: str | None = None, model_revision: str | None = None,
    ) -> "SAEIdentity":
        cfg = getattr(sae, "cfg", None)
        if cfg is None or not hasattr(cfg, "d_in") or not hasattr(cfg, "d_sae"):
            raise ValueError("SAE metadata requires cfg.d_in and cfg.d_sae")
        try:
            dtype = str(next(sae.parameters()).dtype)
        except (AttributeError, StopIteration, TypeError):
            dtype = None
        for size_key, actual in (("d_in", int(cfg.d_in)), ("d_sae", int(cfg.d_sae))):
            claimed = metadata.get(size_key)
            if claimed is not None and int(claimed) != actual:
                raise ValueError(
                    f"SAE metadata {size_key}={claimed} differs from loaded SAE {actual}"
                )
        def required(key: str) -> Any:
            value = metadata.get(key)
            if value is None or value == "":
                raise ValueError(f"missing required SAE metadata field {key!r}")
            return value
        mk = model_key or metadata.get("model_key")
        if not mk:
            raise ValueError("model_key is required to identify an SAE on a host model")
        layer = int(required("layer"))
        hook = str(required("hook_name"))
        # check layer-hook consistency before constructing provenance.
        ActivationSite(layer=layer, hook_name=hook)
        trained_on = metadata.get("trained_on_model_key") or metadata.get("model_key")
        target = metadata.get("target_model_key") or metadata.get("model_key")
        allowed = {metadata.get("model_key"), trained_on, target}
        if mk not in allowed:
            raise ValueError(f"host model {mk!r} not allowed by SAE profile {sorted(str(x) for x in allowed)}")
        return cls(
            model_key=str(mk), sae_key=str(required("sae_key")),
            checkpoint=str(required("sae_release")),
            sae_id=str(required("sae_id")), hook_name=hook, layer=layer,
            loader=str(required("loader")),
            d_in=int(cfg.d_in), d_sae=int(cfg.d_sae),
            revision=revision or metadata.get("revision"),
            model_revision=model_revision or metadata.get("model_revision"),
            loaded_model_name=metadata.get("loaded_model_name"),
            trained_on_model_key=str(trained_on) if trained_on else None,
            target_model_key=str(target) if target else None,
            transfer_kind=str(metadata.get("transfer_kind", "none")),
            normalization=str(metadata.get("normalization") or "unspecified"),
            dtype=dtype,
        )

    @classmethod
    def for_custom(
        cls, *, sae: Any, model_key: str, checkpoint: str,
        hook_name: str, layer: int, sae_key: str = "custom",
        sae_id: str = "custom", revision: str | None = None,
        normalization: str = "unspecified",
    ) -> "SAEIdentity":
        return cls.from_loaded(
            sae=sae, metadata={
                "model_key":model_key,"sae_key":sae_key,"sae_release":checkpoint,
                "sae_id":sae_id,"hook_name":hook_name,"layer":layer,
                "loader":"custom","normalization":normalization,
            }, revision=revision,
        )
