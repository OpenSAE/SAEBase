"""SAE feature capture, editing, and downstream behavioral evaluation.

Feature editing uses `h_new = h + decode(z_new) - decode(z)` where `z` comes
from `encode(h)`. The untouched activation content, including the residual
`h - decode(z)`, is retained. The decoder is called through its actual API so
biases and its internal scaling are treated consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .model import ActivationSite, ModelAdapter


@dataclass(frozen=True)
class FeatureSnapshot:
    """All tensors are copies referring to one prompt and one hook position."""

    tokens: torch.Tensor
    site: ActivationSite
    activation: torch.Tensor
    codes: torch.Tensor
    reconstruction: torch.Tensor

    @property
    def residual(self) -> torch.Tensor:
        return self.activation - self.reconstruction.to(
            device=self.activation.device, dtype=self.activation.dtype
        )


@dataclass(frozen=True)
class FeaturePatch:
    """A modified latent vector and the corresponding model activation."""

    codes: torch.Tensor
    activation: torch.Tensor


@dataclass(frozen=True)
class InterventionResult:
    snapshot: FeatureSnapshot
    patch: FeaturePatch
    baseline_logits: torch.Tensor
    edited_logits: torch.Tensor

    @property
    def logit_delta(self) -> torch.Tensor:
        return self.edited_logits - self.baseline_logits


def _sae_dimensions(sae: Any) -> tuple[int, int]:
    cfg = getattr(sae, "cfg", None)
    if cfg is None or not hasattr(cfg, "d_in") or not hasattr(cfg, "d_sae"):
        raise ValueError("SAE must expose cfg.d_in and cfg.d_sae")
    d_in, d_sae = int(cfg.d_in), int(cfg.d_sae)
    if d_in < 1 or d_sae < 1:
        raise ValueError("SAE dimensions must be positive")
    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise TypeError("SAE must implement encode() and decode()")
    return d_in, d_sae


def _sae_input(sae: Any, activation: torch.Tensor) -> torch.Tensor:
    try:
        weight = next(sae.parameters())
    except (AttributeError, StopIteration, TypeError):
        return activation
    return activation.to(device=weight.device, dtype=weight.dtype)


def _check_vector(name: str, value: torch.Tensor, *, size: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != size:
        raise ValueError(f"{name} must have shape [{size}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _decode(sae: Any, codes: torch.Tensor, d_in: int) -> torch.Tensor:
    result = sae.decode(codes.unsqueeze(0))
    if not isinstance(result, torch.Tensor) or result.ndim != 2 or tuple(result.shape) != (1, d_in):
        raise ValueError(f"sae.decode must return [1, {d_in}]")
    out = result[0]
    if not bool(torch.isfinite(out).all()):
        raise ValueError("SAE decoder produced non-finite values")
    return out


@torch.no_grad()
def capture_features(
    model: ModelAdapter, sae: Any, *, site: ActivationSite,
    prompt: str | None = None, tokens: torch.Tensor | None = None
) -> FeatureSnapshot:
    """Capture activation and SAE encode/decode at a specific token position."""
    d_in, d_sae = _sae_dimensions(sae)
    ids = model.tokens(prompt=prompt, tokens=tokens)
    activation = model.capture(tokens=ids, site=site)
    _check_vector("activation", activation, size=d_in)
    h_for_sae = _sae_input(sae, activation)
    z = sae.encode(h_for_sae.unsqueeze(0))
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or tuple(z.shape) != (1, d_sae):
        raise ValueError(f"sae.encode must return [1, {d_sae}]")
    z = z[0]
    _check_vector("codes", z, size=d_sae)
    reconstruction = _decode(sae, z, d_in).to(
        device=activation.device, dtype=activation.dtype
    )
    return FeatureSnapshot(
        tokens=ids.detach().clone(), site=site,
        activation=activation, codes=z.detach().clone(),
        reconstruction=reconstruction.detach().clone(),
    )


@torch.no_grad()
def edit_features(
    sae: Any, snapshot: FeatureSnapshot, *, replacements: Mapping[int, float | torch.Tensor]
) -> FeaturePatch:
    """Replace selected code values, preserving all other SAE and residual content.

    An empty mapping reproduces the exact original activation. New values are
    literal SAE code values, not multipliers of the original activation.
    """
    d_in, d_sae = _sae_dimensions(sae)
    _check_vector("captured activation", snapshot.activation, size=d_in)
    _check_vector("captured codes", snapshot.codes, size=d_sae)
    if not isinstance(replacements, Mapping):
        raise TypeError("replacements must map feature indices to new code values")
    z_new = snapshot.codes.detach().clone()
    for key, value in replacements.items():
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < d_sae:
            raise IndexError(f"feature index {key!r} outside [0, {d_sae})")
        val = torch.as_tensor(value, device=z_new.device, dtype=z_new.dtype)
        if val.ndim != 0 or not bool(torch.isfinite(val)):
            raise ValueError(f"replacement value for feature {key} must be a finite scalar")
        z_new[key] = val
    before = _decode(sae, snapshot.codes, d_in)
    after = _decode(sae, z_new, d_in)
    delta = (after - before).to(
        device=snapshot.activation.device, dtype=snapshot.activation.dtype
    )
    edited_activation = snapshot.activation + delta
    _check_vector("edited activation", edited_activation, size=d_in)
    return FeaturePatch(codes=z_new.detach().clone(), activation=edited_activation.detach().clone())


def coalition_replacements(
    codes: torch.Tensor, *, selected_indices: Sequence[int], retained_indices: Sequence[int]
) -> dict[int, float]:
    """Remove only *selected* features absent from the retained subset.

    Retained indices are actual SAE feature IDs (rather than offsets into
    the selected list). Features outside selected_indices stay unchanged.
    """
    if not isinstance(codes, torch.Tensor) or codes.ndim != 1:
        raise ValueError("codes must be a 1-D tensor")
    def validate(indices: Sequence[int], name: str) -> set[int]:
        if not isinstance(indices, (list, tuple, set, frozenset)):
            raise TypeError(f"{name} must be a sequence of integer feature IDs")
        ids = list(indices)
        if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < len(codes) for k in ids):
            raise IndexError(f"{name} contains invalid feature ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} contains duplicate features")
        return set(ids)
    selected = validate(selected_indices, "selected_indices")
    retained = validate(retained_indices, "retained_indices")
    if not retained.issubset(selected):
        raise ValueError("retained_indices must be a subset of selected_indices")
    return {idx: 0.0 for idx in selected.difference(retained)}


@torch.no_grad()
def evaluate_edit(
    model: ModelAdapter, sae: Any, *, site: ActivationSite,
    replacements: Mapping[int, float | torch.Tensor],
    prompt: str | None = None, tokens: torch.Tensor | None = None,
    readout_position: int = -1, verify_identity: bool = True,
) -> InterventionResult:
    """Capture, edit, replay, and report baseline/edited logit vectors.

    Identity replay is checked by default to identify a mismatch between the
    capture site and intervention site before reporting effects.
    """
    snapshot = capture_features(model, sae, site=site, prompt=prompt, tokens=tokens)
    if verify_identity:
        model.assert_identity_patch(tokens=snapshot.tokens, site=site)
    patch = edit_features(sae, snapshot, replacements=replacements)
    baseline = model.forward_logits(tokens=snapshot.tokens, readout_position=readout_position)
    edited = model.patch_logits(
        tokens=snapshot.tokens, site=site, activation=patch.activation,
        readout_position=readout_position,
    )
    if baseline.shape != edited.shape or not bool(torch.isfinite(edited).all()):
        raise RuntimeError("edited logits must be finite and match baseline shape")
    return InterventionResult(
        snapshot=snapshot, patch=patch,
        baseline_logits=baseline, edited_logits=edited,
    )
