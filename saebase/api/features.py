"""SAE feature capture, editing, and downstream behavioral evaluation.

Feature editing uses `h_new = h + decode(z_new) - decode(z)` where `z` comes
from `encode(h)`. The untouched activation content, including the residual
`h - decode(z)`, is retained. The decoder is called through its actual API so
biases and its internal scaling are treated consistently.

SAEs with ``normalize_activations="layer_norm"`` (e.g. Mistral TopK under
SAELens) write per-input ``ln_mu`` / ``ln_std`` during encode and delete them
on decode. This module copies those statistics onto ``FeatureSnapshot`` and
restores them before every independent decode so edited reconstructions use
the same affine inverse as the original encode, without leaking state across
prompts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

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
    ln_mu: torch.Tensor | None = None
    ln_std: torch.Tensor | None = None

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


def _clone_ln_stat(value: Any) -> torch.Tensor | None:
    if value is None or not isinstance(value, torch.Tensor):
        return None
    return value.detach().clone()


def _capture_ln_stats(sae: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return _clone_ln_stat(getattr(sae, "ln_mu", None)), _clone_ln_stat(getattr(sae, "ln_std", None))


def _clear_ln_stats(sae: Any) -> None:
    for name in ("ln_mu", "ln_std"):
        if hasattr(sae, name):
            delattr(sae, name)


@contextmanager
def _restore_ln_stats(
    sae: Any,
    ln_mu: torch.Tensor | None,
    ln_std: torch.Tensor | None,
) -> Iterator[None]:
    """Install snapshot LN stats for decode, then always clear SAE buffers."""
    _clear_ln_stats(sae)
    if ln_mu is not None and ln_std is not None:
        sae.ln_mu = ln_mu
        sae.ln_std = ln_std
    try:
        yield
    finally:
        _clear_ln_stats(sae)


def _layer_norm_stats(
    activation: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SAELens ``normalize_activations='layer_norm'`` affine stats."""
    mu = activation.mean(dim=-1, keepdim=True)
    centered = activation - mu
    std = centered.std(dim=-1, keepdim=True)
    return mu, std


def _sae_uses_layer_norm(sae: Any) -> bool:
    cfg = getattr(sae, "cfg", None)
    return getattr(cfg, "normalize_activations", None) == "layer_norm"


def _decode(
    sae: Any,
    codes: torch.Tensor,
    d_in: int,
    *,
    ln_mu: torch.Tensor | None = None,
    ln_std: torch.Tensor | None = None,
    activation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode SAE codes, restoring per-input layer_norm stats when needed.

    ``codes`` may be a single vector ``[d_sae]`` or a batch ``[n, d_sae]``.
    When the SAE uses ``layer_norm`` and explicit ``ln_*`` are absent, stats are
    recomputed from ``activation`` (same formula as SAELens encode).
    """
    single = codes.ndim == 1
    z = codes.unsqueeze(0) if single else codes
    mu, std = ln_mu, ln_std
    if mu is None and std is None and activation is not None and _sae_uses_layer_norm(sae):
        h = activation.unsqueeze(0) if activation.ndim == 1 else activation
        h = _sae_input(sae, h)
        mu, std = _layer_norm_stats(h)
    with _restore_ln_stats(sae, mu, std):
        result = sae.decode(z)
    if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.shape[-1] != d_in:
        raise ValueError(f"sae.decode must return [batch, {d_in}]")
    if single and result.shape[0] != 1:
        raise ValueError(f"sae.decode must return [1, {d_in}] for a single code vector")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("SAE decoder produced non-finite values")
    return result[0] if single else result


@torch.no_grad()
def snapshot_from_activation(
    sae: Any, *, tokens: torch.Tensor, site: ActivationSite, activation: torch.Tensor
) -> FeatureSnapshot:
    """Encode and decode one already-captured activation. Does not run the model."""
    d_in, d_sae = _sae_dimensions(sae)
    _check_vector("activation", activation, size=d_in)
    h_for_sae = _sae_input(sae, activation)
    z = sae.encode(h_for_sae.unsqueeze(0))
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or tuple(z.shape) != (1, d_sae):
        raise ValueError(f"sae.encode must return [1, {d_sae}]")
    z = z[0]
    _check_vector("codes", z, size=d_sae)
    ln_mu, ln_std = _capture_ln_stats(sae)
    reconstruction = _decode(sae, z, d_in, ln_mu=ln_mu, ln_std=ln_std).to(
        device=activation.device, dtype=activation.dtype
    )
    if hasattr(sae, "ln_mu") or hasattr(sae, "ln_std"):
        raise RuntimeError("SAE layer_norm buffers leaked after capture decode")
    return FeatureSnapshot(
        tokens=tokens.detach().clone(),
        site=site,
        activation=activation.detach().clone(),
        codes=z.detach().clone(),
        reconstruction=reconstruction.detach().clone(),
        ln_mu=ln_mu,
        ln_std=ln_std,
    )


@torch.no_grad()
def capture_features(
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    prompt: str | None = None,
    tokens: torch.Tensor | None = None,
) -> FeatureSnapshot:
    """Capture activation and SAE encode/decode at a specific token position."""
    ids = model.tokens(prompt=prompt, tokens=tokens)
    activation = model.capture(tokens=ids, site=site)
    return snapshot_from_activation(sae, tokens=ids, site=site, activation=activation)


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
    if not replacements:
        return FeaturePatch(
            codes=z_new.detach().clone(), activation=snapshot.activation.detach().clone()
        )
    # Decode both code vectors. Qwen bf16 stores reconstruction in the
    # activation dtype, and subtracting that rounded tensor misses the
    # frozen logit bar, so the stored reconstruction is not reused here.
    before = _decode(
        sae,
        snapshot.codes,
        d_in,
        ln_mu=snapshot.ln_mu,
        ln_std=snapshot.ln_std,
    )
    after = _decode(
        sae,
        z_new,
        d_in,
        ln_mu=snapshot.ln_mu,
        ln_std=snapshot.ln_std,
    )
    if hasattr(sae, "ln_mu") or hasattr(sae, "ln_std"):
        raise RuntimeError("SAE layer_norm buffers leaked after edit decode")
    delta = (after - before).to(device=snapshot.activation.device, dtype=snapshot.activation.dtype)
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
        if any(
            isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < len(codes) for k in ids
        ):
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
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    replacements: Mapping[int, float | torch.Tensor],
    prompt: str | None = None,
    tokens: torch.Tensor | None = None,
    readout_position: int = -1,
    verify_identity: bool = True,
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
        tokens=snapshot.tokens,
        site=site,
        activation=patch.activation,
        readout_position=readout_position,
    )
    if baseline.shape != edited.shape or not bool(torch.isfinite(edited).all()):
        raise RuntimeError("edited logits must be finite and match baseline shape")
    return InterventionResult(
        snapshot=snapshot,
        patch=patch,
        baseline_logits=baseline,
        edited_logits=edited,
    )
