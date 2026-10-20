"""Evaluate SAR coalitions using SAEBase feature capture and model patching.

Sampling and SAR metrics are imported from the *unchanged* paper's sar.py.
One coalition means removing absent selected codes and keeping all other
activation content fixed, including the original reconstruction residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features, edit_features


@dataclass(frozen=True)
class SARResult:
    """Paper metrics and sufficient raw arrays for subsequent checking."""

    metrics: dict[str, float]
    feature_ids: np.ndarray
    masks: np.ndarray
    fit_indices: np.ndarray
    held_indices: np.ndarray
    responses: np.ndarray
    full_response: float
    site: ActivationSite
    source_tokens: np.ndarray
    identity: SAEIdentity | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "metrics": {k: (v if np.isfinite(v) else None) for k, v in self.metrics.items()},
            "feature_ids": self.feature_ids.tolist(),
            "site": {"hook_name": self.site.hook_name, "layer": self.site.layer,
                     "position": self.site.position},
            "n_coalitions": int(len(self.masks)),
            "n_fit": int(len(self.fit_indices)),
            "n_held": int(len(self.held_indices)),
            "full_response": self.full_response,
            "identity": self.identity.to_dict() if self.identity else None,
        }


def _scalar(value: Any, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must return a scalar")
        value = value.item()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must return a finite scalar")
    return float(array)


def _check_affine_decode(sae: Any, snapshot: Any, indices: list[int]) -> None:
    """Paper's coalition interpretation requires additive decoded contributions."""
    if len(indices) < 2:
        return
    h = snapshot.activation
    individual = [
        edit_features(sae, snapshot, replacements={i: 0.0}).activation - h
        for i in indices
    ]
    for selected in ((indices[0], indices[1]), tuple(indices)):
        actual = edit_features(
            sae, snapshot, replacements={i: 0.0 for i in selected}
        ).activation
        additive_delta = sum((individual[indices.index(i)] for i in selected),
                             torch.zeros_like(h))
        if not torch.allclose(actual.float(), (h + additive_delta).float(), rtol=1e-4, atol=1e-4):
            raise ValueError(
                "SAE decoder is not additive for the selected features. "
                "SAR coalitions require a linear/affine decoder in model activation coordinates."
            )


@torch.no_grad()
def evaluate_sar(
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    paper: Any,
    readout: Callable[[torch.Tensor], float],
    prompt: str | None = None,
    tokens: torch.Tensor | None = None,
    K: int = 6,
    M: int = 128,
    n_fit: int = 96,
    seed: int = 0,
    p: float = 0.5,
    readout_position: int = -1,
    verify_replay: bool = True,
    identity: SAEIdentity | None = None,
    require_affine_decode: bool = True,
) -> SARResult:
    """Run real model patches and call the original SAR estimator.

    ``readout`` maps a single ``[vocab]`` logit tensor to one scalar. For
    example: ``lambda logits: logits[target_id] - logits[foil_id]``.
    Paper functions ``paper.coalitions`` and ``paper.sar`` are supplied
    directly by the caller. The estimator and sampling math are not copied.

    Sequentially patches M coalitions; useful for demos / correctness checks,
    not intended as an optimized full-scale experiment runner.
    """
    for fname in ("coalitions", "sar"):
        if not callable(getattr(paper, fname, None)):
            raise TypeError(f"paper must expose callable {fname}() from sar.py")
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer >= 2")
    model.capabilities.require("forward")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    if identity is not None and (identity.hook_name != site.hook_name or identity.layer != site.layer):
        raise ValueError("SAE identity site differs from requested activation site")
    ids = model.tokens(prompt=prompt, tokens=tokens)
    snapshot = capture_features(model, sae, tokens=ids, site=site)
    if identity is not None and (identity.d_in != snapshot.activation.numel()
                                 or identity.d_sae != snapshot.codes.numel()):
        raise ValueError("SAE identity dimensions differ from captured features")
    if K > snapshot.codes.numel():
        raise ValueError(f"K={K} exceeds SAE feature count {snapshot.codes.numel()}")
    selected = torch.argsort(snapshot.codes.abs(), descending=True, stable=True)[:K].tolist()
    if require_affine_decode:
        _check_affine_decode(sae, snapshot, selected)

    baseline_logits = model.forward_logits(tokens=ids, readout_position=readout_position)
    clean = _scalar(readout(baseline_logits), name="readout")
    if verify_replay:
        replay_logits = model.patch_logits(
            tokens=ids, site=site, activation=snapshot.activation,
            readout_position=readout_position,
        )
        if not torch.allclose(baseline_logits.float(), replay_logits.float(), rtol=1e-4, atol=1e-4):
            raise RuntimeError("Identity patch changed the behavior; verify hook, model and eval mode")

    masks, fit, held = paper.coalitions(K=K, M=M, p=p, n_fit=n_fit, seed=seed)
    masks = np.asarray(masks, dtype=bool)
    fit, held = np.asarray(fit), np.asarray(held)
    if masks.shape != (M, K):
        raise ValueError(f"Paper coalition sampler returned {masks.shape}; expected {(M, K)}")
    responses = np.empty(M, dtype=np.float64)
    for index, mask in enumerate(masks):
        replacements = {fid: 0.0 for fid, keep in zip(selected, mask) if not keep}
        if replacements:
            patch = edit_features(sae, snapshot, replacements=replacements)
            logits = model.patch_logits(
                tokens=ids, site=site, activation=patch.activation,
                readout_position=readout_position,
            )
        else:
            logits = baseline_logits
        responses[index] = _scalar(readout(logits), name="readout")
    metrics = paper.sar(responses, masks, fit, held, g_full=clean)
    if not isinstance(metrics, dict):
        raise TypeError("sar.py estimator must return a dictionary")
    clean_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 0 or np.isinf(arr):
            raise ValueError(f"metric {key} must be a finite scalar or NaN when undefined")
        clean_metrics[str(key)] = float(arr)
    return SARResult(
        metrics=clean_metrics,
        feature_ids=np.asarray(selected, dtype=int), masks=masks.copy(),
        fit_indices=fit.copy(), held_indices=held.copy(), responses=responses,
        full_response=clean, site=site, source_tokens=ids.detach().cpu().numpy().copy(),
        identity=identity,
    )
