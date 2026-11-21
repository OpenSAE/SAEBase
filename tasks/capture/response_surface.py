"""Evaluate SAR coalitions using SAEBase feature capture and model patching.

Sampling and SAR metrics are imported from the *unchanged* paper's sar.py.
One coalition means removing absent selected codes and keeping all other
activation content fixed, including the original reconstruction residual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features, edit_features
from saebase.api.features import FeatureSnapshot


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
    pairwise: dict | None = None
    pair_coefficients: np.ndarray | None = None
    pair_feature_ids: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return {
            **({"pairwise": self.pairwise} if self.pairwise is not None else {}),
            "metrics": {k: (v if np.isfinite(v) else None) for k, v in self.metrics.items()},
            "feature_ids": self.feature_ids.tolist(),
            "site": {
                "hook_name": self.site.hook_name,
                "layer": self.site.layer,
                "position": self.site.position,
            },
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
    # Diagnose decoder additivity before native model-patch rounding. A BF16
    # h + delta rounds each individual edit differently from a simultaneous edit,
    # even for an exactly affine FP32 decoder. Actual interventions below retain
    # the original snapshot and the model's native precision.
    if snapshot.activation.dtype in (torch.float16, torch.bfloat16):
        snapshot = replace(snapshot, activation=snapshot.activation.float())
    h = snapshot.activation
    individual = [
        edit_features(sae, snapshot, replacements={i: 0.0}).activation - h for i in indices
    ]
    for selected in ((indices[0], indices[1]), tuple(indices)):
        actual = edit_features(sae, snapshot, replacements={i: 0.0 for i in selected}).activation
        additive_delta = sum((individual[indices.index(i)] for i in selected), torch.zeros_like(h))
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
    snapshot: FeatureSnapshot | None = None,
    activation_readout: Callable[[torch.Tensor | None], float] | None = None,
    feature_ids: list[int] | None = None,
    top_pairs: int = 0,
    pair_ranking: str = "absolute",
    coalition_readout: Callable[[list[int], np.ndarray, float], np.ndarray] | None = None,
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
    if pair_ranking not in {"absolute", "signed"}:
        raise ValueError("pair_ranking must be absolute or signed")
    model.capabilities.require("forward")
    model.capabilities.require("capture")
    model.capabilities.require("patch")
    if identity is not None and (
        identity.hook_name != site.hook_name or identity.layer != site.layer
    ):
        raise ValueError("SAE identity site differs from requested activation site")
    ids = model.tokens(prompt=prompt, tokens=tokens)
    if snapshot is None:
        snapshot = capture_features(model, sae, tokens=ids, site=site)
    elif snapshot.site != site or not torch.equal(snapshot.tokens.to(ids.device), ids):
        raise ValueError("SAR snapshot differs from requested tokens or activation site")
    if identity is not None and (
        identity.d_in != snapshot.activation.numel() or identity.d_sae != snapshot.codes.numel()
    ):
        raise ValueError("SAE identity dimensions differ from captured features")
    if K > snapshot.codes.numel():
        raise ValueError(f"K={K} exceeds SAE feature count {snapshot.codes.numel()}")
    selected = (
        torch.argsort(snapshot.codes.abs(), descending=True, stable=True)[:K].tolist()
        if feature_ids is None
        else list(feature_ids)
    )
    if (
        len(selected) != K
        or len(set(selected)) != K
        or any(type(i) is not int or not 0 <= i < snapshot.codes.numel() for i in selected)
    ):
        raise ValueError("Selected SAR feature IDs must be K distinct valid SAE coordinates")
    if require_affine_decode:
        _check_affine_decode(sae, snapshot, selected)

    if activation_readout is not None:
        clean = _scalar(activation_readout(None), name="activation_readout")
        if verify_replay:
            replay = _scalar(activation_readout(snapshot.activation), name="activation_readout")
            if not np.isclose(clean, replay, rtol=1e-4, atol=1e-4):
                raise RuntimeError("Identity patch changed the teacher-forced answer response")
    else:
        baseline_logits = model.forward_logits(tokens=ids, readout_position=readout_position)
        clean = _scalar(readout(baseline_logits), name="readout")
        if verify_replay:
            replay_logits = model.patch_logits(
                tokens=ids,
                site=site,
                activation=snapshot.activation,
                readout_position=readout_position,
            )
            if not torch.allclose(
                baseline_logits.float(), replay_logits.float(), rtol=1e-4, atol=1e-4
            ):
                raise RuntimeError(
                    "Identity patch changed the behavior; verify hook, model and eval mode"
                )
    masks, fit, held = paper.coalitions(K=K, M=M, p=p, n_fit=n_fit, seed=seed)
    masks = np.asarray(masks, dtype=bool)
    fit, held = np.asarray(fit), np.asarray(held)
    if masks.shape != (M, K):
        raise ValueError(f"Paper coalition sampler returned {masks.shape}; expected {(M, K)}")
    if coalition_readout is not None:
        responses = np.asarray(coalition_readout(selected, masks, clean), dtype=np.float64)
        if responses.shape != (M,) or not np.isfinite(responses).all():
            raise ValueError("Coalition readout must return one finite score per sampled mask")
    else:
        responses = np.empty(M, dtype=np.float64)
        for index, mask in enumerate(masks):
            replacements = {fid: 0.0 for fid, keep in zip(selected, mask) if not keep}
            if activation_readout is not None:
                responses[index] = (
                    _scalar(
                        activation_readout(
                            edit_features(sae, snapshot, replacements=replacements).activation
                        ),
                        name="activation_readout",
                    )
                    if replacements
                    else clean
                )
            else:
                if replacements:
                    patch = edit_features(sae, snapshot, replacements=replacements)
                    logits = model.patch_logits(
                        tokens=ids,
                        site=site,
                        activation=patch.activation,
                        readout_position=readout_position,
                    )
                else:
                    logits = baseline_logits
                responses[index] = _scalar(readout(logits), name="readout")

    clean_metrics, pairwise, pair_coefficients, pair_feature_ids = fit_responses(
        paper,
        selected,
        masks,
        responses,
        fit,
        held,
        clean,
        top_pairs=top_pairs,
        pair_ranking=pair_ranking,
    )
    return SARResult(
        metrics=clean_metrics,
        feature_ids=np.asarray(selected, dtype=int),
        masks=masks.copy(),
        fit_indices=fit.copy(),
        held_indices=held.copy(),
        responses=responses,
        full_response=clean,
        site=site,
        source_tokens=ids.detach().cpu().numpy().copy(),
        identity=identity,
        pairwise=pairwise,
        pair_coefficients=pair_coefficients,
        pair_feature_ids=pair_feature_ids,
    )


def fit_responses(
    paper,
    feature_ids,
    masks,
    responses,
    fit,
    held,
    full_response,
    *,
    top_pairs=0,
    pair_ranking="signed",
):
    """Fit the original SAR estimator to measured scores, including exhaustive-score subsets.

    The caller owns measurement provenance; sampling remains paper.coalitions().
    Both live measurement and reuse of an identical measured intervention cube
    use this single fitting and pair-ranking implementation.
    """
    selected = list(feature_ids)
    K = len(selected)
    masks = np.asarray(masks, dtype=bool)
    responses = np.asarray(responses, dtype=np.float64)
    fit, held = np.asarray(fit), np.asarray(held)
    clean = float(full_response)
    if masks.shape != (len(responses), K):
        raise ValueError("SAR measured masks must match response and feature dimensions")
    if pair_ranking not in {"absolute", "signed"}:
        raise ValueError("pair_ranking must be absolute or signed")
    metrics = paper.sar(responses, masks, fit, held, g_full=clean)
    if not isinstance(metrics, dict):
        raise TypeError("sar.py estimator must return a dictionary")
    clean_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 0 or np.isinf(arr):
            raise ValueError(f"metric {key} must be a finite scalar or NaN when undefined")
        clean_metrics[str(key)] = float(arr)
    pairwise = pair_coefficients = pair_feature_ids = None
    if top_pairs:
        from itertools import combinations

        # Use the original estimator's design and the same fitting rows. These
        # coefficients describe the surrogate; they are not separate SAR scores.
        design = paper.design(masks, order=2)
        coefficients, _, rank, _ = np.linalg.lstsq(design[fit], responses[fit], rcond=None)
        pair_coefficients = coefficients[1 + K :]
        pair_feature_ids = np.asarray(list(combinations(selected, 2)), dtype=int)
        expected = K * (K - 1) // 2
        if design.shape[1] != 1 + K + expected or len(pair_coefficients) != expected:
            raise ValueError("Original SAR design differs from intercept/single/pair ordering")
        ranked = sorted(
            range(expected),
            key=lambda i: (
                -abs(pair_coefficients[i]) if pair_ranking == "absolute" else -pair_coefficients[i],
                tuple(sorted(pair_feature_ids[i])),
            ),
        )
        pairwise = {
            "status": "undefined" if clean_metrics.get("var") == 0 else "completed",
            "ranking": "absolute_pairwise_coefficient"
            if pair_ranking == "absolute"
            else "signed_pairwise_coefficient_descending",
            "basis": "binary feature-retention indicators; intercept, single effects and pairs",
            "fit_rank": int(rank),
            "fit_parameters": int(design.shape[1]),
            "pair_count": expected,
            "top_pairs": [
                {
                    "feature_ids": pair_feature_ids[i].tolist(),
                    "coefficient": float(pair_coefficients[i]),
                }
                for i in ranked[:top_pairs]
            ],
        }
        if clean_metrics.get("var") == 0:
            pairwise.update(
                top_pairs=[], reason="All combinations score the same behavior, not a meaningful interactive ranking.."
            )
    return clean_metrics, pairwise, pair_coefficients, pair_feature_ids
