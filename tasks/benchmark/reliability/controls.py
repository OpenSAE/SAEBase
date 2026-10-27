"""Analytic negative controls and only mathematically valid invariances."""

from __future__ import annotations
import warnings
import numpy as np

from .ground_truth import make_factors
from .rss_truth import fixture


def check_coordinates(*, seed: int = 10) -> dict:
    """Permute/reparameterize paired SAE codes/dictionary with exact reconstruction."""
    data = make_factors("superposition", seed=seed)
    z = data.test
    D = data.dictionary
    rng = np.random.default_rng(seed + 999)
    permutation = rng.permutation(z.shape[1])
    factors = rng.uniform(0.4, 1.8, z.shape[1])
    baseline = z @ D.T
    permuted = z[:, permutation] @ D[:, permutation].T
    scaled = (z * factors) @ (D / factors).T
    err_p = float(np.max(np.abs(permuted - baseline)))
    err_s = float(np.max(np.abs(scaled - baseline)))
    # Null patch is exactly the same activation, independent of dictionaries.
    null = float(np.max(np.abs(baseline.copy() - baseline)))
    passed = max(err_p, err_s, null) < 1e-10
    return {
        "status": "passed" if passed else "failed",
        "feature_permutation_max_error": err_p,
        "positive_coordinate_rescaling_max_error": err_s,
        "null_activation_intervention_max_error": null,
        "note": "This does not claim RSS MeanRSS is invariant under arbitrary feature rescaling",
    }


def check_sar_negatives(paper, *, seed: int = 20) -> dict:
    masks, fit, held = paper.coalitions(K=4, M=256, n_fit=190, seed=seed)
    constant = np.full(len(masks), 5.0, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        null = paper.sar(constant, masks, fit, held, g_full=5.0)
    # Undefined rho2 must not be treated as either positive or negative.
    null_pass = abs(null["sar1"]) < 1e-9 and abs(null["sar2"]) < 1e-9 and np.isnan(null["rho2"])
    additive = 1.2 + np.asarray(masks, dtype=float) @ np.asarray([1.0, -0.7, 0.3, 2.0])
    ordinary = paper.sar(additive, masks, fit, held, g_full=3.8)
    affine_pass = ordinary["sar1"] < 1e-8 and ordinary["sar2"] < 1e-8
    return {
        "status": "passed" if null_pass and affine_pass else "failed",
        "constant_behavior_sar1": float(null["sar1"]),
        "constant_behavior_rho2_status": "undefined" if np.isnan(null["rho2"]) else "invalid",
        "affine_behavior_sar1": float(ordinary["sar1"]),
        "affine_behavior_sar2": float(ordinary["sar2"]),
    }


def check_rss_controls(paper, *, seed: int = 21) -> dict:
    f = fixture(seed=seed)
    selected = f["selection"]
    kwargs = {"rho": 0.8, "kappa": 0.9, "tau": 0.1, "M": 12, "K": 3}
    support, direction, mean = paper.rss(selected, **kwargs)
    rng = np.random.default_rng(seed + 19)
    perm = rng.permutation(selected.shape[1])
    changed, p_direction, p_mean = paper.rss(selected[:, perm], **kwargs)
    inverse = np.argsort(perm)
    support_matches = np.array_equal(changed, inverse[support])
    directions_match = np.array_equal(p_direction, direction)
    mean_same = np.isclose(p_mean, mean, atol=1e-12, rtol=1e-12)
    # A *common positive* scale preserves support/directions when tau scales,
    # but MeanRSS scales too. Arbitrary per-feature scaling is NOT invariant.
    scale = 2.0
    scaled, scaled_direction, scaled_mean = paper.rss(
        selected * scale, rho=0.8, kappa=0.9, tau=0.1 * scale, M=12, K=3
    )
    valid_scale = (
        np.array_equal(scaled, support)
        and np.array_equal(scaled_direction, direction)
        and np.isclose(scaled_mean, scale * mean)
    )
    empty, _, empty_mean = paper.rss(np.zeros_like(selected), **kwargs)
    undefined = len(empty) == 0 and np.isnan(empty_mean)
    try:
        paper.rss(np.full_like(selected, float("nan")), **kwargs)
        invalid_input = False
    except ValueError:
        invalid_input = True
    passed = (
        support_matches
        and directions_match
        and mean_same
        and valid_scale
        and undefined
        and invalid_input
    )
    return {
        "status": "passed" if passed else "failed",
        "feature_permutation_support_equivalent": bool(support_matches),
        "feature_permutation_direction_equivalent": bool(directions_match),
        "permuted_mean_rss_unchanged": bool(mean_same),
        "global_positive_scaling_behavior_valid": bool(valid_scale),
        "empty_support_status": "undefined" if undefined else "failed",
        "nonfinite_input_rejected": invalid_input,
    }
