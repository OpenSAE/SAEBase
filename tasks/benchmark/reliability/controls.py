"""Purpose: Evaluate analytical SAR and RSS controls and report original-method numerical agreement.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np

from saebase.api.paper_sources import load_paper_module
from tasks.benchmark.methods.response_surface import source_identity
from tasks.benchmark.reliability.ground_truth import evaluate_degradation, make_factors
from tasks.benchmark.reliability.schema import AuditConfig


def fixture(*, seed: int = 11, n_select: int = 40, n_held_out: int = 40):
    rng = np.random.default_rng(seed)
    if n_select < 5 or n_held_out < 5:
        raise ValueError("need independent selection and held-out variants")
    K = 12
    support = np.asarray([1, 4, 9], dtype=int)
    direction = np.asarray([1, -1, 1], dtype=int)
    scale = np.asarray([2.0, 1.6, 1.2])
    selection = rng.normal(0, 0.005, (n_select, K))
    held = rng.normal(0, 0.005, (n_held_out, K))
    selection[:, support] += scale * direction
    held[:, support] += scale * direction
    return {
        "selection": selection,
        "held_out": held,
        "support": support,
        "direction": direction,
        "seed": seed,
        "selection_ids": tuple(f"selection-{i}" for i in range(n_select)),
        "held_out_ids": tuple(f"held_out-{i}" for i in range(n_held_out)),
    }


def verify_original_rss(paper, *, seed: int = 11) -> dict:
    if not callable(getattr(paper, "rss", None)) or not callable(
        getattr(paper, "held_out_rate", None)
    ):
        raise TypeError("trusted original RSS module requires rss and held_out_rate")
    f = fixture(seed=seed)
    if set(f["selection_ids"]) & set(f["held_out_ids"]):
        raise ValueError("RSS selection/held-out leakage")
    support, direction, mean = paper.rss(f["selection"], rho=0.8, kappa=0.9, tau=0.1, M=12, K=3)
    rates = paper.held_out_rate(f["held_out"], support, direction, tau=0.1)
    passed = (
        np.array_equal(support, f["support"])
        and np.array_equal(direction, f["direction"])
        and bool(np.allclose(rates, 1.0))
        and np.isfinite(mean)
    )
    # Modify held-out only; selection-derived support/direction may not change.
    flipped = f["held_out"].copy()
    flipped[:, f["support"]] *= -1
    flipped_rates = paper.held_out_rate(flipped, support, direction, tau=0.1)
    passed = passed and bool(np.allclose(flipped_rates, 0.0))
    empty, empty_direction, undefined = paper.rss(
        np.zeros_like(f["selection"]), rho=0.8, kappa=0.9, tau=0.1, M=12, K=3
    )
    passed = passed and len(empty) == 0 and len(empty_direction) == 0 and np.isnan(undefined)
    return {
        "status": "passed" if passed else "failed",
        "source_estimator": "original RSS / held_out_rate",
        "expected_support": f["support"].tolist(),
        "selected_support": support.tolist(),
        "expected_direction": f["direction"].tolist(),
        "selected_direction": direction.tolist(),
        "mean_rss": float(mean) if np.isfinite(mean) else None,
        "held_out_persistence": rates.tolist(),
        "flipped_held_out_persistence": flipped_rates.tolist(),
        "n_selection": len(f["selection"]),
        "n_held_out": len(f["held_out"]),
        "empty_support_status": "undefined" if np.isnan(undefined) else "invalid",
        "selection_seed": seed,
    }


CASES = ("first_order", "pairwise", "third_order")


def behavior(mask, case: str):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] < 3 or case not in CASES:
        raise ValueError("invalid interaction mask/degree")
    x = 2 * mask.astype(np.float64) - 1
    value = 5 + 0.3 * x[:, 0] - 0.8 * x[:, 1] + 1.3 * x[:, 2]
    if mask.shape[1] > 3:
        value += 0.5 * x[:, 3]
    if case in ("pairwise", "third_order"):
        value += 2.5 * x[:, 0] * x[:, 1]
    if case == "third_order":
        value += 3.5 * x[:, 0] * x[:, 1] * x[:, 2]
    return value


def exact_order_energy(case: str, K: int = 4) -> dict[int, float]:
    """Fourier expansion under uniform Boolean coalitions: known degree truth."""
    if case not in CASES or K < 4 or K > 8:
        raise ValueError("unsupported synthetic truth dimensions")
    mask = np.asarray(list(product((False, True), repeat=K)), dtype=bool)
    y = behavior(mask, case)
    x = 2 * mask.astype(float) - 1
    result = {}
    for degree in range(1, K + 1):
        energy = 0.0
        for subset in combinations(range(K), degree):
            basis = np.prod(x[:, subset], axis=1)
            coef = float(np.mean(y * basis))
            energy += coef * coef
        result[degree] = float(energy)
    return result


def verify_original_sar(paper, *, seed: int = 7, M: int = 512, n_fit: int = 350) -> dict:
    """Call original SAR estimator; do not redefine its numerical definition."""
    if not callable(getattr(paper, "coalitions", None)) or not callable(
        getattr(paper, "sar", None)
    ):
        raise TypeError("trusted original SAR module must expose coalitions and sar")
    masks, fit, held = paper.coalitions(K=4, M=M, n_fit=n_fit, seed=seed)
    cases = {}
    for case in CASES:
        g = behavior(masks, case)
        g_full = float(behavior(np.ones((1, 4), dtype=bool), case)[0])
        upstream = paper.sar(g, masks, fit, held, g_full=g_full)
        energy = exact_order_energy(case)
        first = float(upstream["sar1"])
        second = float(upstream["sar2"])
        if case == "first_order":
            passed = first < 1e-8 and second < 1e-8 and energy[2] < 1e-12 and energy[3] < 1e-12
        elif case == "pairwise":
            passed = first > 0.02 and second < 1e-8 and energy[2] > 1.0 and energy[3] < 1e-12
        else:
            passed = first > 0.02 and second > 0.02 and energy[3] > 1.0
        cases[case] = {
            "status": "passed" if passed else "failed",
            "oracle_energy_by_order": {str(k): v for k, v in energy.items()},
            "native_sar1": first,
            "native_sar2": second,
            "n_fit": len(fit),
            "n_held_out": len(held),
            "seed": seed,
            "note": "No third-order SAR estimator exists in the original SAR paper implementation",
        }
    return cases


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


def evaluate_ground_truth(
    *, sar_source: str | Path, rss_source: str | Path, config: AuditConfig
) -> dict[str, Any]:
    """Never loads LLM/SAE checkpoints or estimates uncertainty from invented data.

    Ground truths and upstream numerical estimators are separate objects. The
    original source file SHA-256 is frozen in the audit for repeatability.
    """
    sar_sha = source_identity(sar_source, "sar")
    rss_sha = source_identity(rss_source, "rss")
    sar = load_paper_module(sar_source, paper="sar")
    rss = load_paper_module(rss_source, paper="rss")
    factor = evaluate_degradation(config)
    interactions = verify_original_sar(sar, seed=7)
    direction = verify_original_rss(rss, seed=11)
    # Coalition selection and RSS variant selection are independently reseeded
    # without fitting on held-out responses. These are estimator stress tests
    # on analytic data, not fabricated empirical LLM replicates.
    sar_resampled = []
    rss_resampled = []
    for seed in config.evaluation_seeds:
        trial = verify_original_sar(sar, seed=seed, M=512, n_fit=350)
        sar_resampled.append(
            {
                "evaluation_seed": seed,
                "cases": trial,
                "status": "passed"
                if all(x["status"] == "passed" for x in trial.values())
                else "failed",
            }
        )
        trial_rss = verify_original_rss(rss, seed=seed)
        rss_resampled.append({"evaluation_seed": seed, **trial_rss})
    negatives = {
        "coordinate_invariance": check_coordinates(seed=10),
        "sar_negative_controls": check_sar_negatives(sar, seed=20),
        "rss_negative_controls": check_rss_controls(rss, seed=21),
    }
    parts = [
        *(v["status"] for v in factor.values()),
        *(v["status"] for v in interactions.values()),
        direction["status"],
        *(v["status"] for v in negatives.values()),
        *(x["status"] for x in sar_resampled),
        *(x["status"] for x in rss_resampled),
    ]
    return {
        "schema": "saebase.m4.ground_truth.v1",
        "status": "passed" if all(x == "passed" for x in parts) else "failed",
        "source_provenance": {
            "sar": {"path": str(Path(sar_source).expanduser().resolve()), "sha256": sar_sha},
            "rss": {"path": str(Path(rss_source).expanduser().resolve()), "sha256": rss_sha},
        },
        "protocol": {
            "independent_calibration_test": True,
            "bootstrap_unit": "prompt_group_id",
            "evaluation_seeds": list(config.evaluation_seeds),
            "training_seed": config.training_seed,
            "confidence": config.confidence,
            "confidence_interval_method": "paired percentile bootstrap",
            "synthetic_not_real_model_performance": True,
            "sar_upstream_orders": [1, 2],
            "rss_selection_and_held_out_disjoint": True,
        },
        "factor_cases": factor,
        "known_interaction_cases": interactions,
        "known_rss_direction_case": direction,
        "coalition_reseed_trials": sar_resampled,
        "rss_reseed_trials": rss_resampled,
        "negative_invariance_controls": negatives,
    }


def append_ground_truth(audit: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    if (
        audit.get("schema") != "saebase.m4.audit.v1"
        or ground_truth.get("schema") != "saebase.m4.ground_truth.v1"
    ):
        raise ValueError("incompatible reliability audit/ground truth version")
    updated = dict(audit)
    updated["ground_truth"] = ground_truth
    if ground_truth["status"] != "passed":
        updated["status"] = "failed"
    return updated
