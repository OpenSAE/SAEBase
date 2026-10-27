"""Conditional Localize reliability and descriptive cross-SAE comparisons.

One complete Boolean factorial grid is ONE input population, not 2**n
independent observations. Do not apply M4 prompt-group bootstrap to its states.
Finite-pair repetitions assess Monte Carlo estimation stability, not generalization.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any
import math

import numpy as np

from tasks.LLM.benchmark.backend.adapters.localization import (
    grid_from_artifact,
    validate_localization_result,
)
from tasks.LLM.benchmark.integrations.localization import _original_localize


def audit_localization(
    root: Any,
    config: Any,
    row: dict[str, Any],
    cfg: Any,
    *,
    paper: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify full curve, rerun native paired estimates on cached codes only."""
    data = validate_localization_result(root, config, row, paper=paper)
    grid = grid_from_artifact(root, data)  # verifies the original M1 source shards
    native = _original_localize(paper)
    exact = np.asarray(data["curve"], dtype=np.float64)
    W = data["total_variance"]
    sampling = {
        "status": "not_requested",
        "interpretation": "finite-pair Monte Carlo stability conditional on one fixed complete grid",
        "independent_prompt_groups": 1,
        "population_confidence_interval": None,
    }
    if W > 0 and data["pairs_per_scope"] is not None:
        method = getattr(native, "estimated_scope_curve", None)
        if not callable(method):
            raise ValueError("original Localize estimator missing estimated_scope_curve()")
        seed_results = []
        for seed in cfg.evaluation_seeds:
            curve = np.asarray(
                method(
                    grid.codes,
                    pairs_per_scope=data["pairs_per_scope"],
                    seed=seed,
                ),
                dtype=np.float64,
            )
            if curve.shape != exact.shape or not np.isfinite(curve).all():
                raise ValueError("native paired Localize estimator returned invalid curve")
            eligible = np.flatnonzero(curve <= 0.1 * W + 1e-12)
            budget = int(eligible[0]) if eligible.size else None
            seed_results.append(
                {
                    "seed": seed,
                    "budget_at_90pct": budget,
                    "budget_fraction_at_90pct": (
                        budget / (grid.n_factors * grid.n_features) if budget is not None else None
                    ),
                    "max_normalized_curve_error": float(np.max(np.abs(curve - exact)) / W),
                    "mean_normalized_curve_error": float(np.mean(np.abs(curve - exact)) / W),
                }
            )
        valid = [x for x in seed_results if x["budget_at_90pct"] is not None]
        sampling = {
            **sampling,
            "status": "completed" if len(valid) == len(seed_results) else "partially_undefined",
            "pairs_per_candidate_scope": data["pairs_per_scope"],
            "n_evaluation_seeds": len(cfg.evaluation_seeds),
            "seed_results": seed_results,
            "budget_fraction_range": (
                [
                    min(x["budget_fraction_at_90pct"] for x in valid),
                    max(x["budget_fraction_at_90pct"] for x in valid),
                ]
                if valid
                else None
            ),
            "maximum_normalized_curve_error_across_seeds": max(
                x["max_normalized_curve_error"] for x in seed_results
            ),
        }
    audit = {
        "model_id": row["model_id"],
        "sae_id": row["sae_id"],
        "metric_id": row["metric_id"],
        "dataset_id": row["dataset_id"],
        "split": row["split"],
        "condition_id": None,
        "status": "completed",
        "reason": None,
        "metric_version": row["metric_version"],
        "protocol_id": row["protocol_id"],
        "M3_metric_value": row["value"],
        "audit": {
            "status": "completed",
            "type": "controlled_input_grid",
            "point_estimate": row["value"],
            "n_groups": 1,
            "n_states": grid.codes.shape[0],
            "n_features": grid.n_features,
            "design_key": grid.design_key,
            "population_confidence_interval": None,
            "sampling_stability": sampling,
        },
        "sample_size_curve": [],
    }
    return audit, data


def compare_localization(
    rows_and_artifacts: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compare matched design and model/hook only; no pretend independent CI."""
    by_key: dict[tuple[Any, ...], dict[str, tuple[Any, ...]]] = defaultdict(dict)
    for row, artifact, audit in rows_and_artifacts:
        key = (
            row["model_id"],
            row["dataset_id"],
            row["split"],
            row["site"]["layer"],
            row["site"]["hook_name"],
            row["site"]["position"],
            row["metric_id"],
        )
        by_key[key][row["sae_id"]] = (row, artifact, audit)
    comparisons = []
    for key, candidates in sorted(by_key.items(), key=lambda x: str(x[0])):
        for left_id, right_id in combinations(sorted(candidates), 2):
            left, a, a_audit = candidates[left_id]
            right, b, b_audit = candidates[right_id]
            ma, mb = a["design_manifest"], b["design_manifest"]
            if (
                ma["design_key"] != mb["design_key"]
                or ma["factors"] != mb["factors"]
                or a["n_factors"] != b["n_factors"]
                or ma["states"] != mb["states"]
                or a["source_sha256"] != b["source_sha256"]
            ):
                comparisons.append(
                    {
                        "status": "unsupported",
                        "metric_id": left["metric_id"],
                        "left_sae": left_id,
                        "right_sae": right_id,
                        "reason": "different controlled input designs or estimators",
                    }
                )
                continue
            if a["relative_omission"] != b["relative_omission"]:
                raise ValueError("Localize comparisons require equal omission tolerances")
            left_frac = a["budget_fraction_at_90pct"]
            right_frac = b["budget_fraction_at_90pct"]
            if not all(math.isfinite(x) for x in (left_frac, right_frac)):
                raise ValueError("nonfinite Localize fractions")
            la = a_audit["audit"]["sampling_stability"]
            lb = b_audit["audit"]["sampling_stability"]
            paired_differences = None
            if la["status"] == lb["status"] == "completed" and (
                la["pairs_per_candidate_scope"] == lb["pairs_per_candidate_scope"]
            ):
                ordered_left = la["seed_results"]
                ordered_right = lb["seed_results"]
                if [x["seed"] for x in ordered_left] == [x["seed"] for x in ordered_right]:
                    diffs = [
                        x["budget_fraction_at_90pct"] - y["budget_fraction_at_90pct"]
                        for x, y in zip(ordered_left, ordered_right)
                    ]
                    paired_differences = {
                        "minimum": float(min(diffs)),
                        "maximum": float(max(diffs)),
                        "n_paired_seeds": len(diffs),
                        "interpretation": "seed-matched finite-pair sensitivity, not population CI",
                    }
            comparisons.append(
                {
                    "status": "completed",
                    "comparison_type": "descriptive_matched_controlled_grid",
                    "metric_id": left["metric_id"],
                    "model_id": left["model_id"],
                    "dataset_id": left["dataset_id"],
                    "design_key": ma["design_key"],
                    "left_sae": left_id,
                    "right_sae": right_id,
                    "left_n_features": a["n_features"],
                    "right_n_features": b["n_features"],
                    "left_budget": a["budget_at_90pct"],
                    "right_budget": b["budget_at_90pct"],
                    "absolute_budget_difference": (
                        a["budget_at_90pct"] - b["budget_at_90pct"]
                        if a["n_features"] == b["n_features"]
                        else None
                    ),
                    "left_budget_fraction": left_frac,
                    "right_budget_fraction": right_frac,
                    "budget_fraction_difference": left_frac - right_frac,
                    "paired_sampling_difference": paired_differences,
                    "population_confidence_interval": None,
                    "ranking_claim": None,
                    "interpretation": "contextual dependency budget; no SAE quality ranking",
                }
            )
    return comparisons
