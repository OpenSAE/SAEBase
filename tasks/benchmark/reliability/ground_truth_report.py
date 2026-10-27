"""Deterministic scientific validity controls using original SAR/RSS source files."""

from __future__ import annotations
from pathlib import Path
from typing import Any

from tasks.LLM.benchmark.integrations import load_paper_module
from tasks.LLM.benchmark.backend.adapters.papers import source_identity
from .schema import AuditConfig
from .interaction_truth import verify_original_sar
from .rss_truth import verify_original_rss
from .controls import check_coordinates, check_sar_negatives, check_rss_controls
from .discriminability import evaluate_degradation


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
        raise ValueError("incompatible M4 audit/ground truth version")
    updated = dict(audit)
    updated["ground_truth"] = ground_truth
    if ground_truth["status"] != "passed":
        updated["status"] = "failed"
    return updated
