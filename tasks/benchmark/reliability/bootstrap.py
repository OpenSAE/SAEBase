"""Deterministic percentile cluster bootstrap and strictly paired SAE comparisons."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np

from .schema import Observation, AuditConfig
from .resampling import group_samples, group_sums, estimate_with_draws, draw_group_indices


def _summary(values: np.ndarray, point: float, confidence: float) -> dict:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return {
            "status": "undefined",
            "reason": "zero valid bootstrap replicates",
            "n_valid": 0,
            "interval": None,
        }
    alpha = (1 - confidence) / 2
    return {
        "status": "completed",
        "point_estimate": float(point),
        "bootstrap_mean": float(valid.mean()),
        "bootstrap_std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "interval": [float(np.quantile(valid, alpha)), float(np.quantile(valid, 1 - alpha))],
        "confidence": confidence,
        "ci_method": "percentile",
        "n_valid": len(valid),
    }


def bootstrap_metric(observations: Sequence[Observation], config: AuditConfig) -> dict:
    names, n, d = group_sums(observations)
    if len(names) < 2:
        return {
            "status": "undefined",
            "reason": "fewer than two independent prompt groups",
            "n_groups": len(names),
            "n_observations": len(observations),
            "n_valid": 0,
            "seed_kind": "evaluation",
            "training_seed": config.training_seed,
        }
    point = float(n.sum() / d.sum())
    seeded = []
    for seed in config.evaluation_seeds:
        indices = draw_group_indices(len(names), config.bootstrap_replicates, seed=seed)
        stat = estimate_with_draws(n, d, indices)
        seeded.append({"seed": seed, **_summary(stat, point, config.confidence)})
    pooled = np.concatenate(
        [
            estimate_with_draws(
                n, d, draw_group_indices(len(names), config.bootstrap_replicates, seed=s)
            )
            for s in config.evaluation_seeds
        ]
    )
    return {
        "schema": "saebase.m4.metric_audit.v1",
        **_summary(pooled, point, config.confidence),
        "n_groups": len(names),
        "n_observations": len(observations),
        "n_units": int(d.sum()),
        "group_ids": list(names),
        "n_evaluation_seeds": len(config.evaluation_seeds),
        "bootstrap_replicates_per_seed": config.bootstrap_replicates,
        "seed_kind": "evaluation",
        "training_seed": config.training_seed,
        "seed_results": seeded,
        "invalid_replicates": len(pooled) - int(np.isfinite(pooled).sum()),
    }


def _same_sample_keys(a: dict, b: dict):
    if tuple(a) != tuple(b):
        raise ValueError("paired bootstrap needs identical prompt group IDs")
    for group in a:
        left = {(x.sample_id, x.condition_id, x.split) for x in a[group]}
        right = {(x.sample_id, x.condition_id, x.split) for x in b[group]}
        if left != right:
            raise ValueError(f"paired group sample/condition mismatch: {group}")


def paired_bootstrap(
    left: Sequence[Observation], right: Sequence[Observation], config: AuditConfig
) -> dict:
    """Bootstrap a comparison with the *same* sampled prompt groups per SAE."""
    la, ra = group_samples(left), group_samples(right)
    if left[0].metric_id != right[0].metric_id or left[0].dataset_id != right[0].dataset_id:
        raise ValueError("paired comparison metric/dataset mismatch")
    if left[0].split != right[0].split or left[0].condition_id != right[0].condition_id:
        raise ValueError("paired comparison split/condition mismatch")
    if left[0].sae_id == right[0].sae_id:
        raise ValueError("paired comparison needs distinct SAE IDs")
    _same_sample_keys(la, ra)
    lnames, ln, ld = group_sums(left)
    rnames, rn, rd = group_sums(right)
    if lnames != rnames:
        raise ValueError("paired group ordering mismatch")
    if len(lnames) < 2:
        return {
            "status": "undefined",
            "reason": "fewer than two paired prompt groups",
            "n_groups": len(lnames),
            "n_valid": 0,
        }
    point = float(ln.sum() / ld.sum() - rn.sum() / rd.sum())
    samples = []
    for seed in config.evaluation_seeds:
        draw = draw_group_indices(len(lnames), config.bootstrap_replicates, seed=seed)
        samples.append(estimate_with_draws(ln, ld, draw) - estimate_with_draws(rn, rd, draw))
    combined = np.concatenate(samples)
    return {
        "schema": "saebase.m4.paired.v1",
        **_summary(combined, point, config.confidence),
        "left_sae": left[0].sae_id,
        "right_sae": right[0].sae_id,
        "n_groups": len(lnames),
        "n_matched_sample_groups": len(lnames),
        "invalid_replicates": int((~np.isfinite(combined)).sum()),
        "evaluation_seeds": list(config.evaluation_seeds),
        "bootstrap_replicates_per_seed": config.bootstrap_replicates,
    }


def sample_size_curve(observations: Sequence[Observation], config: AuditConfig) -> list[dict]:
    names, n, d = group_sums(observations)
    if len(names) < 2:
        return []
    available = len(names)
    sizes = sorted({min(available, x) for x in config.sample_sizes if x >= 2} | {available})
    curve = []
    for size in sizes:
        observations_by_group = group_samples(observations)
        for seed in config.evaluation_seeds:
            chosen = np.random.default_rng(seed).choice(available, size=size, replace=False)
            selected = [row for idx in chosen for row in observations_by_group[names[int(idx)]]]
            nrep = bootstrap_metric(
                selected,
                AuditConfig(
                    evaluation_seeds=(seed,),
                    bootstrap_replicates=config.bootstrap_replicates,
                    confidence=config.confidence,
                    sample_sizes=(),
                    top_k=config.top_k,
                    training_seed=config.training_seed,
                ),
            )
            curve.append(
                {
                    "requested_groups": size,
                    "actual_groups": len(chosen),
                    "seed": seed,
                    "status": nrep["status"],
                    "point_estimate": nrep.get("point_estimate"),
                    "interval": nrep.get("interval"),
                }
            )
    return curve
