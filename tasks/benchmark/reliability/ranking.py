"""Pair-aligned ranking stability; never rank contextual SAR/RSS and unrelated metrics."""

from __future__ import annotations
import itertools
import numpy as np

from tasks.benchmark.backend.registry import get_metric
from .resampling import group_samples, group_sums, estimate_with_draws, draw_group_indices
from .schema import Observation, AuditConfig


def rank_stability(by_sae: dict[str, list[Observation]], cfg: AuditConfig) -> dict:
    if len(by_sae) < 2:
        return {"status": "undefined", "reason": "ranking requires at least two SAEs"}
    first = next(iter(by_sae.values()))
    spec = get_metric(first[0].metric_id)
    if spec.direction not in {"higher", "lower"}:
        return {"status": "unsupported", "reason": "metric direction is contextual, not a ranking"}
    sae_ids = sorted(by_sae)
    grouped = {s: group_samples(by_sae[s]) for s in sae_ids}
    keys = tuple(grouped[sae_ids[0]])
    for sae in sae_ids[1:]:
        if tuple(grouped[sae]) != keys:
            return {
                "status": "blocked",
                "reason": "SAEs are not aligned on identical prompt groups",
            }
        for g in keys:
            a = {(x.sample_id, x.condition_id) for x in grouped[sae_ids[0]][g]}
            b = {(x.sample_id, x.condition_id) for x in grouped[sae][g]}
            if a != b:
                return {"status": "blocked", "reason": "SAE sample IDs differ within paired group"}
    if len(keys) < 2:
        return {"status": "undefined", "reason": "too few prompt groups for ranking"}
    sums = [group_sums(by_sae[s])[1:] for s in sae_ids]
    means = np.asarray([n.sum() / d.sum() for n, d in sums])
    if len(set(means.tolist())) != len(means):
        return {"status": "undefined", "reason": "tied original SAE scores"}
    sign = -1 if spec.direction == "higher" else 1
    original = np.argsort(sign * means)
    k = min(cfg.top_k, len(sae_ids))
    tau = []
    overlap = []
    flips = 0
    invalid = 0
    for seed in cfg.evaluation_seeds:
        draws = draw_group_indices(len(keys), cfg.bootstrap_replicates, seed=seed)
        for draw in draws:
            estimates = np.asarray([estimate_with_draws(n, d, draw[None, :])[0] for n, d in sums])
            if len(set(estimates.tolist())) != len(estimates):
                invalid += 1
                continue
            rank = np.argsort(sign * estimates)
            reference_pos = {v: i for i, v in enumerate(original)}
            trial_pos = {v: i for i, v in enumerate(rank)}
            pairs = list(itertools.combinations(range(len(sae_ids)), 2))
            discord = sum(
                (reference_pos[i] - reference_pos[j]) * (trial_pos[i] - trial_pos[j]) < 0
                for i, j in pairs
            )
            tau.append(1 - 2 * discord / len(pairs))
            flips += int(discord > 0)
            overlap.append(len(set(original[:k]) & set(rank[:k])) / k)
    if not tau:
        return {"status": "undefined", "reason": "all bootstrap ranks tied"}
    return {
        "schema": "saebase.m4.rank.v1",
        "status": "completed",
        "metric_id": first[0].metric_id,
        "condition_id": first[0].condition_id,
        "direction": spec.direction,
        "original_order": [sae_ids[i] for i in original],
        "n_saes": len(sae_ids),
        "n_groups": len(keys),
        "top_k": k,
        "mean_kendall_tau": float(np.mean(tau)),
        "mean_top_k_overlap": float(np.mean(overlap)),
        "ranking_reversal_frequency": float(flips / len(tau)),
        "n_valid_bootstrap_ranks": len(tau),
        "invalid_tied_ranks": invalid,
        "seed_kind": "evaluation",
        "evaluation_seeds": list(cfg.evaluation_seeds),
    }
