"""Capture-owned rankings of measured answer scores and fixed-background interactions."""

from itertools import combinations
import math


def probabilities(g, token_count):
    rows = []
    for name, factor in [("平均token概率", 1), ("答案联合概率", token_count)]:
        values = {state: 100 * math.exp(factor * value) for state, value in g.items()}
        baselines = [value for state, value in values.items() if state != "11"]
        lift = values["11"] - max(baselines) if "11" in values else values["01"] - values["00"]
        rows.append({"metric": name, "percent": values, "change_percentage_points": lift})
    return rows


def choose_pair(ids, scores, token_count, *, return_ranking=False):
    if return_ranking:
        import numpy as np

        values = np.asarray(scores, dtype=np.float64)
        backgrounds = np.arange(len(values))
        ranking = []
        for i, j in combinations(range(len(ids)), 2):
            bits = (1 << i) | (1 << j)
            valid = backgrounds[(backgrounds & bits) == 0]
            interactions = (
                values[valid | bits]
                - values[valid | (1 << i)]
                - values[valid | (1 << j)]
                + values[valid]
            )
            # argmax preserves the original first-background tie break.
            index = int(np.argmax(interactions))
            background = int(valid[index])
            states = {
                "00": background,
                "10": background | (1 << i),
                "01": background | (1 << j),
                "11": background | bits,
            }
            g = {name: float(values[state]) for name, state in states.items()}
            ranking.append(
                {
                    "pair_ids": [ids[i], ids[j]],
                    "interaction": float(interactions[index]),
                    "g": g,
                    "states": states,
                    "background": {
                        str(fid): (background >> k) & 1
                        for k, fid in enumerate(ids)
                        if k not in (i, j)
                    },
                    "probabilities": probabilities(g, token_count),
                }
            )
        return sorted(ranking, key=lambda row: -row["interaction"])
    per_pair = {}
    best = None
    qualified = None
    all_other_retained = None
    for i, j in combinations(range(len(ids)), 2):
        bits = (1 << i) | (1 << j)
        for background in range(len(scores)):
            if background & bits:
                continue
            states = {
                "00": background,
                "10": background | (1 << i),
                "01": background | (1 << j),
                "11": background | bits,
            }
            g = {name: float(scores[state]) for name, state in states.items()}
            interaction = g["11"] - g["10"] - g["01"] + g["00"]
            candidate = {
                "pair_ids": [ids[i], ids[j]],
                "interaction": interaction,
                "g": g,
                "states": states,
                "background": {
                    str(fid): (background >> k) & 1 for k, fid in enumerate(ids) if k not in (i, j)
                },
                "probabilities": probabilities(g, token_count),
            }
            key = (i, j)
            if key not in per_pair or interaction > per_pair[key]["interaction"]:
                per_pair[key] = candidate
            if best is None or interaction > best["interaction"]:
                best = candidate
            if background | bits == len(scores) - 1:
                if all_other_retained is None or interaction > all_other_retained["interaction"]:
                    all_other_retained = candidate
            p = candidate["probabilities"][0]["percent"]
            base = [p[s] for s in ("00", "10", "01")]
            if (
                min(base) > 50
                and max(base) - min(base) <= 5
                and p["11"] > max(base)
                and interaction > 0
            ):
                if qualified is None or interaction > qualified["interaction"]:
                    qualified = candidate
    if return_ranking:
        return sorted(per_pair.values(), key=lambda row: -row["interaction"])
    return best, qualified, all_other_retained


def rank_singles(feature_ids, baseline, deleted_scores):
    if len(feature_ids) != len(deleted_scores):
        raise ValueError("Single scores must align with candidate IDs")
    return sorted(
        [
            {"feature_id": int(fid), "g": float(score), "delta_g": float(score - baseline)}
            for fid, score in zip(feature_ids, deleted_scores)
        ],
        key=lambda row: -row["g"],
    )
