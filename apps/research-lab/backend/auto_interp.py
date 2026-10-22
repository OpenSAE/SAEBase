"""Explicitly authorized explanation candidates with disjoint held-out checks.

This does not pretend there is a built-in LLM interpretation model or paid
service. The researcher supplies a generator and, optionally, a predictor;
the output is a provenance-bearing hypothesis requiring human publication
approval and further scientific validation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any


def propose_interpretation(
    *,
    site_id: str,
    feature_id: int,
    discovery: Sequence[dict[str, Any]],
    heldout: Sequence[dict[str, Any]],
    generator: Callable[[list[str]], str],
    generator_model: str,
    generator_method: str,
    authorized_budget_usd: float,
    estimated_cost_usd: float,
    predictor: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Generate one description and assess activation prediction when supplied.

    `discovery`/ `heldout` have group_id, text and activation. The predictor
    is a separately supplied classifier; agreement with the SAE's activation
    presence is *not* ground-truth semantic correctness.
    """
    if type(feature_id) is not int or feature_id < 0 or not site_id:
        raise ValueError("Exact SAE feature identity required")
    if not callable(generator) or (predictor is not None and not callable(predictor)):
        raise TypeError("A callable generator/predictor is required")
    if not generator_model or not generator_method:
        raise ValueError("Name the actual generator model and method")
    if (
        type(authorized_budget_usd) not in (int, float)
        or type(estimated_cost_usd) not in (int, float)
        or not math.isfinite(authorized_budget_usd)
        or not math.isfinite(estimated_cost_usd)
        or estimated_cost_usd < 0
        or authorized_budget_usd < estimated_cost_usd
    ):
        raise PermissionError("Explicit sufficient generation budget required before a call")
    if not discovery or not heldout or len(discovery) > 100 or len(heldout) > 100:
        raise ValueError("Separate bounded discovery and evaluation records required")
    def check(rows: Sequence[dict[str, Any]]) -> list[str]:
        ids = []
        for row in rows:
            group, text, value = (
                row.get("group_id"), row.get("text"), row.get("activation")
            )
            if not isinstance(group, str) or not group:
                raise ValueError("Missing independent prompt group")
            if not isinstance(text, str) or not text or len(text) > 6000:
                raise ValueError("Invalid explanation example")
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Missing actual SAE activation")
            ids.append(group)
        if len(set(ids)) != len(ids):
            raise ValueError("Each split requires distinct prompt groups")
        return ids
    discovery_ids, heldout_ids = check(discovery), check(heldout)
    if set(discovery_ids) & set(heldout_ids):
        raise ValueError("Discovery and held-out groups overlap")
    snippets = [row["text"] for row in discovery if row["activation"] != 0]
    if not snippets:
        raise ValueError("No observed activating contexts to describe")
    # External services are never called implicitly. This direct invocation
    # is the owner's chosen provider and budget, not a background job.
    description = generator(snippets)
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 900:
        raise ValueError("Generator returned no usable candidate")
    result: dict[str, Any] = {
        "site_id": site_id,
        "feature_id": feature_id,
        "origin": "model_candidate",
        "description": description.strip(),
        "author_or_model": generator_model,
        "method": generator_method,
        "approved": False,
        "status": "hypothesis",
        "heldout_score": None,
        "discovery_group_ids": discovery_ids,
        "evaluation_group_ids": heldout_ids,
        "evaluation_kind": "SAE_binary_activation_prediction_not_semantic_truth",
        "estimated_cost_usd": float(estimated_cost_usd),
    }
    if predictor is None:
        return result
    labels = [row["activation"] != 0 for row in heldout]
    predicted = [predictor(description, row["text"]) for row in heldout]
    if not all(type(item) is bool for item in predicted):
        raise TypeError("Predictor must return a boolean for each held-out example")
    positives = [i for i, value in enumerate(labels) if value]
    negatives = [i for i, value in enumerate(labels) if not value]
    if not positives or not negatives:
        raise ValueError("Balanced accuracy requires activating and inactive held-out groups")
    tpr = sum(predicted[i] for i in positives) / len(positives)
    tnr = sum(not predicted[i] for i in negatives) / len(negatives)
    result.update({
        "status": "heldout_tested",
        "heldout_score": float((tpr + tnr) / 2),
        "heldout_n_positive": len(positives),
        "heldout_n_negative": len(negatives),
        "heldout_errors": [
            {"group_id": heldout[i]["group_id"], "observed_active": labels[i],
             "predicted_active": predicted[i]}
            for i in range(len(heldout)) if predicted[i] != labels[i]
        ],
    })
    return result
