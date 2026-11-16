"""No-download model comparison preflight; never claims unverified availability."""

from __future__ import annotations
from saebase.api.sae_profiles import get_profile
from .schema import MatrixConfig


def plan_matrix(matrix: MatrixConfig) -> dict:
    pairs = sum(len(t.sae_keys) for t in matrix.targets)
    measurements = sum(
        len(matrix.prompts) * len(matrix.conditions) * len(t.sae_keys) for t in matrix.targets
    )
    if measurements > matrix.max_measurements:
        raise ValueError(
            f"{measurements} planned batch interventions exceed budget {matrix.max_measurements}"
        )
    return {
        "schema": "saebase.model_comparison.plan",
        "matrix_id": matrix.matrix_id,
        "name": matrix.name,
        "model_families": sorted({t.family for t in matrix.targets}),
        "models": len(matrix.targets),
        "distinct_checkpoints": pairs,
        "n_prompts_per_model": len(matrix.prompts),
        "planned_intervention_measurements": measurements,
        "metric_jobs": pairs * len(matrix.metrics),
        "known_sites": [
            {
                "model_key": t.model_key,
                "family": t.family,
                "site": t.to_dict()["site"],
                "saes": [
                    {
                        "key": k,
                        "release": get_profile(k).sae_release,
                        "sae_id": get_profile(k).sae_id(t.site.layer),
                        "loader": get_profile(k).loader,
                    }
                    for k in t.sae_keys
                ],
            }
            for t in matrix.targets
        ],
        "budgets": {
            "max_measurements": matrix.max_measurements,
            "max_runtime_seconds": matrix.max_runtime_seconds,
            "max_token_count": matrix.max_token_count,
        },
        "estimated_seconds": None,
        "estimated_ram_mb": None,
        "checkpoint_downloads": 0,
        "note": "Dry-run checks registration and budget; actual checkpoint identities and license terms must be verified in real acceptance.",
    }
