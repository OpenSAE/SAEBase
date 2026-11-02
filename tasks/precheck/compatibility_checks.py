"""Plan only the declared Dataset/Model/SAE-site compatibility targets.

This module does NOT perform compatibility tests or promote a selected target
to PASS. Those decisions require actual isolated worker receipts.
"""
from __future__ import annotations

import re
from typing import Any

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
STATES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _precondition(required: tuple[str, ...], receipts: dict[str, dict]) -> dict[str, Any]:
    actual = {name: receipts.get(name, {}).get("status", "NOT_RUN") for name in required}
    if any(value not in STATES for value in actual.values()):
        raise ValueError("Unsupported component receipt state")
    if any(value in {"FAIL", "BLOCKED"} for value in actual.values()):
        return {"status": "BLOCKED", "eligible": False, "dependency_states": actual}
    return {
        "status": "NOT_RUN",
        "eligible": all(value == "PASS" for value in actual.values()),
        "dependency_states": actual,
    }


def compatibility_matrix(
    datasets: list[str],
    sites: list[dict[str, Any]],
    receipts: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Enumerate only explicit source-owned Model/SAE sites and QA datasets.

    A PASS in the component receipts is necessary but never sufficient for
    compatibility PASS; every pair must still run its own real test.
    """
    if not isinstance(datasets, list) or not 1 <= len(datasets) <= 500:
        raise ValueError("Invalid declared Dataset selection")
    keys = [_name(key, "Dataset") for key in datasets]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate Dataset in frozen target list")
    if not isinstance(sites, list) or not 1 <= len(sites) <= 500:
        raise ValueError("Invalid declared SAE Site selection")
    models: list[str] = []
    sites_flat = []
    identities: set[str] = set()
    for row in sites:
        if not isinstance(row, dict):
            raise ValueError("Invalid SAE Site row")
        model = _name(row.get("model_key"), "Model")
        sae = _name(row.get("sae_key"), "SAE")
        layer = row.get("layer")
        hook = row.get("hook_name")
        if type(layer) is not int or not 0 <= layer <= 500 or not isinstance(hook, str):
            raise ValueError("Invalid SAE layer or hook")
        expected_id = f"{model}:{sae}:L{layer}"
        if not hook.startswith(f"blocks.{layer}.") or row.get("site_id") != expected_id:
            raise ValueError("Site identity, layer and hook mismatch")
        if expected_id in identities:
            raise ValueError("Duplicate SAE Site in frozen target list")
        identities.add(expected_id)
        sites_flat.append((model, expected_id))
        if model not in models:
            models.append(model)
    if len(keys) * len(sites_flat) > 50_000:
        raise ValueError("Compatibility matrix exceeds bounded target limit")
    supplied = receipts or {}
    if not isinstance(supplied, dict):
        raise ValueError("Invalid component receipt mapping")
    pairs = []
    for model in models:
        for dataset in keys:
            required = (f"dataset:{dataset}", f"model:{model}")
            pairs.append({
                "id": f"model_dataset:{model}:{dataset}",
                "kind": "model_dataset",
                "model_key": model,
                "dataset_id": dataset,
                "required_components": list(required),
                **_precondition(required, supplied),
            })
    for model, site in sites_flat:
        for dataset in keys:
            required = (f"dataset:{dataset}", f"model:{model}", f"sae:{site}")
            pairs.append({
                "id": f"sae_site_dataset:{site}:{dataset}",
                "kind": "sae_site_dataset",
                "site_id": site,
                "model_key": model,
                "dataset_id": dataset,
                "required_components": list(required),
                **_precondition(required, supplied),
            })
    return {
        "schema": "saebase.precheck.compatibility_targets.v1",
        "status": "PLAN_ONLY",
        "models": len(models),
        "datasets": len(keys),
        "sae_sites": len(sites_flat),
        "model_dataset_targets": len(models) * len(keys),
        "sae_site_dataset_targets": len(sites_flat) * len(keys),
        "pairs": pairs,
        "scientific_acceptance": "NOT_COVERED",
    }
