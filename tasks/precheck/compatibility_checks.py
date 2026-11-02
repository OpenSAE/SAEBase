"""Build bounded, source-owned compatibility targets without pretending to run them.

This is a reusable Precheck planner. Target discovery and actual workers are
separate responsibilities; eligible means runnable, never that a pair passed.
"""

from __future__ import annotations

import re
from typing import Any

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_STATES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})


def _keys(values: list[str], kind: str) -> list[str]:
    if not isinstance(values, list) or not values or len(values) > 500:
        raise ValueError(f"Invalid {kind} target selection")
    if any(not isinstance(x, str) or not _NAME.fullmatch(x) for x in values):
        raise ValueError(f"Unsafe {kind} target identity")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {kind} target")
    return sorted(values)


def declared_site(model: str, sae: str, layer: int, hook: str) -> dict[str, Any]:
    """Validate a declared model-owned site, not a guessed default layer."""
    if not isinstance(model, str) or not _NAME.fullmatch(model):
        raise ValueError("Invalid site Model")
    if not isinstance(sae, str) or not _NAME.fullmatch(sae):
        raise ValueError("Invalid site SAE")
    if type(layer) is not int or not 0 <= layer <= 500:
        raise ValueError("Invalid SAE layer")
    if not isinstance(hook, str) or not hook.startswith(f"blocks.{layer}."):
        raise ValueError("Site hook does not match layer")
    return {
        "model_key": model,
        "sae_key": sae,
        "layer": layer,
        "hook_name": hook,
        "site_id": f"{model}:{sae}:L{layer}",
    }


def plan_compatibility(
    datasets: list[str],
    models: list[str],
    sites: list[dict[str, Any]],
    receipts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enumerate real declared targets with local dependency blocking.

    Component PASS permits actual pair work but never marks pair PASS.
    """
    data = _keys(datasets, "Dataset")
    model_keys = _keys(models, "Model")
    if not isinstance(sites, list) or len(sites) > 500:
        raise ValueError("Invalid site list")
    known = set(model_keys)
    selected = []
    seen = set()
    for site in sites:
        if not isinstance(site, dict):
            raise ValueError("Invalid SAE site")
        normalized = declared_site(
            site.get("model_key"), site.get("sae_key"),
            site.get("layer"), site.get("hook_name"),
        )
        if normalized["model_key"] not in known:
            raise ValueError("SAE site belongs to unselected Model")
        if site.get("site_id") != normalized["site_id"]:
            raise ValueError("Site identity mismatch")
        if normalized["site_id"] in seen:
            raise ValueError("Duplicate SAE site")
        seen.add(normalized["site_id"])
        selected.append(normalized)
    if len(data) * (len(model_keys) + len(selected)) > 50000:
        raise ValueError("Target count exceeds safety ceiling")
    receipts = receipts if receipts is not None else {}
    if not isinstance(receipts, dict):
        raise ValueError("Invalid component receipts")
    pairs = []

    def emit(kind: str, dataset: str, model: str, site: dict | None = None) -> None:
        parents = [f"dataset:{dataset}", f"model:{model}"]
        if site:
            parents.append(f"sae:{site['site_id']}")
        states = {key: receipts.get(key, {}).get("status", "NOT_RUN") for key in parents}
        if not set(states.values()) <= _STATES:
            raise ValueError("Invalid component state")
        blocked = any(v in {"FAIL", "BLOCKED"} for v in states.values())
        pair_id = (
            f"model_dataset:{model}:{dataset}" if site is None
            else f"sae_site_dataset:{site['site_id']}:{dataset}"
        )
        pairs.append({
            "id": pair_id,
            "kind": kind,
            "dataset_id": dataset,
            "model_key": model,
            "site_id": site["site_id"] if site else None,
            "status": "BLOCKED" if blocked else "NOT_RUN",
            "eligible": not blocked and all(v == "PASS" for v in states.values()),
            "dependency_states": states,
            "scientific_acceptance": "NOT_COVERED",
        })

    for model in model_keys:
        for dataset in data:
            emit("model_dataset", dataset, model)
    for site in sorted(selected, key=lambda x: x["site_id"]):
        for dataset in data:
            emit("sae_site_dataset", dataset, site["model_key"], site)
    return {
        "schema": "saebase.precheck.compatibility.v1",
        "status": "PLAN_ONLY",
        "models": len(model_keys),
        "datasets": len(data),
        "sites": len(selected),
        "model_dataset_targets": len(model_keys) * len(data),
        "sae_site_dataset_targets": len(selected) * len(data),
        "pairs": pairs,
    }
