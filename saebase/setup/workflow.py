"""Environment inventory, representative selection and measured time estimates.

Setup does not claim a real model pass until an approved smoke has actually
loaded that model. All speed estimates come from timed, observed pilot rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def representative_sites(sites: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Select one existing site per model first; report any uncovered model."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Representative-site budget must be positive")
    if not isinstance(sites, list) or not sites:
        raise ValueError("Registered study sites are required")
    seen: set[str] = set()
    chosen = []
    remaining = []
    keys = []
    for site in sites:
        model = site.get("model_key")
        sae = site.get("sae_key")
        layer = site.get("layer")
        if not isinstance(model, str) or not isinstance(sae, str) or type(layer) is not int:
            raise ValueError("Each site needs an actual model, SAE and integer layer")
        key = (model, sae, layer)
        if key in keys:
            raise ValueError("Duplicate site in scientific configuration")
        keys.append(key)
        if model not in seen and len(chosen) < limit:
            chosen.append(dict(site))
            seen.add(model)
        else:
            remaining.append(dict(site))
    for site in remaining:
        if len(chosen) >= limit:
            break
        chosen.append(site)
    missing_models = sorted({s["model_key"] for s in sites} - set(seen))
    return {
        "sites": chosen,
        "uncovered_models": missing_models,
        "unverified_sites": len(sites) - len(chosen),
        "coverage": "representative_only",
    }


def estimate_runtime(pilot_rows: list[dict[str, Any]], total_samples: int) -> dict[str, Any]:
    """Extrapolate observed throughput, never invent timings when no pilot ran."""
    if type(total_samples) is not int or total_samples < 0:
        raise ValueError("Total samples must be a nonnegative integer")
    times = []
    samples = 0
    for row in pilot_rows:
        count = row.get("samples")
        duration = row.get("elapsed_seconds")
        if (
            type(count) is not int
            or count < 1
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Invalid measured pilot timing")
        samples += count
        times.append(float(duration))
    if not times:
        return {"status": "NOT_RUN", "estimated_seconds": None, "measured_samples": 0}
    elapsed = sum(times)
    return {
        "status": "MEASURED_ESTIMATE",
        "measured_samples": samples,
        "measured_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "estimated_seconds": total_samples * elapsed / samples,
        "uncertainty": "Approximation excludes unmeasured model startup, network and retries",
    }


def setup_report(
    *,
    dataset_ids: list[str],
    sites: list[dict[str, Any]],
    representative_limit: int,
    require_cuda: bool,
    cache_path: Path,
    real_gpu_probe: bool = False,
    pilot_rows: list[dict[str, Any]] | None = None,
    total_samples: int = 0,
) -> dict[str, Any]:
    """Read actual hardware with the existing Runtime; never download assets."""
    from saebase.runtime.hardware import detect

    if (
        not isinstance(dataset_ids, list)
        or not dataset_ids
        or any(not isinstance(x, str) or not x for x in dataset_ids)
        or len(set(dataset_ids)) != len(dataset_ids)
    ):
        raise ValueError("Invalid or duplicate Dataset identities")
    if real_gpu_probe and not require_cuda:
        raise ValueError("Real GPU probe requires a declared CUDA task")
    choice = representative_sites(sites, representative_limit)
    hardware = detect(probe_gpu=real_gpu_probe, cache_path=cache_path)
    blocked = require_cuda and not hardware.get("cuda_available")
    return {
        "schema": "saebase.setup.v1",
        "status": "BLOCKED" if blocked else "PREFLIGHT_ONLY",
        "datasets": dataset_ids,
        "representatives": choice,
        "hardware": hardware,
        "runtime_estimate": estimate_runtime(pilot_rows or [], total_samples),
        "real_model_test": "NOT_RUN",
        "remaining_scope": "Every untested model/SAE/dataset/task still requires its own receipt",
    }
