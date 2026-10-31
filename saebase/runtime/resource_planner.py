"""Hardware-capability resource selection for registered model tasks.

This selects an actual compatible CUDA device, not a substitute model,
precision or task. It does not predict model+SAE peak memory without a real
Pilot. When safe execution cannot be established, the worker remains blocked.
"""
from __future__ import annotations

from typing import Any


def select_cuda_device(
    hardware: dict[str, Any], *, minimum_free_bytes: int = 1 << 30,
) -> dict[str, Any]:
    if not isinstance(minimum_free_bytes, int) or minimum_free_bytes < 0:
        raise ValueError("Invalid physical GPU free-memory requirement")
    if not hardware.get("cuda_available"):
        raise RuntimeError("BLOCKED_ENV_NO_CUDA: real GPU task requires a CUDA device")

    candidates = []
    for entry in hardware.get("gpu", []):
        if entry.get("cuda_kernel_ok") is False:
            continue
        if not isinstance(entry.get("index"), int) or entry["index"] < 0:
            continue
        free = entry.get("free_vram_bytes")
        total = entry.get("total_vram_bytes")
        if not isinstance(free, int) or not isinstance(total, int):
            continue
        if free < minimum_free_bytes or free > total:
            continue
        candidates.append(entry)

    if not candidates:
        raise RuntimeError(
            "BLOCKED_RESOURCE: no CUDA device passed the actual free-VRAM "
            "and executable-kernel checks"
        )
    selected = sorted(
        candidates, key=lambda item: (-item["free_vram_bytes"], item["index"])
    )[0]
    return {
        "schema": "saebase.runtime.device_selection.v1",
        "device": f"cuda:{selected['index']}",
        "strategy": "highest_actual_free_vram_on_supported_device",
        "device_index": selected["index"],
        "free_vram_bytes_at_selection": selected["free_vram_bytes"],
        "total_vram_bytes": selected["total_vram_bytes"],
        "compute_capability": selected.get("compute_capability"),
        "cuda_kernel_ok": selected.get("cuda_kernel_ok"),
        "minimum_free_bytes": minimum_free_bytes,
        "precision_change": False,
        "model_or_dataset_substitution": False,
        "offload_requested": False,
        "remaining_uncertainty": "peak_model_and_sae_memory_must_be_measured_in_real_pilot",
    }
