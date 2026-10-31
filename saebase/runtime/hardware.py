"""Environment capability inventory for any SAEBase GPU experiment.

This is detection and provenance, not an authority to change scientific
configuration. Unknown quantities remain null; an unavailable GPU is not a
simulated CPU pass for GPU tasks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


def _available_host_ram_bytes() -> int | None:
    # Linux memory headroom is obtained from /proc rather than total physical.
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def detect(*, probe_gpu: bool = False, cache_path: Path | None = None) -> dict[str, Any]:
    import torch

    target = cache_path or Path.cwd()
    free_disk = shutil.disk_usage(target).free
    info: dict[str, Any] = {
        "schema": "saebase.environment.v1",
        "os": platform.system(), "machine": platform.machine(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
        "gpu": [], "available_host_ram_bytes": _available_host_ram_bytes(),
        "disk_free_bytes": free_disk,
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                props = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                entry: dict[str, Any] = {
                    "index": index, "name": props.name,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_vram_bytes": int(total), "free_vram_bytes": int(free),
                    "supports_bf16": bool(torch.cuda.is_bf16_supported()),
                    "cuda_kernel_ok": None,
                }
                if probe_gpu:
                    try:
                        # Force an actual kernel so an unsupported CUDA wheel
                        # does not register as a usable GPU installation.
                        with torch.no_grad():
                            result = torch.ones((8,), device=f"cuda:{index}").sum().item()
                        entry["cuda_kernel_ok"] = result == 8
                    except Exception as exc:
                        entry["cuda_kernel_ok"] = False
                        entry["kernel_error_type"] = type(exc).__name__
                info["gpu"].append(entry)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect actual SAEBase runtime resources")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-probe", action="store_true")
    args = parser.parse_args(argv)
    report = detect(probe_gpu=args.device_probe)
    data = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)
    return 0 if all(x.get("cuda_kernel_ok") is not False for x in report["gpu"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
