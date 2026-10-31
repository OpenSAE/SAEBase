"""Diagnose R17 same-seed checkpoint divergence without changing its rubric.

Runs on two actual SAE weight files; accepts safetensors and torch weight-only
files. A test fixture may use synthetic weights, but a real R17 decision needs
three native training logs, frozen model/corpus identity and actual checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, sha256, write_json


def load_weights(path: Path, tensor_name: str) -> tuple[object, str]:
    import torch

    path = path.resolve(strict=True)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = f.keys()
            matched = [k for k in keys if k == tensor_name or k.endswith("." + tensor_name)]
            if len(matched) != 1:
                raise ValueError(
                    f"expected exactly one {tensor_name} in safetensors, found {matched}"
                )
            return f.get_tensor(matched[0]), matched[0]
    if path.suffix not in {".pt", ".pth"}:
        raise ValueError("use a .safetensors or a .pt/.pth weight-only checkpoint")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must contain a tensor state dict")
    if isinstance(data.get("state_dict"), dict):
        data = data["state_dict"]
    matched = [
        k
        for k, v in data.items()
        if isinstance(k, str)
        and (k == tensor_name or k.endswith("." + tensor_name))
        and isinstance(v, torch.Tensor)
    ]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one {tensor_name} tensor, found {matched}")
    return data[matched[0]], matched[0]


def compare_checkpoints(
    path_a: Path,
    path_b: Path,
    *,
    tensor_name: str = "W_dec",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    plan_a: Path | None = None,
    plan_b: Path | None = None,
) -> dict:
    import torch

    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("nonnegative finite tolerances required")
    a, key_a = load_weights(path_a, tensor_name)
    b, key_b = load_weights(path_b, tensor_name)
    if a.shape != b.shape or not a.numel():
        raise ValueError("actual checkpoint tensor shapes differ or are empty")
    a = a.detach().to(dtype=torch.float64, device="cpu")
    b = b.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("checkpoint tensor contains nonfinite values")
    delta = (a - b).abs()
    threshold = atol + rtol * b.abs()
    passed = bool(torch.all(delta <= threshold))
    failed = int((delta > threshold).sum().item())
    max_abs = float(delta.max().item())
    max_ref = float(b.abs().max().item())
    denom = torch.maximum(b.abs(), torch.tensor(atol, dtype=torch.float64))
    max_relative = float((delta / denom).max().item())
    plans = {}
    for label, path in (("a", plan_a), ("b", plan_b)):
        if path:
            obj = read_json(path)
            plans[label] = {
                "sha256": sha256(path),
                "seed": obj.get("seed"),
                "config": obj.get("config"),
                "training_tokens": obj.get("training_tokens"),
            }
    seed_same = (plans["a"].get("seed") == plans["b"].get("seed")) if len(plans) == 2 else None
    return {
        "schema": "saebase.wp21.training_repro.v1",
        "status": "WITHIN_FROZEN_TOLERANCE" if passed else "OUTSIDE_FROZEN_TOLERANCE",
        "tensor_name": tensor_name,
        "tensor_keys": [key_a, key_b],
        "checkpoints": [
            {"file_sha256": sha256(path_a), "path": str(path_a.resolve())},
            {"file_sha256": sha256(path_b), "path": str(path_b.resolve())},
        ],
        "shape": list(a.shape),
        "numel": a.numel(),
        "rtol": rtol,
        "atol": atol,
        "comparison": {
            "max_abs": max_abs,
            "max_relative_over_atol_floored_reference": max_relative,
            "max_reference_abs": max_ref,
            "failing_elements": failed,
            "allclose": passed,
        },
        "plans": plans,
        "same_seed_reported": seed_same,
        "current_diagnostic_runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "note": "This inspects saved checkpoint tensors only. Current process settings do not prove the historical trainer used them. Preserve frozen rtol/atol and compare the actual training logs before attribution.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-a", required=True, type=Path)
    p.add_argument("--weights-b", required=True, type=Path)
    p.add_argument("--tensor", default="W_dec")
    p.add_argument("--plan-a", type=Path)
    p.add_argument("--plan-b", type=Path)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        result = compare_checkpoints(
            a.weights_a,
            a.weights_b,
            tensor_name=a.tensor,
            rtol=a.rtol,
            atol=a.atol,
            plan_a=a.plan_a,
            plan_b=a.plan_b,
        )
        write_json(a.output, result, exclusive=True)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "max_abs": result["comparison"]["max_abs"],
                    "failed_elements": result["comparison"]["failing_elements"],
                    "output": str(a.output),
                }
            )
        )
        return 0 if result["status"] == "WITHIN_FROZEN_TOLERANCE" else 3
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError) as exc:
        print(
            json.dumps({"status": "DIAGNOSTIC_ERROR", "reason": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
