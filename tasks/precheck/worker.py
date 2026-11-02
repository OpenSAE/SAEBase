"""Isolated, budgeted Precheck worker. Only this module runs real test operations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tasks.precheck.report import error_result


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_is_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(v) for v in value)
    return True


def _validate_task(task: str, data: Any) -> None:
    """The Task must have run, and its native minimum evidence must exist."""
    if not isinstance(data, dict) or data.get("status") != "executed":
        raise RuntimeError(f"{task} did not return an executed result")
    if data.get("operation") != task or not _is_finite(data):
        raise RuntimeError(f"{task} operation identity or numeric result is invalid")
    if task == "capture":
        if not isinstance(data.get("feature_count"), int) or data["feature_count"] < 1:
            raise RuntimeError("Capture did not return any SAE features")
        if not isinstance(data.get("top_features"), list):
            raise RuntimeError("Capture did not return feature observations")
    elif task == "intervene":
        if not isinstance(data.get("logit_delta_l2"), (float, int)):
            raise RuntimeError("Intervene did not return measured logit differences")
    elif task == "generate":
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Generate did not return a real generation trace")
        for key in ("baseline_generated_token_ids", "steered_generated_token_ids"):
            if not isinstance(result.get(key), list) or not result[key]:
                raise RuntimeError(f"Generate missing real generated tokens: {key}")
        if not isinstance(result.get("steered_steps"), list) or not result["steered_steps"]:
            raise RuntimeError("Generate did not record real intervention steps")
    elif task == "circuits":
        if not data.get("native") or not list(Path("figures").rglob("*.*")):
            raise RuntimeError("Circuit tracer returned no real graph artifacts")
    elif task == "explain":
        hypothesis = data.get("hypothesis")
        if not isinstance(hypothesis, dict) or hypothesis.get("status") != "heldout_tested":
            raise ValueError("Explain PASS requires a real heldout predictor and tested result")
        if not isinstance(hypothesis.get("heldout_score"), (int, float)):
            raise RuntimeError("Explain heldout metric missing")
    elif task == "benchmark":
        metrics = data.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise RuntimeError("Benchmark produced no real metric records")
        if any(not isinstance(m, dict) or m.get("status") != "completed" or int(m.get("n_valid", 0)) < 1 for m in metrics):
            raise ValueError("Benchmark PASS requires every selected metric completed with real valid samples")
    elif task == "train":
        if not data.get("native") or not any(p.is_file() for p in Path("checkpoints").rglob("*")):
            raise RuntimeError("Train produced no native checkpoint")


def check(request: dict[str, Any]) -> dict[str, Any]:
    kind = str(request["kind"])
    try:
        if kind == "sae":
            from tasks.precheck.sae_checks import check_real_sae

            result = check_real_sae(
                model_key=str(request["model_key"]),
                sae_key=str(request["sae_key"]),
                layer=int(request["layer"]),
                prompt=str(request["prompt"]),
                device=str(request["device"]),
            )
        elif kind == "task":
            from tasks.sae_runtime import execute_task

            name = str(request["operation"])
            cfg = OmegaConf.create(request["config"])
            result = execute_task(name, cfg)
            _validate_task(name, result)
            result = {
                "status": "PASS",
                "task": name,
                "scope": "one bounded native Task smoke, not scientific performance acceptance",
                "native_result": result,
            }
        else:
            raise TypeError(f"Invalid worker kind: {kind}")
        return result
    except Exception as exc:
        return error_result(
            exc,
            stage=f"{kind}:{request.get('operation', request.get('sae_key'))}",
            attempted=["isolated real worker invoked"],
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    req = json.loads(args.input.read_text(encoding="utf-8"))
    result = check(req)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    # Parent orchestrator reads the receipt even on a BLOCKED/FAIL.
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
