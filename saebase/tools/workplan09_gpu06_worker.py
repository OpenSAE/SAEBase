#!/usr/bin/env python3
"""Run inside the interp-engine environment. Records what actually executes."""

from __future__ import annotations

import importlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

RUN = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021")
PROMPT = "The capital of France is"
MODEL = "Qwen/Qwen3-1.7B-Base"
LAYER = 7


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    found = {}
    for name in ("interp-engine", "vllm", "torch", "transformers"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not-installed"
    return found


def summarize(logits: torch.Tensor) -> dict[str, Any]:
    vector = logits.detach().float().cpu().reshape(-1)
    return {
        "argmax": int(torch.argmax(vector).item()),
        "max_logit": float(vector.max().item()),
        "shape": list(logits.shape),
    }


def try_backend(backend: str) -> dict[str, Any]:
    from interp_engine import Address, capture, load_model

    if backend.startswith("vllm"):
        try:
            importlib.import_module("vllm")
        except ModuleNotFoundError:
            return {
                "backend": backend,
                "model": MODEL,
                "utc": utc_now(),
                "packages": package_versions(),
                "status": "BLOCKED",
                "capture_status": "BLOCKED",
                "patch_status": "BLOCKED",
                "generate_status": "BLOCKED",
                "error": "vllm is not installed in /workspace/venvs/wp09-interp",
            }

    row: dict[str, Any] = {
        "backend": backend,
        "model": MODEL,
        "utc": utc_now(),
        "packages": package_versions(),
    }
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        row["free_vram_bytes_before"] = int(free_b)
        row["total_vram_bytes"] = int(total_b)
        model = load_model(MODEL, backend=backend, dtype="bfloat16", device="cuda")
        tokens = model.to_tokens(PROMPT)
        point = Address("resid_post", LAYER)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            cache = capture(model, tokens, [point])
        except Exception as exc:  # noqa: BLE001
            row["capture_status"] = "BLOCKED"
            row["capture_error"] = f"{type(exc).__name__}: {exc}"[:800]
            row["error"] = row["capture_error"]
            cache = None
        if cache is not None:
            torch.cuda.synchronize()
            activation = cache[point]
            row["capture_seconds"] = time.perf_counter() - started
            row["activation_shape"] = list(activation.shape)
            row["activation_finite"] = bool(torch.isfinite(activation).all().item())
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            row["allocated_bytes_after_capture"] = int(torch.cuda.memory_allocated())
            row["capture_status"] = "PASS" if row["activation_finite"] else "FAIL"
        else:
            activation = None
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        from interp_engine import SteerMethod, SteerSpec, decode_residuals, generate_stream, steer

        if activation is None:
            row["vector_ablate_status"] = "BLOCKED"
            row["vector_ablate_error"] = "capture returned no residual"
            direction = None
            last = None
        else:
            flat = activation.detach()
            if flat.ndim == 3:
                last = flat[0, -1]
            elif flat.ndim == 2:
                last = flat[-1]
            else:
                last = flat.reshape(-1)
            logits = decode_residuals(model, last.float().unsqueeze(0))
            row["clean_logit"] = summarize(logits)
            row["activation_dtype"] = str(activation.dtype).replace("torch.", "")
            if backend == "eager":
                torch.save(
                    {
                        "last_resid": last.detach().float().cpu(),
                        "token_ids": [int(x) for x in tokens.reshape(-1).tolist()],
                        "dtype": row["activation_dtype"],
                    },
                    RUN / "gpu06" / "eager_last_resid.pt",
                )
            direction = torch.zeros(last.numel(), device=last.device, dtype=last.dtype)
            direction[0] = 1
        spec = (
            [
                SteerSpec(
                    vector=direction,
                    layer=LAYER,
                    coeff=1.0,
                    method=SteerMethod.ABLATE,
                    point="resid_post",
                )
            ]
            if direction is not None
            else None
        )
        try:
            if spec is None:
                raise RuntimeError(row["vector_ablate_error"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with steer(model, spec, prompt_token_ids=tokens, generated=False):
                steered_cache = capture(model, tokens, [point])
            torch.cuda.synchronize()
            steered = steered_cache[point]
            if steered.ndim == 3:
                steered_last = steered[0, -1]
            elif steered.ndim == 2:
                steered_last = steered[-1]
            else:
                steered_last = steered.reshape(-1)
            delta = (steered_last.float() - last.float()).abs()
            row["vector_ablate_seconds"] = time.perf_counter() - started
            row["vector_ablate_max_abs"] = float(delta.max().item())
            row["vector_ablate_status"] = (
                "PASS" if torch.isfinite(steered_last).all().item() else "FAIL"
            )
        except Exception as exc:  # noqa: BLE001
            row["vector_ablate_status"] = "BLOCKED"
            row["vector_ablate_error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["patch_status"] = "BLOCKED"
        row["patch_reason"] = (
            "interp-engine steer() applies a residual vector. This PR has no SAE feature "
            "adapter, so the call does not certify SAE intervention equivalence."
        )
        row["public_intervention_name"] = "steer"
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            steps = list(
                generate_stream(
                    model, tokens, max_tokens=4, temperature=0.0, seed=7, stop_at_eos=False
                )
            )
            torch.cuda.synchronize()
            row["generate_seconds"] = time.perf_counter() - started
            row["generate_status"] = "PASS" if len(steps) == 4 else "FAIL"
            row["generated_token_count"] = len(steps)
            row["generate_has_logits"] = all(step.logits is not None for step in steps)
        except Exception as exc:  # noqa: BLE001
            row["generate_status"] = "BLOCKED"
            row["generate_reason"] = f"{type(exc).__name__}: {exc}"[:500]
        if row.get("capture_status") == "PASS":
            row["status"] = "PASS"
        elif row.get("generate_status") == "PASS":
            row["status"] = "BLOCKED"
        else:
            row["status"] = "FAIL" if row.get("capture_status") == "FAIL" else "BLOCKED"
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:  # noqa: BLE001
        row["status"] = "BLOCKED"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)[:1500]
        row["traceback"] = traceback.format_exc()[-2500:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def main() -> None:
    selected = os.environ.get("WP09_BACKENDS", "eager,vllm,vllm-generate")
    backends = [item.strip() for item in selected.split(",") if item.strip()]
    attempt_path = RUN / "gpu06" / "backend_attempt.json"
    previous: dict[str, dict[str, Any]] = {}
    if attempt_path.exists():
        loaded = json.loads(attempt_path.read_text(encoding="utf-8"))
        for item in loaded.get("rows", []):
            previous[item["backend"]] = item
    rows = []
    for backend in backends:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = try_backend(backend)
        rows.append(row)
        append_jsonl(
            RUN / "backend_parity.jsonl",
            {
                "utc": utc_now(),
                "backend": backend,
                "status": row.get("status"),
                "capture_status": row.get("capture_status"),
                "patch_status": row.get("patch_status"),
                "generate_status": row.get("generate_status"),
                "vector_ablate_status": row.get("vector_ablate_status"),
                "activation_shape": row.get("activation_shape"),
                "error": row.get("error"),
                "model": MODEL,
                "layer": LAYER,
                "packages": row.get("packages"),
            },
        )
        print(
            json.dumps(
                {
                    "backend": backend,
                    "status": row.get("status"),
                    "error": row.get("error", "")[:180],
                }
            )
        )
        if row.get("status") == "BLOCKED" and backend == "eager":
            break
        previous[backend] = row
    ordered = [previous[name] for name in ("eager", "vllm", "vllm-generate") if name in previous]
    write_json(RUN / "gpu06" / "backend_attempt.json", {"utc": utc_now(), "rows": ordered})
    any_pass = any(row.get("status") == "PASS" for row in rows)
    raise SystemExit(0 if any_pass else 2)


if __name__ == "__main__":
    main()
