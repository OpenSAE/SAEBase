#!/usr/bin/env python3
"""Capture every vLLM residual and compare it with the SAEBase layer trace.

CUDA_HOME is set before vLLM imports. Configurations are selected with
--config and are not enabled as a SAEBase backend. Prompt text and token ids
are not written.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

CU = Path("/workspace/venvs/wp09-interp/lib/python3.12/site-packages/nvidia/cu13")
os.environ["CUDA_HOME"] = str(CU)
os.environ["PATH"] = str(CU / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = str(CU / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import torch  # noqa: E402

OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/gpu06")
ATOL = 1e-2
RTOL = 1e-2


def _last(activation: torch.Tensor) -> torch.Tensor:
    return activation[0, -1] if activation.ndim == 3 else activation[-1]


def _err(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    delta = (left.float() - right.float()).abs()
    denom = float(right.float().norm().item())
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "rel_l2": float(delta.norm().item() / denom) if denom else 0.0,
        "allclose": bool(torch.allclose(left.float(), right.float(), rtol=RTOL, atol=ATOL)),
    }


def _capture(token_ids: list[int], n_layers: int, extra: dict) -> dict[str, torch.Tensor]:
    from interp_engine import Address, capture, load_model

    model = load_model(
        "Qwen/Qwen3-1.7B-Base",
        backend="vllm",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.55,
        extra_vllm_kwargs={"enable_prefix_caching": False, **extra},
    )
    tokens = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    points = [Address("resid_post", index) for index in range(n_layers)]
    optional = [Address(name, 0) for name in ("resid_pre", "resid_mid", "attn_out", "mlp_out")]
    try:
        cache = capture(model, tokens, points + optional)
        used = points + optional
    except Exception:
        cache = capture(model, tokens, points)
        used = points
    saved = {}
    for point in used:
        saved[f"{point.name}.{point.layer}"] = _last(cache[point]).detach().float().cpu()
    del cache, model
    gc.collect()
    torch.cuda.empty_cache()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=("triton", "triton_native_norm", "flash_native_norm"), required=True
    )
    args = parser.parse_args()
    reference = torch.load(OUT / "reference_last_resid.pt", map_location="cpu", weights_only=False)
    hf = torch.load(OUT / "hf_layers.pt", map_location="cpu", weights_only=False)
    token_ids = [int(x) for x in reference["token_ids"]]
    n_layers = int(hf["n_layers"])
    configs = {
        "triton": {"attention_config": {"backend": "TRITON_ATTN"}},
        "triton_native_norm": {
            "attention_config": {"backend": "TRITON_ATTN"},
            "kernel_config": {
                "ir_op_priority": {"rms_norm": ["native"], "fused_add_rms_norm": ["native"]}
            },
        },
        "flash_native_norm": {
            "attention_config": {"backend": "FLASH_ATTN"},
            "kernel_config": {
                "ir_op_priority": {"rms_norm": ["native"], "fused_add_rms_norm": ["native"]}
            },
        },
    }
    captured = _capture(token_ids, n_layers, configs[args.config])
    curve = []
    for index in range(n_layers):
        key = f"resid_post.{index}"
        curve.append({"layer": index, **_err(captured[key], hf[key])})
    first = next((row for row in curve if row["max_abs"] > 0), None)
    components = {}
    for name, hf_key in (
        ("resid_pre.0", "layer0_in"),
        ("resid_mid.0", "layer0_after_attn"),
        ("attn_out.0", "layer0_attn"),
        ("mlp_out.0", "layer0_mlp"),
    ):
        if name in captured and hf_key in hf:
            components[name] = _err(captured[name], hf[hf_key])
    report = {
        "config": args.config,
        "torch": torch.__version__,
        "n_layers": n_layers,
        "components_vs_hf": components,
        "by_layer": curve,
        "first_nonzero": first,
        "layer7": next(row for row in curve if row["layer"] == 7),
        "worst": max(curve, key=lambda row: row["max_abs"]),
        "allclose_all_layers": all(row["allclose"] for row in curve),
    }
    torch.save(captured, OUT / f"vllm_{args.config}_layers.pt")
    (OUT / f"vllm_{args.config}_layers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "config": args.config,
                "first_nonzero": first,
                "layer7": report["layer7"],
                "worst": report["worst"],
                "components_vs_hf": components,
                "allclose_all_layers": report["allclose_all_layers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
