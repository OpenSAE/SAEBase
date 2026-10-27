#!/usr/bin/env python3
"""vLLM capture, identity, and generation against the SAEBase residual.

CUDA_HOME is set before vLLM imports. Attention is Triton and the sampler is
native, because that is the combination that completes a forward on this GPU.
Prompt text is not written.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

CU = Path("/workspace/venvs/wp09-interp/lib/python3.12/site-packages/nvidia/cu13")
os.environ["CUDA_HOME"] = str(CU)
os.environ["PATH"] = str(CU / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = str(CU / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

ROOT = Path("/workspace/SAEBase-wp09-candidate")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

PROMPT = "The capital of France is"
LAYER = 7
OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/gpu06")
ATOL = 1e-2
RTOL = 1e-2


def err(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    d = (a.float() - b.float()).abs()
    denom = float(b.float().norm().item())
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "rel_l2": float(d.norm().item() / denom) if denom else 0.0,
        "allclose": bool(torch.allclose(a.float(), b.float(), rtol=RTOL, atol=ATOL)),
    }


def main() -> None:
    reference = torch.load(OUT / "reference_last_resid.pt", map_location="cpu", weights_only=False)
    report: dict[str, object] = {
        "cuda_home": os.environ["CUDA_HOME"],
        "torch": torch.__version__,
        "attention_backend": "TRITON_ATTN",
    }
    delta_path = OUT / "sae_delta.pt"
    delta = None
    if delta_path.exists():
        saved = torch.load(delta_path, map_location="cpu", weights_only=False)
        delta = saved["delta"]
        report["ablation_feature_id"] = int(saved["feature_id"])
    from interp_engine import (
        AddSpec,
        Address,
        SteeringSpec,
        capture,
        generate_stream,
        load_model,
        steer,
    )

    model = load_model(
        "Qwen/Qwen3-1.7B-Base",
        backend="vllm",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.55,
        extra_vllm_kwargs={
            "attention_config": {"backend": "TRITON_ATTN"},
            "enable_prefix_caching": False,
        },
    )
    tokens = model.to_tokens(PROMPT)
    ids = [int(x) for x in tokens.reshape(-1).tolist()]
    report["token_ids_equal"] = ids == reference["token_ids"]
    report["prefix_caching"] = False
    point = Address("resid_post", LAYER)
    cache = capture(model, tokens, [point])
    activation = cache[point]
    last = activation[0, -1] if activation.ndim == 3 else activation[-1]
    report["capture"] = err(last, reference["last_resid"].to(last.device))
    report["capture_shape"] = list(activation.shape)
    zeros = torch.zeros_like(last)
    ident_spec = SteeringSpec.at(f"resid_post.{LAYER}", AddSpec(vector=zeros, scale=0.0))
    with steer(model, ident_spec, prompt_token_ids=tokens, generated=False):
        ident = capture(model, tokens, [point])
    ident_last = ident[point][0, -1] if ident[point].ndim == 3 else ident[point][-1]
    report["identity_vs_own_capture"] = err(ident_last, last)
    if delta is not None:
        try:
            exclude = list(range(len(ids) - 1))
            ablate_spec = SteeringSpec.at(
                f"resid_post.{LAYER}",
                AddSpec(vector=delta.to(dtype=last.dtype), scale=1.0),
            )
            with steer(
                model,
                ablate_spec,
                prompt_token_ids=tokens,
                position_mask=exclude,
                generated=False,
            ):
                ablated = capture(model, tokens, [point])
                ab_last = ablated[point][0, -1] if ablated[point].ndim == 3 else ablated[point][-1]
                target = reference["last_resid"].float() + delta.float()
                report["ablation_residual"] = err(ab_last, target.to(ab_last.device))
                steered_steps = list(
                    generate_stream(
                        model, tokens, max_tokens=4, temperature=0.0, seed=7, stop_at_eos=False
                    )
                )
            report["stepwise_token_ids"] = [int(step.token_id) for step in steered_steps]
            report["stepwise_matches_saebase_greedy"] = report["stepwise_token_ids"] == [
                12095,
                13,
                12095,
                374,
            ]
        except Exception as exc:  # noqa: BLE001
            report["ablation_error"] = f"{type(exc).__name__}: {exc}"[:400]
    steps = list(
        generate_stream(model, tokens, max_tokens=4, temperature=0.0, seed=7, stop_at_eos=False)
    )
    generated = [int(step.token_id) for step in steps]
    report["generation"] = {
        "token_ids": generated,
        "matches_saebase_greedy": generated == [12095, 13, 12095, 374],
        "n": len(generated),
    }
    report["status"] = "measured"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vllm_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
