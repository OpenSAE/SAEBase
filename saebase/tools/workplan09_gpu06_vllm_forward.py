#!/usr/bin/env python3
"""One real vLLM forward on the RTX 5070.

CUDA_HOME must be a CUDA >= 12.9 toolkit before FlashInfer is imported.
The process records whether generation returns tokens. Prompt text is not written.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

CU = Path("/workspace/venvs/wp09-interp/lib/python3.12/site-packages/nvidia/cu13")
os.environ["CUDA_HOME"] = str(CU)
os.environ["PATH"] = str(CU / "bin") + os.pathsep + os.environ.get("PATH", "")
lib = str(CU / "lib")
os.environ["LD_LIBRARY_PATH"] = lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

PROMPT = "The capital of France is"
OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/gpu06")


def attempt(backend: str | None) -> dict[str, object]:
    import torch
    from vllm import LLM, SamplingParams

    row: dict[str, object] = {
        "cuda_home": os.environ["CUDA_HOME"],
        "flashinfer_sampler": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "torch": torch.__version__,
        "capability": list(torch.cuda.get_device_capability()),
        "attention_backend": backend or "auto",
        "enforce_eager": True,
    }
    kwargs: dict[str, object] = {}
    if backend is not None:
        kwargs["attention_config"] = {"backend": backend}
    llm = LLM(
        model="Qwen/Qwen3-1.7B-Base",
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.55,
        **kwargs,
    )
    outputs = llm.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=4, seed=7),
    )
    generated = [int(x) for x in outputs[0].outputs[0].token_ids]
    row["status"] = "PASS" if len(generated) == 4 else "FAIL"
    row["generated_token_count"] = len(generated)
    row["generated_token_ids"] = generated
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    attempts = []
    chosen = None
    for backend in ("TRITON_ATTN", "FLASH_ATTN", None):
        try:
            row = attempt(backend)
        except Exception as exc:  # noqa: BLE001
            row = {
                "attention_backend": backend or "auto",
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}"[:800],
                "traceback": traceback.format_exc()[-1500:],
            }
        attempts.append({k: v for k, v in row.items() if k != "traceback"})
        print(json.dumps(attempts[-1]), flush=True)
        if row.get("status") == "PASS":
            chosen = row
            break
    payload = {"attempts": attempts, "status": "PASS" if chosen else "BLOCKED"}
    if chosen:
        payload["forward"] = {k: v for k, v in chosen.items() if k != "traceback"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vllm_forward.json").write_text(json.dumps(payload, indent=2) + "\n")
    if chosen is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
