"""Environment probe for transformer_lens / sae_lens / CUDA."""

from __future__ import annotations

import sys


def check_sae_env() -> int:
    print("[sae] python:", sys.executable)
    missing = []
    for mod, pip in (
        ("torch", "torch"),
        ("transformer_lens", "transformer_lens"),
        ("sae_lens", "sae-lens"),
        ("huggingface_hub", "huggingface_hub"),
    ):
        try:
            __import__(mod)
            print(f"[sae] ok import {mod}")
        except ImportError:
            missing.append(pip)
            print(f"[sae] missing {mod}")
    try:
        import torch

        print(f"[sae] cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[sae] device={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[sae] torch probe failed: {exc}")
        return 1
    if missing:
        print(f"[sae] install: pip install {' '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(check_sae_env())
