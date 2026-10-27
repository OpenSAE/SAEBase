#!/usr/bin/env python3
"""See which attention/matmul settings make a Qwen batch match batch-size 1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from saebase.sae.loading import load_model_and_sae  # noqa: E402
from workplan09_gpu_pilot import _build_token_rows, _samples_from_token_rows  # noqa: E402


def capture(model, ids, layer, positions):
    found = {}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        rows = torch.arange(hs.shape[0], device=hs.device)
        found["h"] = hs[rows, torch.tensor(positions, device=hs.device)].detach().clone()
        found["y"] = None
        return out

    handle = model._layer_module(layer).register_forward_hook(hook)
    try:
        logits = model._model(input_ids=ids).logits
    finally:
        handle.remove()
    read = torch.tensor(positions, device=ids.device)
    return found["h"], logits[torch.arange(ids.shape[0], device=ids.device), read].detach().clone()


def flips(sae, left, right) -> int:
    n = 0
    for i in range(left.shape[0]):
        a = set(
            torch.nonzero(sae.encode(left[i].float().unsqueeze(0))[0] != 0, as_tuple=False)
            .flatten()
            .tolist()
        )
        b = set(
            torch.nonzero(sae.encode(right[i].float().unsqueeze(0))[0] != 0, as_tuple=False)
            .flatten()
            .tolist()
        )
        n += int(a != b)
    return n


def main() -> None:
    model, sae, _meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", 7, device="cuda")
    rows, _pad = _build_token_rows(model)
    samples = _samples_from_token_rows(rows)
    chosen = [s for s in samples if s.record.prompt_group_id == "wp09-eq"][:3]
    ids = torch.tensor([s.tokens for s in chosen], dtype=torch.long, device="cuda")
    positions = [s.patch_position for s in chosen]
    singles = []
    single_y = []
    for i in range(ids.shape[0]):
        h, y = capture(model, ids[i : i + 1], 7, [positions[i]])
        singles.append(h[0])
        single_y.append(y[0])
    ref_h = torch.stack(singles)
    ref_y = torch.stack(single_y)

    def trial(name, fn):
        h, y = fn()
        dh = (h.float() - ref_h.float()).abs()
        dy = (y.float() - ref_y.float()).abs()
        row = {
            "name": name,
            "act_max": float(dh.max().item()),
            "act_rel": float(dh.norm().item() / ref_h.float().norm().item()),
            "logit_max": float(dy.max().item()),
            "flips": flips(sae, ref_h, h),
            "act_close": bool(torch.allclose(h.float(), ref_h.float(), rtol=1e-2, atol=1e-2)),
            "logit_close": bool(torch.allclose(y.float(), ref_y.float(), rtol=1e-2, atol=1e-2)),
        }
        print(json.dumps(row), flush=True)

    trial("default_batch", lambda: capture(model, ids, 7, positions))

    def no_reduced():
        prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            return capture(model, ids, 7, positions)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev

    trial("bf16_full_reduction", no_reduced)

    def math_sdp():
        with sdpa_kernel(SDPBackend.MATH):
            return capture(model, ids, 7, positions)

    trial("sdpa_math", math_sdp)

    def flash_sdp():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return capture(model, ids, 7, positions)

    trial("sdpa_flash", flash_sdp)

    # Reference itself under math SDP, then the batch under the same kernel.
    with sdpa_kernel(SDPBackend.MATH):
        math_singles = []
        for i in range(ids.shape[0]):
            h, _y = capture(model, ids[i : i + 1], 7, [positions[i]])
            math_singles.append(h[0])
        math_ref = torch.stack(math_singles)
        math_batch, _y = capture(model, ids, 7, positions)
    dh = (math_batch.float() - math_ref.float()).abs()
    print(
        json.dumps(
            {
                "name": "math_batch_vs_math_single",
                "act_max": float(dh.max().item()),
                "flips": flips(sae, math_ref, math_batch),
                "act_close": bool(
                    torch.allclose(math_batch.float(), math_ref.float(), rtol=1e-2, atol=1e-2)
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
