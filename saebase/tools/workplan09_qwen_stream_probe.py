#!/usr/bin/env python3
"""Time independent Qwen forwards against one fused batch, and check exact agreement."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from saebase.api.sae_loading import load_model_and_sae  # noqa: E402
from workplan09_gpu_pilot import _build_token_rows, _samples_from_token_rows  # noqa: E402


def one(model, ids, layer, pos):
    found = {}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        found["h"] = hs[0, pos].detach().clone()
        return out

    handle = model._layer_module(layer).register_forward_hook(hook)
    try:
        logits = model._model(input_ids=ids).logits
    finally:
        handle.remove()
    return found["h"], logits[0, pos].detach().clone()


def serial(model, rows, layer):
    hs, ys = [], []
    for ids, pos in rows:
        h, y = one(model, ids, layer, pos)
        hs.append(h)
        ys.append(y)
    return torch.stack(hs), torch.stack(ys)


def streamed(model, rows, layer, n_streams: int):
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    hs = [None] * len(rows)
    ys = [None] * len(rows)
    for start in range(0, len(rows), n_streams):
        chunk = rows[start : start + n_streams]
        for offset, (ids, pos) in enumerate(chunk):
            stream = streams[offset]
            with torch.cuda.stream(stream):
                h, y = one(model, ids, layer, pos)
                hs[start + offset] = h
                ys[start + offset] = y
        for stream in streams[: len(chunk)]:
            stream.synchronize()
    return torch.stack(hs), torch.stack(ys)


def main() -> None:
    model, sae, _meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", 7, device="cuda")
    rows, _pad = _build_token_rows(model)
    samples = [
        s
        for s in _samples_from_token_rows(rows)
        if s.record.sample_id.endswith("-final") or s.record.prompt_group_id == "wp09-eq"
    ]
    work = [
        (torch.tensor([s.tokens], dtype=torch.long, device="cuda"), s.patch_position)
        for s in samples
    ]
    for _ in range(1):
        serial(model, work, 7)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    ref_h, ref_y = serial(model, work, 7)
    end.record()
    end.synchronize()
    serial_s = start.elapsed_time(end) / 1000

    results = [{"mode": "serial", "seconds": serial_s, "n": len(work)}]
    for n_streams in (2, 4):
        for _ in range(1):
            streamed(model, work, 7, n_streams)
        torch.cuda.synchronize()
        start.record()
        h, y = streamed(model, work, 7, n_streams)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000
        results.append(
            {
                "mode": f"streams_{n_streams}",
                "seconds": seconds,
                "ratio_vs_serial": serial_s / seconds if seconds else None,
                "act_max": float((h.float() - ref_h.float()).abs().max().item()),
                "logit_max": float((y.float() - ref_y.float()).abs().max().item()),
            }
        )
        del sae
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
