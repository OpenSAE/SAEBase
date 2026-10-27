#!/usr/bin/env python3
"""Compare single-example and masked HF forwards on Qwen, stage by stage.

Prints lengths and numeric errors only. Prompt text stays in the private pilot file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from models.hf_hooks_backend import HFHooksModel  # noqa: E402
from saebase.api import ModelAdapter  # noqa: E402
from saebase.sae.loading import load_model_and_sae  # noqa: E402
from workplan09_gpu_pilot import _build_token_rows, _samples_from_token_rows  # noqa: E402


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    d = (a.float() - b.float()).abs()
    denom = float(b.float().norm().item())
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "rel_l2": float(d.norm().item() / denom) if denom else 0.0,
        "allclose_bf16": bool(torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)),
    }


def topk_ids(sae, h: torch.Tensor, k: int = 50) -> set[int]:
    z = sae.encode(h.float().unsqueeze(0))[0]
    nz = torch.nonzero(z != 0, as_tuple=False).flatten()
    if nz.numel() == 0:
        values, index = torch.topk(z, k=min(k, z.numel()))
        del values
        return set(int(i) for i in index.tolist())
    return set(int(i) for i in nz.tolist())


def single_resid_logits(
    wrapper: HFHooksModel, ids: torch.Tensor, layer: int, pos: int, readout: int
):
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["h"] = hs[0, pos].detach().clone()
        return out

    handle = wrapper._layer_module(layer).register_forward_hook(hook)
    try:
        out = wrapper._model(input_ids=ids)
    finally:
        handle.remove()
    return captured["h"], out.logits[0, readout].detach().clone()


def main() -> None:
    raw, sae, meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", 7, device="cuda")
    wrapper = raw if isinstance(raw, HFHooksModel) else raw
    adapter = ModelAdapter(wrapper)
    rows, pad = _build_token_rows(wrapper)
    samples = _samples_from_token_rows(rows)
    layer = 7
    report: dict[str, object] = {
        "backend": adapter.backend,
        "dtype": str(next(wrapper.parameters()).dtype).replace("torch.", ""),
        "loaded": meta.get("loaded_model_name"),
        "pad_distinct": True,
        "groups": [],
    }

    def run_group(name: str, chosen: list) -> None:
        lengths = [s.valid_length for s in chosen]
        width = max(lengths)
        ids = torch.full((len(chosen), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        positions = []
        readouts = []
        for i, sample in enumerate(chosen):
            row = torch.tensor(sample.tokens, dtype=torch.long, device="cuda")
            ids[i, : sample.valid_length] = row
            mask[i, : sample.valid_length] = 1
            positions.append(sample.patch_position)
            readouts.append(
                sample.readout_index if sample.readout_index >= 0 else sample.valid_length - 1
            )
        singles_h = []
        singles_y = []
        for sample, pos, read in zip(chosen, positions, readouts):
            one = torch.tensor([sample.tokens], dtype=torch.long, device="cuda")
            h, y = single_resid_logits(wrapper, one, layer, pos, read)
            singles_h.append(h)
            singles_y.append(y)
        single_h = torch.stack(singles_h)
        single_y = torch.stack(singles_y)

        variants = {}
        # Current masked path.
        bh, by = wrapper.run_batch(
            ids, layer=layer, positions=positions, readout_positions=readouts, attention_mask=mask
        )
        variants["masked_run_batch"] = (bh, by)
        # Same tokens, no attention_mask argument, only when there is no padding.
        if len(set(lengths)) == 1 and lengths[0] == width:
            captured: dict[str, torch.Tensor] = {}

            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, tuple) else out
                rows_i = torch.arange(hs.shape[0], device=hs.device)
                captured["h"] = (
                    hs[rows_i, torch.tensor(positions, device=hs.device)].detach().clone()
                )
                return out

            handle = wrapper._layer_module(layer).register_forward_hook(hook)
            try:
                out = wrapper._model(input_ids=ids)
            finally:
                handle.remove()
            nomask_y = out.logits[
                torch.arange(len(chosen), device="cuda"), torch.tensor(readouts, device="cuda")
            ]
            variants["equal_no_mask"] = (captured["h"], nomask_y.detach().clone())

        group = {"name": name, "n": len(chosen), "lengths": lengths, "stages": []}
        for label, (h, y) in variants.items():
            feature_flips = 0
            for i in range(len(chosen)):
                left = topk_ids(sae, single_h[i])
                right = topk_ids(sae, h[i])
                if left != right:
                    feature_flips += 1
            group["stages"].append(
                {
                    "path": label,
                    "activation": stats(h, single_h),
                    "logits": stats(y, single_y),
                    "feature_id_flips": feature_flips,
                }
            )
        report["groups"].append(group)
        print(json.dumps(group), flush=True)

    finals = [
        s
        for s in samples
        if s.record.sample_id.endswith("-final") and s.record.prompt_group_id != "wp09-eq"
    ]
    equal = [s for s in samples if s.record.prompt_group_id == "wp09-eq"]
    run_group("equal_length", equal[:4] if equal else finals[:2])
    run_group("mixed_length", finals[:6])
    # Repeated identical rows: isolates batch-size effects from padding.
    if finals:
        run_group("repeated_equal", [finals[0], finals[0], finals[0], finals[0]])
    out = Path(
        os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwen_stage_diff.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
