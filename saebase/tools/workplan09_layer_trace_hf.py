#!/usr/bin/env python3
"""Per-layer SAEBase and eager-interp residuals, plus one serial Qwen profile.

Token ids come from the existing GPU-06 reference file. This script does not
write prompt text or token ids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from saebase.api.sae_loading import load_model_and_sae  # noqa: E402

OUT = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021/gpu06")
LAYER_SITE = 7


def _last(tensor: torch.Tensor) -> torch.Tensor:
    hidden = tensor[0] if isinstance(tensor, tuple) else tensor
    return hidden[0, -1].detach().float().cpu()


def _capture_saebase(
    model: torch.nn.Module, token_ids: list[int]
) -> dict[str, torch.Tensor | int | str]:
    layers = model.model.layers
    saved: dict[str, torch.Tensor | int | str] = {
        "n_layers": len(layers),
        "attn": str(getattr(model.config, "_attn_implementation", None)),
    }
    handles = []
    for index, layer in enumerate(layers):

        def hook(
            _module: torch.nn.Module, _inputs: tuple, output: object, index: int = index
        ) -> None:
            saved[f"resid_post.{index}"] = _last(output)

        handles.append(layer.register_forward_hook(hook))

    layer0 = layers[0]

    def pre_hook(_module: torch.nn.Module, inputs: tuple) -> None:
        saved["layer0_in"] = _last(inputs[0])

    def attn_hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
        saved["layer0_attn"] = _last(output)

    def after_attn(_module: torch.nn.Module, inputs: tuple) -> None:
        saved["layer0_after_attn"] = _last(inputs[0])

    def norm_hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
        saved["layer0_input_norm"] = _last(output)

    def mlp_hook(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
        saved["layer0_mlp"] = _last(output)

    handles.append(layer0.register_forward_pre_hook(pre_hook))
    handles.append(layer0.self_attn.register_forward_hook(attn_hook))
    handles.append(layer0.input_layernorm.register_forward_hook(norm_hook))
    handles.append(layer0.post_attention_layernorm.register_forward_pre_hook(after_attn))
    handles.append(layer0.mlp.register_forward_hook(mlp_hook))
    ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        saved["logits"] = logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return saved


def _capture_eager(token_ids: list[int], n_layers: int) -> dict[str, torch.Tensor | int | str]:
    from interp_engine import Address, capture, load_model

    model = load_model(
        "Qwen/Qwen3-1.7B-Base",
        backend="eager",
        dtype="bfloat16",
        device="cuda",
        attn_implementation="sdpa",
    )
    tokens = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    points = [Address("resid_post", index) for index in range(n_layers)]
    points.append(Address("resid_pre", 0))
    cache = capture(model, tokens, points)
    saved: dict[str, torch.Tensor | int | str] = {
        "n_layers": n_layers,
        "attn": "sdpa",
        "torch": torch.__version__,
    }
    for index in range(n_layers):
        activation = cache[Address("resid_post", index)]
        last = activation[0, -1] if activation.ndim == 3 else activation[-1]
        saved[f"resid_post.{index}"] = last.detach().float().cpu()
    pre = cache[Address("resid_pre", 0)]
    saved["resid_pre.0"] = (pre[0, -1] if pre.ndim == 3 else pre[-1]).detach().float().cpu()
    del model, cache
    torch.cuda.empty_cache()
    return saved


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _profile(model: object, sae: object) -> dict[str, float | int]:
    import shutil

    sys.path.insert(0, str(ROOT / "scripts"))
    import workplan09_gpu_pilot as pilot
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions import BatchEngine, ResultWriter

    adapter = ModelAdapter(model)
    rows, pad = pilot._build_token_rows(model)
    samples = pilot._samples_from_token_rows(rows)
    site = ActivationSite(
        LAYER_SITE, f"blocks.{LAYER_SITE}.hook_resid_post", samples[0].patch_position
    )
    snapshot = capture_features(
        adapter,
        sae,
        site=site,
        tokens=adapter.tokens(tokens=torch.tensor([samples[0].tokens], dtype=torch.long)),
    )
    nonzero = [
        int(i) for i in torch.nonzero(snapshot.codes != 0, as_tuple=False).flatten().tolist()
    ]
    conditions = pilot._conditions_for(nonzero, int(sae.cfg.d_sae))
    from saebase.store import sha256_state_dict

    identity = pilot._experiment(
        adapter,
        sae,
        pilot.SITES[1],
        sha256_state_dict(model),
        sha256_state_dict(sae),
        "a" * 64,
    )
    stats = {"model_s": 0.0, "model_n": 0, "encode_s": 0.0, "encode_n": 0}
    original_forward = model._model.forward
    original_encode = sae.encode

    def timed_forward(*args: object, **kwargs: object) -> object:
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        stats["model_s"] += time.perf_counter() - started
        stats["model_n"] += 1
        return output

    def timed_encode(*args: object, **kwargs: object) -> object:
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = original_encode(*args, **kwargs)
        torch.cuda.synchronize()
        stats["encode_s"] += time.perf_counter() - started
        stats["encode_n"] += 1
        return output

    model._model.forward = timed_forward
    sae.encode = timed_encode
    folder = OUT / "profile-serial-once"
    if folder.exists():
        shutil.rmtree(folder)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine = BatchEngine(adapter, sae, identity, pad_token_id=pad)
    with ResultWriter(folder, identity) as writer:
        engine.run(
            samples,
            conditions,
            writer,
            sample_batch=8,
            microbatch=4,
            verify_identity=False,
            tolerance=1e-2,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    model._model.forward = original_forward
    sae.encode = original_encode
    return {
        "n_samples": len(samples),
        "n_conditions": len(conditions),
        "wall_seconds": wall,
        "model_forward_seconds": stats["model_s"],
        "model_forward_count": stats["model_n"],
        "encode_seconds": stats["encode_s"],
        "encode_count": stats["encode_n"],
        "other_seconds": wall - stats["model_s"] - stats["encode_s"],
    }


def main() -> None:
    reference = torch.load(OUT / "reference_last_resid.pt", map_location="cpu", weights_only=False)
    token_ids = [int(x) for x in reference["token_ids"]]
    raw, sae, _meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", LAYER_SITE, device="cuda")
    hf = _capture_saebase(raw._model, token_ids)
    hf["torch"] = torch.__version__
    profile = _profile(raw, sae)
    n_layers = int(hf["n_layers"])
    del raw, sae
    torch.cuda.empty_cache()
    eager = _capture_eager(token_ids, n_layers)
    curve = []
    for index in range(n_layers):
        curve.append(
            {
                "layer": index,
                "eager_vs_hf_max_abs": _max_abs(
                    eager[f"resid_post.{index}"], hf[f"resid_post.{index}"]
                ),
            }
        )
    pre = _max_abs(eager["resid_pre.0"], hf["layer0_in"])
    report = {
        "torch": torch.__version__,
        "hf_attn": hf["attn"],
        "eager_attn": eager["attn"],
        "n_layers": n_layers,
        "eager_vs_hf_resid_pre0_max_abs": pre,
        "eager_vs_hf_by_layer": curve,
        "first_nonzero_layer": next(
            (row["layer"] for row in curve if row["eager_vs_hf_max_abs"] > 0),
            None,
        ),
        "profile_serial_qwen": profile,
    }
    torch.save({key: value for key, value in hf.items() if key != "attn"}, OUT / "hf_layers.pt")
    torch.save(
        {key: value for key, value in eager.items() if key not in {"attn", "torch"}},
        OUT / "eager_layers.pt",
    )
    (OUT / "layer_trace_eager.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
