#!/usr/bin/env python3
"""Align eager interp-engine with SAEBase on one torch, then compare edits.

Residual readouts, token positions, and bf16 logits are compared directly.
decode_residuals is not used: it is a final-norm lens, not the model logits.
Prompt text is not written.
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

from saebase.api import ActivationSite, ModelAdapter, capture_features  # noqa: E402
from saebase.interventions import Condition, apply_condition  # noqa: E402
from saebase.sae.loading import load_model_and_sae  # noqa: E402

PROMPT = "The capital of France is"
LAYER = 7
MODEL = "Qwen/Qwen3-1.7B-Base"
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
    raw, sae, meta = load_model_and_sae("qwen3_1_7b_base_resid_w32k", LAYER, device="cuda")
    adapter = ModelAdapter(raw)
    ids = [int(x) for x in raw.to_tokens(PROMPT)[0].tolist()]
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    position = len(ids) - 1
    site = ActivationSite(LAYER, meta["hook_name"], position)
    clean = adapter.forward_logits(tokens=tokens, readout_position=-1)
    snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
    identity = adapter.patch_logits(
        tokens=tokens, site=site, activation=snapshot.activation, readout_position=-1
    )
    nonzero = [
        int(i) for i in torch.nonzero(snapshot.codes != 0, as_tuple=False).flatten().tolist()
    ]
    feature = nonzero[0] if nonzero else 0
    patched = apply_condition(sae, snapshot, Condition("ablate", "ablate", (feature,)))
    ablated = adapter.patch_logits(
        tokens=tokens, site=site, activation=patched.activation, readout_position=-1
    )
    greedy = []
    current = list(ids)
    for _ in range(4):
        step_tokens = torch.tensor([current], dtype=torch.long, device="cuda")
        logits = adapter.forward_logits(tokens=step_tokens, readout_position=-1)
        nxt = int(torch.argmax(logits).item())
        greedy.append(nxt)
        current.append(nxt)
    saved = {
        "token_ids": ids,
        "hook": meta["hook_name"],
        "dtype": str(snapshot.activation.dtype).replace("torch.", ""),
        "resid": snapshot.activation.detach().float().cpu(),
        "clean_logits": clean.detach().float().cpu(),
        "identity_logits": identity.detach().float().cpu(),
        "ablated_logits": ablated.detach().float().cpu(),
        "delta": (patched.activation - snapshot.activation).detach().float().cpu(),
        "feature_id": feature,
        "greedy": greedy,
        "attn": str(getattr(raw._model.config, "_attn_implementation", None)),
    }
    del raw, sae, adapter, snapshot, patched
    torch.cuda.empty_cache()

    from interp_engine import (
        Address,
        SteerMethod,
        SteerSpec,
        capture,
        generate_stream,
        load_model,
        steer,
    )

    model = load_model(
        MODEL,
        backend="eager",
        dtype="bfloat16",
        device="cuda",
        attn_implementation="sdpa",
    )
    itokens = model.to_tokens(PROMPT)
    interp_ids = [int(x) for x in itokens.reshape(-1).tolist()]
    point = Address("resid_post", LAYER)
    cache = capture(model, itokens, [point])
    activation = cache[point]
    last = activation[0, -1]
    logits = cache.output.logits[0, -1]
    report: dict[str, object] = {
        "torch": torch.__version__,
        "saebase_hook": saved["hook"],
        "interp_point": "resid_post",
        "layer": LAYER,
        "token_ids_equal": interp_ids == saved["token_ids"],
        "n_tokens": len(interp_ids),
        "saebase_dtype": saved["dtype"],
        "interp_dtype": str(last.dtype).replace("torch.", ""),
        "saebase_attn": saved["attn"],
        "interp_attn": str(getattr(model.hf_model.config, "_attn_implementation", None)),
        "residual": err(last, saved["resid"].to(last.device)),
        "clean_logits": err(logits, saved["clean_logits"].to(logits.device)),
        "feature_id": saved["feature_id"],
    }
    zeros = torch.zeros_like(last)
    spec = SteerSpec(
        vector=zeros, layer=LAYER, coeff=0.0, method=SteerMethod.ADDITIVE, point="resid_post"
    )
    with steer(model, [spec], prompt_token_ids=itokens, generated=False):
        ident_cache = capture(model, itokens, [point])
    report["identity_patch_logits"] = err(
        ident_cache.output.logits[0, -1], saved["clean_logits"].to(logits.device)
    )
    report["identity_patch_residual"] = err(ident_cache[point][0, -1], last)
    exclude = list(range(len(interp_ids) - 1))
    delta = saved["delta"].to(device=last.device, dtype=last.dtype)
    ablate_spec = SteerSpec(
        vector=delta, layer=LAYER, coeff=1.0, method=SteerMethod.ADDITIVE, point="resid_post"
    )
    with steer(
        model,
        [ablate_spec],
        prompt_token_ids=itokens,
        position_mask=exclude,
        generated=False,
    ):
        ablate_cache = capture(model, itokens, [point])
    report["sae_ablation_logits"] = err(
        ablate_cache.output.logits[0, -1], saved["ablated_logits"].to(logits.device)
    )
    report["sae_ablation_residual"] = err(
        ablate_cache[point][0, -1],
        (saved["resid"] + saved["delta"]).to(last.device),
    )
    steps = list(
        generate_stream(model, itokens, max_tokens=4, temperature=0.0, seed=7, stop_at_eos=False)
    )
    generated = [int(step.token_id) for step in steps]
    report["generation"] = {
        "interp_token_ids": generated,
        "saebase_token_ids": saved["greedy"],
        "equal": generated == saved["greedy"],
        "n": len(steps),
    }
    # Stepwise: recompute the full prefix at each new token and add the same
    # SAE delta only at the original prompt's last position. This is the
    # full-prefix oracle, not a KV-cached steer.
    steered_ids = []
    running = list(interp_ids)
    for _ in range(4):
        step_tokens = torch.tensor([running], dtype=torch.long, device="cuda")
        with steer(
            model,
            [ablate_spec],
            prompt_token_ids=torch.tensor([interp_ids], device="cuda"),
            position_mask=exclude,
            generated=False,
        ):
            step_cache = capture(model, step_tokens, [point])
        nxt = int(torch.argmax(step_cache.output.logits[0, -1]).item())
        steered_ids.append(nxt)
        running.append(nxt)
    report["stepwise_fixed_delta_tokens"] = steered_ids
    report["argmax"] = {
        "saebase_clean": int(torch.argmax(saved["clean_logits"]).item()),
        "interp_clean": int(torch.argmax(logits).item()),
        "saebase_identity": int(torch.argmax(saved["identity_logits"]).item()),
        "interp_identity": int(torch.argmax(ident_cache.output.logits[0, -1]).item()),
        "saebase_ablated": int(torch.argmax(saved["ablated_logits"]).item()),
        "interp_ablated": int(torch.argmax(ablate_cache.output.logits[0, -1]).item()),
    }
    report["stepwise_note"] = (
        "interp adds the prompt-final SAE delta at that fixed position on each "
        "full prefix; it does not recode the SAE"
    )
    out = Path(
        os.environ.get("WP09_RUN", "/workspace/experiments/performance/workplan09/rtx5070-20261021")
    )
    (out / "gpu06").mkdir(parents=True, exist_ok=True)
    (out / "gpu06" / "eager_align_same_torch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
