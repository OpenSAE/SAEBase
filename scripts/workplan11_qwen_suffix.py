"""RTX 5070 parity and timing for opt-in block-suffix replay.

Public prompts only. Raw series go to experiments/performance/workplan11/, which
is gitignored. The committed report must quote this output rather than invent it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from models.hf_hooks_backend import load_hf_hooks_model
from saebase.api import ActivationSite, ModelAdapter, capture_features
from saebase.interventions.clean_prefix import CLEAN_KV_REASON, CLEAN_KV_STATUS
from saebase.interventions.operators import apply_condition
from saebase.interventions.specs import Condition
from saebase.sae.loading import load_sae_for_profile

PROMPTS = (
    "The capital of France is",
    "A sparse autoencoder reconstructs",
    "Residual stream features can be",
    "Counting tokens is not the same as",
    "The experiment keeps the prompt fixed and",
    "Layer seven is the selected site for",
    "Positive and negative contributions differ when",
    "A short public sentence ends with",
)
LAYER = 7
SITE = f"blocks.{LAYER}.hook_resid_post"
REPEATS = 5
OUT = Path("experiments/performance/workplan11/qwen_l7")


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    handle = load_hf_hooks_model(
        "Qwen/Qwen3-1.7B-Base",
        key="qwen3_1_7b_base",
        family="qwen",
        device=device,
        dtype=torch.bfloat16,
    )
    model = handle.model
    adapter = ModelAdapter(model)
    sae, _meta = load_sae_for_profile("qwen3_1_7b_base_resid_w32k", LAYER, device=device)
    rows = []
    feature_flips = 0
    max_abs = 0.0
    token_rows = []
    for prompt in PROMPTS:
        ids = model.to_tokens(prompt, prepend_bos=False)
        hidden, logits = model.cache_site_with_logits(ids, LAYER)
        reference = model.capture_with_logits(ids, LAYER, [0], readout_offset=0)
        row_error = float((hidden[0, -1].float() - reference[0][0].float()).abs().max())
        logit_error = float((logits[0, -1].float() - reference[1].float()).abs().max())
        max_abs = max(max_abs, row_error, logit_error)
        token_rows.append({"prompt_tokens": int(ids.shape[1]), "row_abs": row_error, "logit_abs": logit_error})
        site = ActivationSite(LAYER, SITE, int(ids.shape[1]) - 1)
        snapshot = capture_features(adapter, sae, site=site, tokens=ids)
        cached_codes = sae.encode(hidden[0, -1].unsqueeze(0))[0]
        fresh_codes = snapshot.codes
        if not torch.equal((cached_codes != 0).cpu(), (fresh_codes != 0).cpu()):
            feature_flips += 1
        edits = {
            "identity": hidden[0, -1],
            "ablate": apply_condition(sae, snapshot, Condition("ablate", "ablate", (0,))).activation,
            "scale": apply_condition(sae, snapshot, Condition("scale", "scale", (0,), 2.0)).activation,
            "clamp": apply_condition(sae, snapshot, Condition("clamp", "clamp", (0,), 1.0)).activation,
            "add": apply_condition(sae, snapshot, Condition("add", "add", (0,), 1.0)).activation,
            "coalition": apply_condition(
                sae, snapshot, Condition("coalition", "coalition", (0, 1), retained_ids=(0,))
            ).activation,
            "reconstruction": apply_condition(
                sae, snapshot, Condition("reconstruction", "reconstruction")
            ).activation,
        }
        early = hidden.clone()
        early[0, 0] = early[0, 0] + 0.25
        cases = {name: (vector, int(ids.shape[1]) - 1, 0) for name, vector in edits.items()}
        cases["early"] = (early[0, 0], 0, 0)
        for name, (vector, position, readout_offset) in cases.items():
            patched = hidden.clone()
            patched[0, position] = vector.to(device=patched.device, dtype=patched.dtype)
            offset = int(ids.shape[1]) - 1 - position
            actual = model.logits_from_site_hidden(ids, LAYER, patched, readout_offset=readout_offset)
            expected = model.run_with_patch(ids, LAYER, {offset: patched[0, position]}, readout_offset=readout_offset)
            error = float((actual.float() - expected.float()).abs().max())
            max_abs = max(max_abs, error)
            rows.append({"prompt_tokens": int(ids.shape[1]), "edit": name, "max_abs": error})

    prepared = []
    for prompt in PROMPTS:
        ids = model.to_tokens(prompt, prepend_bos=False)
        prepared.append((ids, model.cache_site_hidden(ids, LAYER)))

    def time_full() -> float:
        sync()
        t0 = time.perf_counter()
        for ids, hidden in prepared:
            for _edit in range(7):
                model.run_with_patch(ids, LAYER, {0: hidden[0, -1]}, readout_offset=0)
        sync()
        return time.perf_counter() - t0

    def time_suffix() -> float:
        sync()
        t0 = time.perf_counter()
        for ids, hidden in prepared:
            for _edit in range(7):
                model.logits_from_site_hidden(ids, LAYER, hidden, readout_offset=0)
        sync()
        return time.perf_counter() - t0

    time_full()
    time_suffix()
    full = [time_full() for _ in range(REPEATS)]
    suffix = [time_suffix() for _ in range(REPEATS)]
    full_sorted = sorted(full)
    suffix_sorted = sorted(suffix)
    speedup = full_sorted[len(full_sorted) // 2] / suffix_sorted[len(suffix_sorted) // 2]
    payload = {
        "started_utc": started,
        "model": "Qwen/Qwen3-1.7B-Base",
        "sae": "qwen3_1_7b_base_resid_w32k",
        "layer": LAYER,
        "dtype": "bfloat16",
        "prompts": len(PROMPTS),
        "feature_support_flips": feature_flips,
        "max_abs_logits_or_activation": max_abs,
        "token_checks": token_rows,
        "edit_checks": rows,
        "full_seconds": full,
        "suffix_seconds": suffix,
        "median_speedup_full_over_suffix": speedup,
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
        "clean_kv_status": CLEAN_KV_STATUS,
        "clean_kv_reason": CLEAN_KV_REASON,
        "gate_1_15": speedup >= 1.15 and feature_flips == 0 and max_abs <= 1e-2,
    }
    (OUT / "qwen_l7.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "feature_support_flips", "max_abs_logits_or_activation",
        "median_speedup_full_over_suffix", "gate_1_15", "full_seconds", "suffix_seconds",
    )}))


if __name__ == "__main__":
    main()
