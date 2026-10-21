"""RTX 5070 benchmarks for *implemented* cached SAE steering backends.

Modes:
 hf-reference: old clean-full-prefix SAE ablation, no KV (reproduction baseline)
 hf-live: one prefill + one HF KV forward/token, dynamic SAE re-encode
 vllm: interp-engine vLLM cached generation + fixed decoder feature direction
 vllm-static: static-tap graph variant, if RTX 5070 VRAM permits

These are DIFFERENT scientific protocols. Compare speeds, but never claim that
fixed decoder direction or persistent edited KV reproduces old SAEBase logits.
Save raw token IDs only under the private WP11_RUN directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch

MODEL = "Qwen/Qwen3-1.7B-Base"
SAE_KEY = "qwen3_1_7b_base_resid_w32k"
LAYER = 7
PROMPTS = (
    "The capital of France is",
    "A sparse autoencoder reconstructs",
    "The next step of this computation is",
)
OUT = Path(os.environ.get("WP11_RUN", "/workspace/experiments/performance/workplan11/final_5070"))


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def git_sha() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def reference_stepper(model, sae, token_ids, max_tokens: int, feature_id: int):
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.interventions.operators import apply_condition
    from saebase.interventions.specs import Condition

    adapter = ModelAdapter(model)
    condition = Condition("ablate", "ablate", (feature_id,))
    current = token_ids.detach().clone()
    result = []
    for _step in range(max_tokens):
        site = ActivationSite(LAYER, f"blocks.{LAYER}.hook_resid_post", int(current.shape[1]) - 1)
        snapshot = capture_features(adapter, sae, site=site, tokens=current)
        patch = apply_condition(sae, snapshot, condition)
        logits = adapter.patch_logits(
            tokens=current, site=site, activation=patch.activation, readout_position=-1
        )
        nxt = int(torch.argmax(logits.float()).item())
        result.append(nxt)
        current = torch.cat(
            (current, torch.tensor([[nxt]], dtype=torch.long, device=current.device)), dim=1
        )
    return {
        "ids": tuple(result),
        "forwards": 2 * max_tokens,
        "protocol": "reference_clean_prefix_stepwise_v1",
    }


def load_hf_pair():
    from saebase.sae.loading import load_model_and_sae

    return load_model_and_sae(SAE_KEY, LAYER, device="cuda")


def hf_generator(mode, model, sae, ids, max_tokens, feature_id):
    if mode == "hf-reference":
        return reference_stepper(model, sae, ids, max_tokens, feature_id)
    from saebase.interventions.hf_cached_steering import generate_live_sae_cached
    from saebase.interventions.specs import Condition

    trace = generate_live_sae_cached(
        model,
        sae,
        ids,
        layer=LAYER,
        condition=Condition("ablate", "ablate", (feature_id,)),
        max_new_tokens=max_tokens,
        max_context_tokens=1024,
    )
    return {
        "ids": trace.generated_token_ids,
        "forwards": trace.model_forwards,
        "protocol": trace.protocol,
    }


def vllm_generator(model, sae, ids, max_tokens, feature_id, strength):
    from saebase.vllm_backend.fast_steering import generate_decoder_direction_cached

    trace = generate_decoder_direction_cached(
        model,
        sae,
        ids,
        layer=LAYER,
        feature_id=feature_id,
        strength=strength,
        max_new_tokens=max_tokens,
        temperature=0.0,
        seed=0,
        stop_at_eos=False,
    )
    return {"ids": trace.generated_token_ids, "forwards": None, "protocol": trace.protocol}


def select_feature_id(model, sae, ids):
    from saebase.api import ActivationSite, ModelAdapter, capture_features

    adapter = ModelAdapter(model)
    snap = capture_features(
        adapter, sae, site=ActivationSite(LAYER, f"blocks.{LAYER}.hook_resid_post", -1), tokens=ids
    )
    active = torch.nonzero(snap.codes != 0).flatten().tolist()
    if not active:
        raise RuntimeError("prompt has no active feature at the chosen Qwen/SAE site")
    return int(active[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("hf-reference", "hf-live", "vllm", "vllm-static"),
    )
    parser.add_argument("--lengths", default="8,32,128")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--feature-id", type=int, default=None)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--utilization", type=float, default=0.55)
    args = parser.parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    if not lengths or any(x < 1 or x > 256 for x in lengths):
        parser.error("lengths must contain integers from 1 through 256")
    if args.repeats < 5:
        parser.error("at least 5 timed repetitions required")
    if args.mode in ("vllm", "vllm-static") and args.feature_id is None:
        parser.error("vLLM runs require --feature-id from a verified SAE capture")
    if not torch.cuda.is_available():
        parser.error("RTX CUDA GPU required")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = None
    engine = None
    if args.mode.startswith("hf-"):
        raw, sae, _ = load_hf_pair()
        tokenizer = raw
    else:
        from saebase.sae.loading import load_sae_for_profile
        from saebase.vllm_backend.sae_interventions import InterpVLLMSAE

        sae, _ = load_sae_for_profile(SAE_KEY, LAYER, device="cuda")
        utilization = (
            max(0.70, args.utilization) if args.mode == "vllm-static" else args.utilization
        )
        engine = InterpVLLMSAE.load(
            MODEL,
            sae,
            layer=LAYER,
            backend=args.mode,
            gpu_memory_utilization=utilization,
            max_model_len=1024,
        )
        tokenizer = engine.model

    try:
        report = {
            "git_sha": git_sha(),
            "mode": args.mode,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "repeats": args.repeats,
            "site": f"{MODEL}:L{LAYER}:{SAE_KEY}",
            "results": [],
            "generation_controls": [],
            "comparability": (
                "old clean-prefix scientific protocol"
                if args.mode == "hf-reference"
                else "persistent live SAE edit KV"
                if args.mode == "hf-live"
                else "persistent fixed SAE decoder-direction, not live ablation"
            ),
        }
        for prompt_index, prompt in enumerate(PROMPTS):
            ids = tokenizer.to_tokens(prompt, prepend_bos=False)
            if isinstance(ids, torch.Tensor):
                ids = ids.reshape(1, -1)
            else:
                raise RuntimeError("model.to_tokens must return a tensor")
            selected = args.feature_id
            if selected is None:
                if raw is not None:
                    selected = select_feature_id(raw, sae, ids)
                else:
                    raise RuntimeError("provide an explicit, source-verified SAE feature ID")
            if raw is None:
                # A real baseline -> steered -> baseline request sequence. The
                # worker owns steering per request, and no test may infer a
                # working hook solely from the presence of generated text.
                from saebase.vllm_backend.fast_steering import generate_decoder_direction_cached

                probe = generate_decoder_direction_cached(
                    engine.model,
                    sae,
                    tuple(int(x) for x in ids[0].tolist()),
                    layer=LAYER,
                    feature_id=selected,
                    strength=args.strength,
                    max_new_tokens=min(8, max(lengths)),
                    temperature=0.0,
                    seed=0,
                    compare_unsteered=True,
                    stop_at_eos=False,
                )
                before = tuple(probe.baseline_token_ids or ())
                after = tuple(probe.generated_token_ids)
                report["generation_controls"].append(
                    {
                        "prompt_index": prompt_index,
                        "unsteered_token_ids_private": list(before),
                        "steered_token_ids_private": list(after),
                        "token_disagreement": sum(a != b for a, b in zip(before, after)),
                        "note": "Token agreement alone does not establish that taps were bypassed",
                    }
                )
            for length in lengths:
                series = []
                token_ids = None
                for rep in range(args.repeats + 1):
                    sync()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    if raw is not None:
                        trace = hf_generator(args.mode, raw, sae, ids, length, selected)
                    else:
                        trace = vllm_generator(
                            engine.model,
                            sae,
                            tuple(int(x) for x in ids[0].tolist()),
                            length,
                            selected,
                            args.strength,
                        )
                    sync()
                    elapsed = time.perf_counter() - start
                    if rep:
                        series.append(
                            {
                                "seconds": elapsed,
                                "tokens": len(trace["ids"]),
                                "forwards": trace["forwards"],
                                "allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                "reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            }
                        )
                        token_ids = list(trace["ids"])
                sorted_times = sorted(x["seconds"] for x in series)
                median = sorted_times[len(sorted_times) // 2]
                total = len(token_ids or ())
                report["results"].append(
                    {
                        "prompt_index": prompt_index,
                        "original_token_count": int(ids.shape[1]),
                        "feature_id": selected,
                        "max_new_tokens": length,
                        "protocol": trace["protocol"],
                        "median_seconds": median,
                        "new_tokens_per_second": total / median if median else None,
                        "repetitions": series,
                        "new_token_ids_private": token_ids,
                    }
                )
                name = f"fast_steering_{args.mode.replace('-', '_')}.json"
                (OUT / name).write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "site": report["site"],
                    "summary": [
                        {
                            "prompt_index": row["prompt_index"],
                            "tokens": row["max_new_tokens"],
                            "median_seconds": row["median_seconds"],
                            "tokens_per_second": row["new_tokens_per_second"],
                        }
                        for row in report["results"]
                    ],
                    "private_report": str(
                        OUT / f"fast_steering_{args.mode.replace('-', '_')}.json"
                    ),
                },
                indent=2,
            )
        )
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
