#!/usr/bin/env python3
"""Local generation-time SAE steering. Records continuations and per-step shifts.

This is greedy or sampled token generation, not a teacher-forced logit card.
Prompt text is read from a local file and is not printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from saebase.api import ActivationSite, load_registered
from saebase.interventions.specs import Condition, MeasurementSample
from saebase.interventions.steering import GenerationSpec, compare_steered_generation
from saebase.store.identity import SampleRecord, TokenSelection

SPECIAL = {"<bos>", "<eos>", "<pad>", "<unk>", "<start_of_turn>", "<end_of_turn>"}


def _sample(model, prompt: str, sample_id: str) -> tuple[MeasurementSample, list[str]]:
    token_ids = model.tokens(prompt=prompt)
    ids = tuple(int(value) for value in token_ids[0].tolist())
    tokenizer = model.model.tokenizer
    texts = tokenizer.convert_ids_to_tokens(list(ids))
    record = SampleRecord(
        sample_id=sample_id,
        token_ids=ids,
        selection=TokenSelection(mode="last_valid"),
        prompt_group_id=sample_id,
        split="discovery",
    )
    return MeasurementSample(record=record), [str(text) for text in texts]


def _active_feature(
    model, sae, site_hook: str, layer: int, sample: MeasurementSample, texts: list[str]
) -> tuple[int, int, float]:
    ids = torch.tensor([sample.tokens], dtype=torch.long)
    for position in range(len(sample.tokens) - 1, -1, -1):
        if texts[position] in SPECIAL:
            continue
        site = ActivationSite(layer, site_hook, position)
        from saebase.api import capture_features

        snapshot = capture_features(model, sae, site=site, tokens=ids)
        active = torch.nonzero(snapshot.codes, as_tuple=True)[0]
        if active.numel() == 0:
            continue
        values = snapshot.codes[active].abs()
        choice = int(active[int(torch.argmax(values).item())].item())
        return position, choice, float(snapshot.codes[choice].detach().float().item())
    raise RuntimeError("No active SAE feature at a content token")


def _run_one(pair, prompt: str, sample_id: str, max_new_tokens: int, scale: float) -> dict:
    sample, texts = _sample(pair.model, prompt, sample_id)
    position, feature_id, activation = _active_feature(
        pair.model, pair.sae, pair.identity.hook_name, pair.identity.layer, sample, texts
    )
    site = ActivationSite(pair.identity.layer, pair.identity.hook_name, position)
    spec = GenerationSpec(max_new_tokens=max_new_tokens, temperature=0.0, seed=0)
    conditions = [
        Condition(name="noop", operation="identity"),
        Condition(name="ablate", operation="ablate", feature_ids=(feature_id,)),
        Condition(name="scale", operation="scale", feature_ids=(feature_id,), value=scale),
    ]
    decode = pair.model.model.tokenizer.decode
    runs = []
    for condition in conditions:
        result = compare_steered_generation(
            pair.model, pair.sae, site=site, sample=sample, condition=condition, spec=spec
        )
        runs.append(
            {
                "condition": condition.to_dict(),
                "same_continuation": result["same_continuation"],
                "first_different_generated_index": result["first_different_generated_index"],
                "baseline_generated_token_ids": result["baseline_generated_token_ids"],
                "edited_generated_token_ids": result["steered_generated_token_ids"],
                "baseline_continuation": decode(
                    result["baseline_generated_token_ids"], skip_special_tokens=False
                ),
                "edited_continuation": decode(
                    result["steered_generated_token_ids"], skip_special_tokens=False
                ),
                "noop_equal_baseline": result["same_continuation"]
                if condition.operation == "identity"
                else None,
                "steps": result.get("steered_steps", []),
            }
        )
    return {
        "sample_id": sample_id,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "site_id": f"{pair.identity.model_key}:{pair.identity.sae_key}:L{pair.identity.layer}",
        "hook_name": pair.identity.hook_name,
        "feature_position": position,
        "feature_id": feature_id,
        "feature_activation": activation,
        "protocol": "greedy_full_prefix_recompute",
        "max_new_tokens": max_new_tokens,
        "generated_text": True,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, action="append", required=True)
    parser.add_argument("--sae-key", default="gemma270m_scope_resid")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing steering result")
    if not 16 <= args.max_new_tokens <= 32:
        parser.error("WP15 generation length must be 16–32 tokens")
    pair = load_registered(sae_key=args.sae_key, layer=args.layer, device=args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for index, path in enumerate(args.prompt_file):
        prompt = path.read_text(encoding="utf-8")
        if not prompt.strip():
            parser.error("Prompt file is empty: " + str(path))
        rows.append(
            _run_one(pair, prompt, f"WP15-G2-{index + 1:02d}", args.max_new_tokens, args.scale)
        )
    document = {
        "schema": "opensae.wp15.generation_steering.v1",
        "sae_key": args.sae_key,
        "layer": args.layer,
        "scale": args.scale,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "prompts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompts": [
                    {
                        "sample_id": row["sample_id"],
                        "feature_id": row["feature_id"],
                        "conditions": [
                            {
                                "name": run["condition"]["name"],
                                "same_continuation": run["same_continuation"],
                                "max_abs_logit_delta": max(
                                    step["max_abs_logit_delta"] for step in run["steps"]
                                ),
                            }
                            for run in row["runs"]
                        ],
                    }
                    for row in rows
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
