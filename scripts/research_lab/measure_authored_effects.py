#!/usr/bin/env python3
"""Measure next-word lists for a few project-authored prompts.

This is a separate micro-study from the six saved score-only pairs. It uses
models already on disk, at most two sites, and does not rewrite those pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

RUN_CACHE = Path(
    "/workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/cache/hf"
)
os.environ.setdefault("HF_HOME", str(RUN_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(RUN_CACHE / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROTOCOL = "authored_prompt.paired_topk.v1"
RUN_ID = "effects-authored-20261021"
TOP_K = 5
PROMPTS = (
    {"id": "capital", "category": "place name", "text": "The capital of France is"},
    {"id": "sum", "category": "arithmetic", "text": "Two plus two equals"},
    {"id": "opposite", "category": "opposite", "text": "The opposite of hot is"},
    {"id": "animal", "category": "animal name", "text": "A baby cat is called a"},
)
SITES = (
    {
        "site_id": "pythia410m:pythia410m_mlp_topk:L12",
        "sae": "pythia410m_mlp_topk",
        "layer": 12,
        "hook": "blocks.12.hook_mlp_out",
        "model_label": "Pythia 410M",
    },
    {
        "site_id": "gemma1b:gemma1b_scope_resid:L13",
        "sae": "gemma1b_scope_resid",
        "layer": 13,
        "hook": "blocks.13.hook_resid_post",
        "model_label": "Gemma 1B",
    },
)


def visible_label(text: str) -> str:
    if text == " ":
        return "space"
    if text == "\n":
        return "new line"
    if text.strip() == "":
        return "blank"
    return text


def readout(logits, tokenizer) -> dict:
    import torch

    values = logits.float()
    logsumexp = float(torch.logsumexp(values, dim=-1).item())
    probabilities = torch.softmax(values, dim=-1)
    scores, indices = torch.topk(probabilities, TOP_K)
    rows = []
    for probability, index in zip(scores.tolist(), indices.tolist(), strict=True):
        token_id = int(index)
        token = tokenizer.decode([token_id])
        logit = float(values[token_id].item())
        rebuilt = math.exp(logit - logsumexp)
        if not math.isclose(probability, rebuilt, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError("top-token probability does not match a full-vocabulary softmax")
        rows.append(
            {
                "label": visible_label(token),
                "logit": logit,
                "probability": float(probability),
                "token": token,
                "token_id": token_id,
            }
        )
    return {"logsumexp": logsumexp, "top_tokens": rows}


def measure_site(spec: dict, prompts: tuple[dict, ...]) -> list[dict]:
    import torch

    from saebase.api import ActivationSite, ModelAdapter
    from saebase.api.features import capture_features
    from saebase.interventions.operators import apply_condition
    from saebase.interventions.specs import Condition
    from saebase.sae.loading import load_model_and_sae
    from saebase.store.identity import sha256_state_dict

    model, sae, _meta = load_model_and_sae(spec["sae"], spec["layer"], device="cuda")
    adapter = ModelAdapter(model)
    tokenizer = model.tokenizer
    model_sha = sha256_state_dict(model)
    sae_sha = sha256_state_dict(sae)
    site = ActivationSite(spec["layer"], spec["hook"], position=-1)
    cases = []
    try:
        for prompt in prompts:
            tokens = model.to_tokens(prompt["text"])
            adapter.assert_identity_patch(tokens=tokens, site=site)
            snapshot = capture_features(adapter, sae, site=site, tokens=tokens)
            active = torch.nonzero(snapshot.codes, as_tuple=False).flatten()
            if active.numel() == 0:
                raise RuntimeError(f"no active feature for {prompt['id']} on {spec['site_id']}")
            feature = int(torch.argmax(snapshot.codes.abs()).item())
            baseline = adapter.forward_logits(tokens=tokens, readout_position=site.position)
            before = readout(baseline, tokenizer)
            edits = (
                ("ablate", "Remove this feature", Condition("ablate", "ablate", (feature,))),
                ("scale", "Cut its activity in half", Condition("scale", "scale", (feature,), 0.5)),
                ("add", "Add 1 to its activity", Condition("add", "add", (feature,), 1.0)),
            )
            operators = []
            for name, label, condition in edits:
                patch = apply_condition(sae, snapshot, condition)
                edited = adapter.patch_logits(
                    tokens=tokens,
                    site=site,
                    activation=patch.activation,
                    readout_position=site.position,
                )
                distance = float(torch.linalg.vector_norm((edited.float() - baseline.float())).item())
                if not math.isfinite(distance) or distance < 0:
                    raise RuntimeError(f"non-finite logit distance for {name}")
                after = readout(edited, tokenizer)
                operators.append(
                    {
                        "edited_top_tokens": after["top_tokens"],
                        "edited_logsumexp": after["logsumexp"],
                        "feature_ids": [feature],
                        "label": label,
                        "logit_l2": distance,
                        "operation": name,
                        "operation_parameters": {"value": condition.value} if name != "ablate" else {},
                    }
                )
            primary = operators[0]
            cases.append(
                {
                    "condition": "ablate",
                    "evidence_url": "https://github.com/OpenSAE/SAEBase/blob/work-plan-06-real-evidence/scripts/research_lab/measure_authored_effects.py",
                    "feature_id": feature,
                    "id": f"authored:{spec['site_id']}:{prompt['id']}",
                    "identity_patch_passed": True,
                    "layer": spec["layer"],
                    "measured_delta_l2": primary["logit_l2"],
                    "measured_fields": ["top_tokens", "logit_l2"],
                    "model": spec["model_label"],
                    "model_key": spec["site_id"].split(":")[0],
                    "model_sha256": model_sha,
                    "note": "Next-word chances come from a full-vocabulary softmax at the last token. A larger score shift does not mean a better answer.",
                    "operators": operators,
                    "original_logsumexp": before["logsumexp"],
                    "original_top_tokens": before["top_tokens"],
                    "edited_top_tokens": primary["edited_top_tokens"],
                    "prompt": prompt["text"],
                    "protocol_id": PROTOCOL,
                    "public_text_reviewed": True,
                    "rights_reviewed": False,
                    "run_id": RUN_ID,
                    "sae": spec["sae"],
                    "sae_sha256": sae_sha,
                    "sample_count": 1,
                    "site_id": spec["site_id"],
                    "source_category": prompt["category"],
                    "source_relative_path": "scripts/research_lab/measure_authored_effects.py",
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "source_type": "real_checkpoint",
                    "text_stored": True,
                    "title": f"{spec['model_label']} · {prompt['category']}",
                }
            )
    finally:
        del model, sae
        torch.cuda.empty_cache()
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site", action="append", choices=["pythia", "gemma"])
    args = parser.parse_args()
    selected = {"pythia": SITES[0], "gemma": SITES[1]}
    wanted = [selected[name] for name in (args.site or ["pythia", "gemma"])]
    if len(PROMPTS) * len(wanted) > 12:
        raise SystemExit("prompt-site budget is 12")
    cases = []
    for spec in wanted:
        cases.extend(measure_site(spec, PROMPTS))
    payload = {
        "cases": cases,
        "prompt_site_pairs": len(cases),
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "sites": [spec["site_id"] for spec in wanted],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
