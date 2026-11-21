"""Capture one clean and two independent five-variant RSS groups on one loaded pair."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from saebase.api import load_registered, capture_features
from saebase.api.neuronpedia import annotate_features
from tasks.capture.operation import neuronpedia_settings
from tasks.capture.rss_mode import measure


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-offload", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((ROOT / "assets/reproduction_inputs/stable_support/files/banana_ab_inputs.json").read_text())
    write_json(out / "inputs.json", inputs)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    pair = load_registered(
        model_key="llama31_8b",
        sae_key="llama31_8b_lxr_8x",
        layer=21,
        device="cpu" if args.prefix_offload else "cuda",
        require_patch=False,
    )
    if args.prefix_offload:
        raw = pair.model.model
        raw.embed.to("cuda")
        if hasattr(raw, "pos_embed"):
            raw.pos_embed.to("cuda")
        for block in raw.blocks[:22]:
            block.to("cuda")
        raw.cfg.device = "cuda"
        pair.sae.to("cuda")
        print("GPU capture prefix 0–21 + embedding + SAE; unused tail remains on CPU", flush=True)
    pair.model.model.eval()
    with torch.inference_mode():
        clean = capture_features(pair.model, pair.sae, site=pair.site, prompt=inputs["clean"])
    results = {}
    ids = set()
    for name, variants in inputs["groups"].items():
        directory = out / name
        directory.mkdir()
        result, codes, delta = measure(
            pair,
            inputs["clean"],
            [r["prefix"] for r in variants],
            ROOT / "tasks/capture/scoring/stable_support.py",
            variants=[r["prompt"] for r in variants],
            clean_snapshot=clean,
        )
        result["group"] = name
        result["shared_clean_capture"] = True
        result["selection_completed_at"] = datetime.now(timezone.utc).isoformat()
        np.savez_compressed(
            directory / "rss_activations.npz", clean=codes[0], variants=codes[1:], delta=delta
        )
        d = delta.astype(np.float64)
        magnitude = np.abs(d)
        params = result["rss_parameters"]
        tau = params["tau"]
        up = (d > tau).sum(0)
        down = (d < -tau).sum(0)
        moved = up + down
        h = magnitude.mean(0)
        recurrence = moved / 5
        consistency = np.divide(np.maximum(up, down), moved, out=np.zeros_like(h), where=moved > 0)
        pool = np.argsort(-h, kind="stable")[: params["M"]]
        support = {r["feature_id"] for r in result["rss_features"]}
        all_pool = []
        for rank, f in enumerate(pool, 1):
            f = int(f)
            reasons = []
            if recurrence[f] < params["rho"]:
                reasons.append("recurrence_below_rho")
            if consistency[f] < params["kappa"]:
                reasons.append("consistency_below_kappa")
            if not reasons and f not in support:
                reasons.append("outside_top_K")
            all_pool.append(
                {
                    "feature_id": f,
                    "pool_rank": rank,
                    "clean_activation": float(codes[0, f]),
                    "variant_activations": codes[1:, f].tolist(),
                    "delta": d[:, f].tolist(),
                    "recurrence": float(recurrence[f]),
                    "consistency": float(consistency[f]),
                    "mean_absolute_change": float(h[f]),
                    "rss_selected": f in support,
                    "in_top_M": True,
                    "exclusion_reasons": [] if f in support else reasons,
                }
            )
        result["all_pool_candidates"] = all_pool
        strict = []
        for v, variant in enumerate(variants):
            rows = [
                r
                for r in all_pool
                if not r["rss_selected"]
                and r["delta"][v] > tau
                and all(abs(x) <= tau for j, x in enumerate(r["delta"]) if j != v)
            ]
            rows.sort(key=lambda r: -r["delta"][v])
            strict.append({"variant": v + 1, "topic": variant["topic"], "features": rows})
        result["strict_context_candidates"] = strict
        result["captured_last_tokens"] = [
            pair.model.model.tokenizer.decode([token_ids[-1]]) for token_ids in result["token_ids"]
        ]
        ids.update(int(f) for f in pool)
        results[name] = result
        write_json(directory / "rss_case.json", result)
        print(
            name,
            "RSS",
            len(support),
            "strict candidates",
            [len(g["features"]) for g in strict],
            flush=True,
        )
    # Both groups have independent numeric selection before this shared lookup.
    info, annotations = annotate_features(
        pair, sorted(ids), neuronpedia_settings(OmegaConf.create({"task": {}}))
    )
    for name, result in results.items():
        for row in (
            result["rss_features"]
            + result["all_pool_candidates"]
            + [r for g in result["context_candidates"] for r in g["features"]]
        ):
            if row["feature_id"] in annotations:
                row["annotation"] = annotations[row["feature_id"]]
        result["neuronpedia"] = info
        write_json(out / name / "rss_case.json", result)
    write_json(out / "annotations.json", {"info": info, "features": annotations})
    OmegaConf.save(
        OmegaConf.create(
            {
                "identity": pair.identity.to_dict(),
                "rss": results["A"]["rss_parameters"],
                "capture_position": -1,
                "model_load_count": 1,
                "prefix_offload": args.prefix_offload,
                "model_dtype": str(next(pair.model.model.parameters()).dtype),
                "clean_capture_count": 1,
                "variant_capture_count": 10,
                "sar_filter": False,
                "standardization": False,
                "inputs": inputs,
                "command": sys.argv,
            }
        ),
        out / "config.yaml",
    )
    write_json(
        out / "run.json",
        {
            "status": "complete",
            "model_load_count": 1,
            "clean_capture_count": 1,
            "variant_capture_count": 10,
            "independent_rss_groups": 2,
            "annotation": info,
        },
    )
    print("COMPLETE", out, flush=True)
    for name, result in results.items():
        print("\nGROUP", name, flush=True)
        for row in result["rss_features"]:
            print(
                "RSS",
                row["feature_id"],
                row["direction"],
                row["delta"],
                "; ".join(e["description"] for e in row["annotation"]["explanations"]),
                flush=True,
            )
        for g in result["strict_context_candidates"]:
            print(
                "CONTEXT",
                g["topic"],
                [
                    (
                        r["feature_id"],
                        round(r["delta"][g["variant"] - 1], 4),
                        "; ".join(e["description"] for e in r["annotation"]["explanations"]),
                    )
                    for r in g["features"]
                ],
                flush=True,
            )


if __name__ == "__main__":
    main()
