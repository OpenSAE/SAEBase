"""Measure the fixed unrelated-prefix banana bread case with strict RSS-pool candidates."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
import saebase
from saebase.api import load_registered
from saebase.api.neuronpedia import annotate_features
from tasks.capture.operation import neuronpedia_settings
from tasks.capture.rss_mode import measure


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "assets/reproduction_inputs/stable_support/files/semantic_preservation_inputs.json"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(args.inputs.read_text())
    source = ROOT / "tasks/capture/scoring/stable_support.py"
    source_bytes = source.read_bytes()
    provenance = {
        **json.loads((ROOT / "assets/reproduction_inputs/stable_support/files/source.json").read_text()),
        "local_selector_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "local_selector_git_blob": hashlib.sha1(
            f"blob {len(source_bytes)}\0".encode() + source_bytes
        ).hexdigest(),
    }
    provenance.pop("inputs", None)
    write_json(
        out / "run.json",
        {"status": "loading", "command": sys.argv, "inputs": inputs, "selector_source": provenance},
    )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    pair = load_registered(
        model_key="llama31_8b",
        sae_key="llama31_8b_lxr_8x",
        layer=21,
        device="cuda",
        require_patch=False,
    )
    result, codes, delta = measure(
        pair,
        inputs["clean"],
        [r["prefix"] for r in inputs["variants"]],
        source,
        variants=[r["prompt"] for r in inputs["variants"]],
    )
    d = delta.astype(np.float64)
    mag = np.abs(d)
    tau = result["rss_parameters"]["tau"]
    h = mag.mean(0)
    moved = mag > tau
    recurrence = moved.mean(0)
    pool = np.argsort(-h, kind="stable")[: result["rss_parameters"]["M"]]
    selected = {r["feature_id"] for r in result["rss_features"]}
    groups = []
    all_ids = set(selected)
    for v, variant in enumerate(inputs["variants"]):
        ids = [
            int(f)
            for f in pool
            if f not in selected and d[v, f] > tau and np.all(np.delete(mag[:, f], v) <= tau)
        ]
        ids.sort(key=lambda f: (-d[v, f], f))
        rows = []
        for rank, f in enumerate(ids, 1):
            rows.append(
                {
                    "feature_id": f,
                    "candidate_rank": rank,
                    "clean_activation": float(codes[0, f]),
                    "variant_activations": codes[1:, f].tolist(),
                    "delta": d[:, f].tolist(),
                    "recurrence": float(recurrence[f]),
                    "consistency": 1.0,
                    "mean_absolute_change": float(h[f]),
                    "in_top_M": True,
                    "rss_selected": False,
                    "exclusion_reasons": ["recurrence_below_rho"],
                    "selection_stage": "numeric_before_annotation",
                }
            )
        all_ids.update(ids)
        groups.append(
            {
                "variant": v + 1,
                "topic": variant["topic"],
                "prefix": variant["prefix"],
                "features": rows,
            }
        )
    result["context_candidates"] = groups
    result["candidate_rule"] = (
        "From original top-M=400 pool only; unselected; delta[v,f]>tau; abs(delta[u,f])<=tau for every u!=v; therefore recurrence=0.2."
    )
    result.pop("dominance_ratio", None)
    result["candidate_pool_feature_ids"] = pool.tolist()
    result["inputs"] = inputs
    result["selector_source"] = provenance
    # Record the actual readout index and offsets in the unchanged target sentence.
    alignment = []
    tokenizer = pair.model.model.tokenizer
    for prompt, ids in zip(
        [inputs["clean"]] + [r["prompt"] for r in inputs["variants"]], result["token_ids"]
    ):
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        shift = len(ids) - len(encoded["input_ids"])
        target_start = len(prompt) - len(inputs["clean"])
        target_indices = [
            i + shift
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < len(prompt) and b > target_start
        ]
        alignment.append(
            {
                "prompt_length_characters": len(prompt),
                "token_count": len(ids),
                "capture_position_from_end": -1,
                "capture_token_index": len(ids) - 1,
                "capture_token_id": ids[-1],
                "capture_token_text": tokenizer.decode([ids[-1]]),
                "target_character_start": target_start,
                "target_character_end": len(prompt),
                "target_token_indices": target_indices,
                "target_text_identical": prompt[target_start:] == inputs["clean"],
                "blank_line_before_target": target_start == 0
                or prompt[:target_start].endswith("\n\n"),
                "captured_token_in_target": len(ids) - 1 in target_indices,
                "tokenization_matches_captured_ids": ids[shift:] == encoded["input_ids"],
            }
        )
    result["alignment"] = alignment
    result["numeric_selection_completed_at"] = datetime.now(timezone.utc).isoformat()
    np.savez_compressed(
        out / "rss_activations.npz", clean=codes[0], variants=codes[1:], delta=delta
    )
    write_json(out / "rss_case.json", result)
    config = {
        "saebase_version": saebase.__version__,
        "model_key": "llama31_8b",
        "sae_key": "llama31_8b_lxr_8x",
        "layer": 21,
        "hook": "blocks.21.hook_resid_post",
        "capture_position": -1,
        "native_raw_codes": True,
        "sar_filter": False,
        "rss": result["rss_parameters"],
        "identity": pair.identity.to_dict(),
        "model_dtype": str(next(pair.model.model.parameters()).dtype),
        "seed": 0,
        "selector_source": provenance,
        "inputs": inputs,
        "software_versions": {
            n: version(n) for n in ["torch", "numpy", "sae-lens", "transformer-lens"]
        },
        "command": sys.argv,
    }
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml")
    info, annotations = annotate_features(
        pair, sorted(all_ids), neuronpedia_settings(OmegaConf.create({"task": {}}))
    )
    for row in result["rss_features"] + [r for g in groups for r in g["features"]]:
        row["annotation"] = annotations[row["feature_id"]]
    result["neuronpedia"] = info
    write_json(out / "rss_case.json", result)
    write_json(
        out / "run.json",
        {
            "status": "complete",
            "capture_count": 6,
            "codes_shape": list(codes.shape),
            "delta_shape": list(delta.shape),
            "rss_selected_count": len(selected),
            "strict_context_candidate_counts": [len(g["features"]) for g in groups],
            "alignment": alignment,
            "annotation": info,
            "selector_source": provenance,
            "command": sys.argv,
        },
    )
    print("Output:", out, flush=True)
    print(
        "RSS:",
        len(selected),
        "strict context candidates:",
        [len(g["features"]) for g in groups],
        "annotations:",
        info["status"],
        flush=True,
    )
    for row in result["rss_features"]:
        print(
            "RSS",
            row["feature_id"],
            row["direction"],
            row["delta"],
            "; ".join(e["description"] for e in row["annotation"]["explanations"]),
            flush=True,
        )
    for g in groups:
        print("CONTEXT", g["topic"], flush=True)
        for row in g["features"]:
            print(
                row["feature_id"],
                row["delta"],
                "; ".join(e["description"] for e in row["annotation"]["explanations"]),
                flush=True,
            )


if __name__ == "__main__":
    main()
