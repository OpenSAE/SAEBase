"""Measure the four literal RSS diagnostic groups, including supplied newlines."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from saebase.api import load_registered
from saebase.api.neuronpedia import annotation_session
from tasks.capture.operation import neuronpedia_settings
from tasks.capture.rss_mode import measure, annotate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((ROOT / "scripts/RSS/diagnostic_inputs.json").read_text())
    write_json(out / "inputs.json", inputs)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    pair = load_registered(
        model_key="llama31_8b",
        sae_key="llama31_8b_lxr_8x",
        layer=21,
        device="cuda",
        require_patch=False,
    )
    config = {
        "identity": pair.identity.to_dict(),
        "capture_position": -1,
        "seed": 0,
        "model_dtype": str(next(pair.model.model.parameters()).dtype),
        "versions": {
            name: version(name) for name in ["torch", "numpy", "sae-lens", "transformer-lens"]
        },
        "whitespace_policy": inputs["whitespace_policy"],
    }
    write_json(out / "config.json", config)
    with annotation_session():
        for case in inputs["cases"]:
            directory = out / case["id"]
            directory.mkdir()
            result, codes, delta = measure(
                pair,
                case["clean"],
                case["contexts"],
                ROOT / "scripts/RSS/rss.py",
                variants=case["variants"],
            )
            result["diagnostic_case"] = case
            result["captured_last_tokens"] = [
                pair.model.model.tokenizer.decode([ids[-1]]) for ids in result["token_ids"]
            ]
            np.savez_compressed(
                directory / "rss_activations.npz", clean=codes[0], variants=codes[1:], delta=delta
            )
            write_json(directory / "rss_case.json", result)
            annotate(pair, result, neuronpedia_settings(OmegaConf.create({"task": {}})))
            write_json(directory / "rss_case.json", result)
            print(
                case["id"],
                f"RSS={len(result['rss_features'])}",
                f"annotations={result['neuronpedia']['status']}",
                f"last={result['captured_last_tokens']}",
                flush=True,
            )
    write_json(
        out / "run.json",
        {
            "status": "complete",
            "case_count": len(inputs["cases"]),
            "input_count": 6 * len(inputs["cases"]),
            "configuration": "config.json",
            "inputs": "inputs.json",
        },
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
