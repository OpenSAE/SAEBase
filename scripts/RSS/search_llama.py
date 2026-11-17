"""Search explicit prompt designs on Llama 3.1, retaining every real case."""

import argparse
import gc
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from saebase.api import LoadedResearchPair, SAEIdentity, load_registered
from saebase.api.sae_loading import load_sae_for_profile
from saebase.api.neuronpedia import annotation_session
from tasks.capture.operation import neuronpedia_settings
from tasks.capture.rss_mode import measure, annotate, markdown

TOPICS = [
    r"basketball|NBA|sports|athletic",
    r"space travel|space exploration|astronaut|outer space|astronomy|stargazing|rocket",
    r"music|violin|symphon|orchestra|concert",
    r"chess",
    r"wildlife|animal|elephant|safari|savanna",
]
TARGET = r"bak|cook|recipe|ingredient|bread|banana"
ORIGINAL = [
    "A basketball player scored a three-point shot.",
    "An astronaut floated outside a spaceship.",
    "A violinist performed a classical symphony.",
    "A chess player moved a knight across the board.",
    "An elephant walked across the African savanna.",
]
RICH = [
    "The scene is an NBA basketball game. LeBron James dribbles the basketball down the court, scores a three-point shot, and celebrates with his basketball team.",
    "The scene is a NASA space mission. An astronaut floats in zero gravity outside a spaceship, beside a space station orbiting Earth.",
    "The scene is a classical music concert. A violinist plays a violin concerto by Beethoven while an orchestra performs a symphony in the concert hall.",
    "The scene is a world chess championship. Magnus Carlsen moves a knight across the chessboard, attacks the queen, and checks his opponent's king.",
    "The scene is a wildlife safari in the African savanna. A herd of elephants walks past giraffes, lions, and zebras near a watering hole.",
]
BANKS = [
    ("rich_scene", RICH),
    ("explicit_background", ["Keep this background in mind for the request: " + c for c in RICH]),
    ("original", ORIGINAL),
    ("repeated_scene", [" ".join([c] * 3) for c in RICH]),
]
PROMPTS = [
    "Adapt a banana bread recipe to this setting.",
    "Describe how to bake banana bread in this scene.",
    "Suggest a banana bread recipe for this scene.",
    "For this setting, give a banana bread recipe.",
    "Write a cooking recipe inspired by this scene.",
    "Please outline the steps for baking banana bread.",
    "Banana bread recipe:",
    "Bake banana bread.",
    "Banana bread",
    "Baking",
    "Cooking",
    "Recipe?",
]


def text(row):
    return "; ".join(e["description"] for e in row.get("annotation", {}).get("explanations", []))


def navigate(result):
    groups = [
        [r for r in g["features"] if r["delta"][v] >= 0.25 and re.search(TOPICS[v], text(r), re.I)]
        for v, g in enumerate(result["context_candidates"])
    ]
    targets = [r for r in result["rss_features"] if re.search(TARGET, text(r), re.I)]
    return {
        "contexts": groups,
        "target": targets,
        "context_count": sum(bool(g) for g in groups),
        "bidirectional_target": {r["direction"] for r in targets} == {-1, 1},
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 21])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    (out / "design.json").write_text(
        json.dumps(
            {
                "contexts": BANKS,
                "targets": PROMPTS,
                "topic_navigation": TOPICS,
                "target_navigation": TARGET,
                "minimum_display_change": 0.25,
                "note": "Prompt search is exploratory. Explanation navigation is applied after unchanged RSS selection.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    pair = load_registered(
        model_key="llama31_8b",
        sae_key="llama31_8b_lxr_8x",
        layer=args.layers[0],
        device="cuda",
        require_patch=False,
    )
    model = pair.model
    options = neuronpedia_settings(OmegaConf.create({"task": {}}))
    cases = []
    count = 0
    with annotation_session():
        for layer in args.layers:
            if pair.identity.layer != layer:
                del pair
                gc.collect()
                torch.cuda.empty_cache()
                sae, metadata = load_sae_for_profile("llama31_8b_lxr_8x", layer, device="cuda")
                pair = LoadedResearchPair(
                    model=model,
                    sae=sae,
                    identity=SAEIdentity.from_loaded(
                        sae=sae, metadata=metadata, model_key="llama31_8b"
                    ),
                )
            for bank, contexts in BANKS:
                for prompt in PROMPTS:
                    count += 1
                    directory = out / f"case_{count:03d}_l{layer}_{bank}"
                    directory.mkdir()
                    result, codes, delta = measure(
                        pair, prompt, contexts, ROOT / "scripts/RSS/rss.py"
                    )
                    result["search_design"] = {"bank": bank, "layer": layer, "case": count}
                    np.savez_compressed(
                        directory / "rss_activations.npz",
                        clean=codes[0],
                        variants=codes[1:],
                        delta=delta,
                    )
                    (directory / "rss_case.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                    )
                    annotate(pair, result, options)
                    (directory / "rss_case.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                    )
                    (directory / "rss_case.md").write_text(markdown(result))
                    nav = navigate(result)
                    (directory / "navigation.json").write_text(
                        json.dumps(nav, ensure_ascii=False, indent=2) + "\n"
                    )
                    cases.append(
                        {
                            "directory": str(directory),
                            "layer": layer,
                            "bank": bank,
                            "prompt": prompt,
                            "contexts": nav["context_count"],
                            "target_count": len(nav["target"]),
                            "target_bidirectional": nav["bidirectional_target"],
                        }
                    )
                    (out / "progress.json").write_text(
                        json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
                    )
                    print(
                        f"{count:03d} L{layer} {bank}: contexts={nav['context_count']}/5 target={len(nav['target'])} bidirectional={nav['bidirectional_target']} | {prompt}",
                        flush=True,
                    )
                    if nav["context_count"] == 5 and len(nav["target"]) >= 2:
                        (out / "provisional_match.json").write_text(
                            json.dumps(cases[-1], ensure_ascii=False, indent=2) + "\n"
                        )
                        print(f"PROVISIONAL MATCH: {directory}", flush=True)
                        return


if __name__ == "__main__":
    main()
