"""Check formal Capture selection against the existing fifty real measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from tasks.capture.scoring.reproduction_paths import RESULTS, question_directory
from tasks.capture.behavior import choose_pair, rank_singles
from tasks.capture.selection import candidate_ids, prepare_filter
from tasks.capture.whitening import Normalization


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, help="Write the verification JSON to this location")
    args = parser.parse_args(argv)
    root = RESULTS
    plan = json.loads((root / "plan.json").read_text())
    expected = {q["key"]: q for q in json.loads((root / "report.json").read_text())["questions"]}
    verified = []
    for question in plan["questions"]:
        reference = root / "reference" / question["dataset"]
        semantic = json.loads((reference / "sar_feature_filter.json").read_text())
        cohort = json.loads((reference / "sar_cohort_filter.json").read_text())
        metadata = json.loads((reference / "capture_normalization.json").read_text())
        with np.load(reference / "capture_normalization.npz") as norm:
            normalization = Normalization(
                norm["mean"], norm["std"], norm["variance"], metadata["identity"], metadata
            )
        codes = torch.zeros(question["feature_count"])
        for feature in question["active_features"]:
            codes[feature["feature_id"]] = feature["activation"]
        cfg = OmegaConf.create(
            {
                "task": {
                    "name": "capture",
                    "capture": {
                        "mode": "I",
                        "selection": {"candidate_count": 10, "feature_selection": "whiten"},
                        "output": {"feature_count": 1},
                    },
                }
            }
        )
        context = prepare_filter(cfg)
        context.update(
            blocked_features={int(k): v for k, v in semantic["blocked_features"].items()},
            cohort_blocked_features=set(cohort["blocked_ids"]),
        )
        selected = candidate_ids(cfg, None, codes, normalization, context, {})
        with np.load(question_directory(root, question["key"]) / "coalitions.npz") as arrays:
            ids, scores, token_count = (
                arrays["feature_ids"].tolist(),
                arrays["responses"],
                len(arrays["answer_token_ids"]),
            )
        assert selected == ids
        old = expected[question["key"]]
        singles = rank_singles(
            ids, scores[-1], [scores[(len(scores) - 1) ^ (1 << i)] for i in range(len(ids))]
        )
        assert singles[0]["feature_id"] == old["max_g01_single"]["feature"]["feature_id"]
        assert singles[0]["g"] == old["max_g01_single"]["g"]["01"]
        assert singles[0]["delta_g"] == old["max_g01_single"]["delta_g"]
        best, _, _ = choose_pair(ids, scores, token_count)
        assert best["pair_ids"] == old["max_interaction_pair"]["pair_ids"]
        assert best["g"] == old["max_interaction_pair"]["g"]
        assert best["interaction"] == old["max_interaction_pair"]["interaction"]
        cfg.task.capture.selection.candidate_count = "all"
        for mode, field in [
            ("whiten", "whiten_top_feature"),
            ("non_whiten", "absolute_activation_top_feature"),
        ]:
            cfg.task.capture.mode = mode
            assert (
                candidate_ids(cfg, None, codes, normalization, None, {})[0]
                == old[field]["feature_id"]
            )
        verified.append(
            {
                "question": question["key"],
                "candidates": selected,
                "single": singles[0],
                "pair": best["pair_ids"],
                "I": best["interaction"],
            }
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "validation_kind": "formal selection on previously measured real coalitions; no new model inference",
                    "question_count": len(verified),
                    "questions": verified,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        )
    print(f"Formal Capture matches all four selections on {len(verified)} real questions")


if __name__ == "__main__":
    main()
