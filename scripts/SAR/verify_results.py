"""Independently audit the fifty-question report against complete mask cubes."""

import json
import math

import numpy as np

from scripts.SAR.exhaustive import OUTPUT, original_scores, source_arrays, write_json


def main(*, require_complete=True, report_name="report"):
    report = json.loads((OUTPUT / f"{report_name}.json").read_text())
    plan = json.loads((OUTPUT / "plan.json").read_text())
    sources = {q["key"]: q for q in plan["questions"]}
    checked = []
    for question in report["questions"]:
        key = question["key"]
        with np.load(OUTPUT / key / "coalitions.npz") as arrays:
            scores = arrays["responses"].copy()
            masks = arrays["masks"].copy()
            ids = arrays["feature_ids"].tolist()
        assert scores.shape == (1024,) and masks.shape == (1024, 10)
        assert np.array_equal(masks.dot(1 << np.arange(10)), np.arange(1024))
        for state, value in original_scores(source_arrays(sources[key])).items():
            assert scores[state] == value
        # Independent vectorized finite differences of the 10-dimensional cube.
        cube = scores.reshape((2,) * 10)
        maxima = []
        for first in range(10):
            for second in range(first + 1, 10):
                delta = np.diff(np.diff(cube, axis=first), axis=second)
                maxima.append(float(delta.max()))
        pair = question["max_interaction_pair"]
        assert math.isclose(pair["interaction"], max(maxima), rel_tol=0, abs_tol=1e-12)
        first, second = [ids.index(fid) for fid in pair["pair_ids"]]
        background = sum(
            int(pair["background"][str(fid)]) << index
            for index, fid in enumerate(ids)
            if index not in (first, second)
        )
        expected = [
            background,
            background | (1 << first),
            background | (1 << second),
            background | (1 << first) | (1 << second),
        ]
        for name, state in zip(("00", "10", "01", "11"), expected):
            assert pair["g"][name] == scores[state]
        single = question["max_g01_single"]
        full_state = len(scores) - 1
        assert single["g"]["00"] == scores[full_state]
        assert single["g"]["01"] == max(scores[full_state ^ (1 << index)] for index in range(10))
        assert single["state"] == full_state ^ (1 << ids.index(single["feature"]["feature_id"]))
        for candidate in [pair, single]:
            for row, factor in zip(candidate["probabilities"], [1, question["answer_token_count"]]):
                for state, value in candidate["g"].items():
                    assert row["percent"][state] == 100 * math.exp(factor * value)
                values = row["percent"]
                lift = (
                    values["11"] - max(values[s] for s in ("00", "10", "01"))
                    if "11" in values
                    else values["01"] - values["00"]
                )
                assert row["change_percentage_points"] == lift
        features = sources[key]["active_features"]
        raw_winner = max(features, key=lambda f: (abs(f["activation"]), -f["feature_id"]))
        assert question["absolute_activation_top_feature"]["feature_id"] == raw_winner["feature_id"]
        required = [
            *pair["features"],
            single["feature"],
            question["whiten_top_feature"],
            question["absolute_activation_top_feature"],
        ]
        assert all(f["explanation_zh"] != "尚无中文解释" for f in required)
        assert question["verification"]["activation_max_abs_error"] == 0
        assert question["verification"]["codes_max_abs_error"] == 0
        assert question["verification"].get("native_edit_patch_max_abs_error", 0) == 0
        checked.append(
            {
                "question": key,
                "states": 1024,
                "quartets": 11520,
                "max_interaction": pair["interaction"],
            }
        )
    if require_complete:
        assert len(checked) == 50
    write_json(
        OUTPUT / ("audit.json" if require_complete else "audit_partial.json"),
        {
            "status": "passed",
            "questions": checked,
            "states": 1024 * len(checked),
            "quartets": 11520 * len(checked),
            "checks": [
                "original scores unchanged",
                "all masks covered",
                "independent cube finite-difference maximum",
                "fixed pair backgrounds",
                "single maximum",
                "both probability scales and percentage-point changes",
                "native absolute activation winner",
                "Chinese explanations",
                "native replay gates",
            ],
        },
    )
    print(f"Audited {len(checked)} questions, {1024 * len(checked)} states")


if __name__ == "__main__":
    main()
