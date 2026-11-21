"""Search saved SAR measurements for a flat three-state baseline and joint gain.

The four states must share an identical background for all other features.
Reports separate direct observations from optional quadratic-surrogate predictions;
no model inference is performed and source measurements remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def search_patterns(feature_ids, masks, responses, *, minimum=0.5, spread=0.05, gain=None):
    ids = list(feature_ids)
    masks = np.asarray(masks)
    responses = np.asarray(responses, dtype=np.float64)
    if masks.shape != (len(responses), len(ids)) or not np.isin(masks, [0, 1]).all():
        raise ValueError("Expected binary masks aligned with responses and feature IDs")
    if not np.isfinite(responses).all() or np.any(responses > 1e-7):
        raise ValueError("Expected finite nonpositive mean log probabilities")
    if (
        len(set(ids)) != len(ids)
        or not 0 <= minimum < 1
        or not 0 <= spread <= 1
        or (gain is not None and not 0 <= gain <= 1)
    ):
        raise ValueError("Invalid IDs or probability thresholds")
    masks = masks.astype(int)
    candidates = []
    quartets = 0
    for i, j in combinations(range(len(ids)), 2):
        others = [k for k in range(len(ids)) if k not in (i, j)]
        groups = {}
        for index, mask in enumerate(masks):
            background = tuple(int(mask[k]) for k in others)
            state = (int(mask[i]), int(mask[j]))
            groups.setdefault(background, {}).setdefault(state, []).append(index)
        for background, states in groups.items():
            if len(states) != 4:
                continue
            quartets += 1
            indices = {f"{a}{b}": states[(a, b)][0] for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))}
            g = {state: float(responses[index]) for state, index in indices.items()}
            p = {state: float(np.exp(value)) for state, value in g.items()}
            baseline = [p[state] for state in ("00", "10", "01")]
            gap = max(baseline) - min(baseline)
            lift = p["11"] - max(baseline)
            interaction = g["11"] - g["10"] - g["01"] + g["00"]
            if min(baseline) <= minimum or gap > spread:
                continue
            candidates.append(
                {
                    "pair_ids": [int(ids[i]), int(ids[j])],
                    "g": g,
                    "probability": p,
                    "baseline_spread": gap,
                    "both_gain_over_highest_baseline": lift,
                    "interaction": interaction,
                    "strict_match": bool(
                        lift > 0 and interaction > 0 and (gain is None or lift >= gain)
                    ),
                    "background": {str(ids[k]): int(v) for k, v in zip(others, background)},
                    "coalition_indices_zero_based": indices,
                    "duplicate_condition_indices": {
                        f"{a}{b}": states[(a, b)] for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))
                    },
                }
            )
    candidates.sort(
        key=lambda row: (
            not row["strict_match"],
            -row["interaction"],
            row["baseline_spread"],
            row["pair_ids"],
            list(row["background"].values()),
        )
    )
    return {
        "complete_four_state_backgrounds": quartets,
        "baseline_qualified_count": len(candidates),
        "strict_match_count": sum(row["strict_match"] for row in candidates),
        "best": next((row for row in candidates if row["strict_match"]), None),
        "ranking": "descending_observed_logprob_interaction",
        "candidates": candidates,
    }


def scan_run(run, output, *, minimum=0.5, spread=0.05, gain=None, surrogate=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    source = run / "reports/question_features_pairs_zh.json"
    document = json.loads(source.read_text())
    paper = None
    if surrogate:
        from omegaconf import OmegaConf
        from tasks.capture.sar_mode import prepare_sar

        paper = prepare_sar(OmegaConf.load(run / "metadata/config.yaml"))["paper"]
    questions = []
    for number, q in enumerate(document["questions"], 1):
        artifact = run / q["sar_artifact"]["path"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != q["sar_artifact"]["sha256"]:
            raise ValueError(f"SAR artifact checksum mismatch: {artifact}")
        with np.load(artifact, allow_pickle=False) as arrays:
            ids = arrays["feature_ids"].tolist()
            selected = {f["feature_id"]: f for f in q["features"]}
            observed = search_patterns(
                ids, arrays["masks"], arrays["responses"], minimum=minimum, spread=spread, gain=gain
            )
            for candidate in observed["candidates"]:
                candidate["feature_explanations_zh"] = [
                    selected[i].get("explanation_zh", "Explanation missing") for i in candidate["pair_ids"]
                ]
            result = {
                "number": number,
                "sample_id": q["sample_id"],
                "question": q["question"],
                "question_zh": q["question_zh"],
                "answer": q["answer"],
                "answer_zh": q.get("answer_zh", q["answer"]),
                "answer_token_count": len(arrays["answer_token_ids"]),
                "selected_feature_ids": ids,
                "artifact": q["sar_artifact"],
                "observed_probability_range": [
                    float(np.exp(arrays["responses"].min())),
                    float(np.exp(arrays["responses"].max())),
                ],
                "observed": observed,
            }
            if paper is not None:
                if len(ids) > 16:
                    raise ValueError("Exhaustive surrogate scan is bounded to K <= 16")
                masks = ((np.arange(2 ** len(ids))[:, None] >> np.arange(len(ids))) & 1).astype(
                    bool
                )
                design = paper.design(arrays["masks"], order=2)
                coefficients, _, _, _ = np.linalg.lstsq(
                    design[arrays["fit_indices"]],
                    arrays["responses"][arrays["fit_indices"]],
                    rcond=None,
                )
                predicted = paper.design(masks, order=2) @ coefficients
                if np.any(predicted > 1e-7):
                    result["surrogate"] = {
                        "status": "outside_logprob_range",
                        "reason": "Some quadratic predictions exceed zero; no probability interpretation applied",
                    }
                else:
                    result["surrogate"] = search_patterns(
                        ids, masks, predicted, minimum=minimum, spread=spread, gain=gain
                    )
                    result["surrogate"]["status"] = "predictions_only_not_model_measurements"
            questions.append(result)
    result = {
        "source_run": str(run),
        "model": document["model"],
        "layer": document["layer"],
        "dataset": document["dataset"],
        "thresholds": {
            "baseline_probability_strictly_above": minimum,
            "max_baseline_spread": spread,
            "minimum_both_gain": gain,
            "positive_interaction_required": True,
        },
        "scope": "All pairs among each question's original selected features; only identical-background four-state observations qualify",
        "probability": "exp(mean answer-token logprob), not full-answer joint probability",
        "duplicate_policy": "First original measurement per identical mask; duplicate row indices retained",
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "questions": questions,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "search.json"
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Flat baseline and dual feature gain search",
        "",
        f"Model {result['model']},L{result['layer']},Data Set {result['dataset']},{len(questions)}Question.",
        "",
        f"Conditions: 00/10/01 Average token It’s more likely than that.{minimum * 100:g}%,The maximum difference does not exceed{spread * 100:g}Percentage; 11 Higher than the three highest and measured logprob Four sets are positive. interaction Order down. Other features of the four states must be identical..",
        "No additional probabilities threshold."
        if gain is None
        else f"This is a significant increase threshold.:{gain * 100:g}Percentage.",
        "",
        "Only search for the previously selected feature for each subject. Real measurements are saved separately from the proposed projection for the second stage; projections cover all binary systems mask,However, this cannot be used as an undetermined combination of experimental evidence. Repeated conditions take the first measurement from the original record, and do not select the rows by rating..",
        "",
        "| Node | Original full fourgs | Baseline eligibility | Qualified four. | Highest interaction pair | p00 / p10 / p01 / p11 | 11 Beyond the highest baseline |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for q in questions:
        observed = q["observed"]
        best = observed["best"]
        if best:
            probs = " / ".join(
                f"{best['probability'][s] * 100:.4f}%" for s in ["00", "10", "01", "11"]
            )
            pair = " × ".join(map(str, best["pair_ids"]))
            lift = f"{best['both_gain_over_highest_baseline'] * 100:+.4f}Percentage"
        else:
            pair = "—"
            probs = "—"
            lift = "—"
        lines.append(
            f"| {q['number']} | {observed['complete_four_state_backgrounds']} | {observed['baseline_qualified_count']} | {observed['strict_match_count']} | {pair} | {probs} | {lift} |"
        )
    for q in questions:
        lines += [
            "",
            f"## {q['number']}. {q['question_zh']}",
            "",
            q["question"],
            "",
            f"Standard answer:{q['answer']}({q['answer_zh']}).",
        ]
        best = q["observed"]["best"]
        if not best:
            lines += [
                "",
                f"Original 384 Average in group observations token Probability range:{q['observed_probability_range'][0] * 100:.4f}%～{q['observed_probability_range'][1] * 100:.4f}%,No baseline 4 was found.",
            ]
            continue
        lines += [
            "",
            f"Highest interaction pair:{best['pair_ids']},"
            + " × ".join(best["feature_explanations_zh"]),
            "",
            f"Fixed Background:{best['background']};Interactive I={best['interaction']:+.9f}.",
            "",
            "| Indicators | 00 | 10 | 01 | 11 | Beyond the highest baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, factor in [("Average token Probability", 1), ("Joint probability of answers", q["answer_token_count"])]:
            values = [
                float(100 * np.exp(best["g"][state] * factor)) for state in ["00", "10", "01", "11"]
            ]
            gain_pp = values[3] - max(values[:3])
            numbers = " | ".join(f"{value:.4f}%" for value in values)
            lines.append(f"| {label} | {numbers} | {gain_pp:+.4f} Percentage |")
        lines += ["", f"Original measurement line(00/10/01/11,From 0 Count):{best['coalition_indices_zero_based']}."]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=0.05)
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Optional explicit probability gain floor; no floor by default",
    )
    parser.add_argument("--surrogate", action="store_true")
    args = parser.parse_args()
    result = scan_run(
        args.run,
        args.output,
        minimum=args.minimum,
        spread=args.spread,
        gain=args.gain,
        surrogate=args.surrogate,
    )
    print(
        json.dumps(
            {
                "questions": len(result["questions"]),
                "observed_quartets": sum(
                    q["observed"]["complete_four_state_backgrounds"] for q in result["questions"]
                ),
                "observed_matches": sum(
                    q["observed"]["strict_match_count"] for q in result["questions"]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
