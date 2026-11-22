"""Purpose: To be measured by experience interaction Collapse Five Data Sets 50 Qualified. pair,It shows both probability and growth on a case-by-case basis..
Scope of application: read-only available SAR measurements; retaining original problems and fixed background, eliminating additional 5 Threshold of percentage point increase.
"""

import json
import math
from pathlib import Path


import numpy as np

ROOT = Path("/workspace/SAEBase")
OUT = ROOT / "experiment/batches/capture/sar_qa_comparison"
selected = []
for dataset in ["GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"]:
    run = ROOT / f"experiment/{dataset}/llama31_8b/capture/reference_additivity"
    raw = json.loads((run / "reports/pair_pattern_search_by_interaction/search.json").read_text())
    original = json.loads((run / "reports/question_features_pairs_zh.json").read_text())
    by_id = {q["sample_id"]: q for q in original["questions"]}
    for q in raw["questions"]:
        best = q["observed"]["best"]
        if best is None:
            continue
        with np.load(run / q["artifact"]["path"], allow_pickle=False) as arrays:
            ids = arrays["feature_ids"].tolist()
            for state, index in best["coalition_indices_zero_based"].items():
                assert float(arrays["responses"][index]) == best["g"][state]
                expected = dict(best["background"])
                expected.update(
                    {
                        str(best["pair_ids"][0]): int(state[0]),
                        str(best["pair_ids"][1]): int(state[1]),
                    }
                )
                assert arrays["masks"][index].astype(int).tolist() == [
                    expected[str(i)] for i in ids
                ]
        assert best["interaction"] > 0 and best["both_gain_over_highest_baseline"] > 0
        assert (
            min(best["probability"][s] for s in ["00", "10", "01"]) > 0.5
            and best["baseline_spread"] <= 0.05
        )
        row = {
            **q,
            "dataset": dataset,
            "source_run": str(run),
            "candidate": best,
            "full_prompt": by_id[q["sample_id"]]["prompt"],
        }
        row.pop("observed")
        row["table"] = []
        for label, factor in [("Average token Probability", 1), ("Joint probability of answers", q["answer_token_count"])]:
            values = {s: 100 * math.exp(factor * best["g"][s]) for s in ["00", "10", "01", "11"]}
            row["table"].append(
                {
                    "label": label,
                    "values_percent": values,
                    "gain_percentage_points": values["11"]
                    - max(values[s] for s in ["00", "10", "01"]),
                }
            )
        selected.append(row)
selected.sort(key=lambda row: -row["candidate"]["interaction"])
lines = [
    "# Based on the facts. interaction Sorted SAR Characteristics Yes",
    "",
    "Llama 3.1 8B Base,SAE L21;Research the originals of the five data sets. SAR Measurements, everything. Top10 All 45 Yeah..",
    "",
    "3 baseline average token Average probability>50%,Closeness is the worst use this time.≤5 Percentage; 11 Higher than the three baseline peaks, measured I>0.Cancel Extra 5 Raises the threshold in percentage points. I Biggest candidate, then cross the theme. I descending order;not ranking by probability increase.",
    "",
    "I=g11−g10−g01+g00.Four. g From a fixed background where other characteristics are fully consistent. token Probability=100×exp(g),Joint probability=100×exp(Tg),does not contain EOS.The last row by this line 11 Less the top three values in this line, using unrounded values. All data are measured in the original experiment and no second-stage model is used to predict.",
    "",
    f"Number of eligible topics:{len(selected)};Show only eligible entries.",
    "",
]
for rank, q in enumerate(selected, 1):
    b = q["candidate"]
    lines += [
        f"## {rank}. {q['dataset']} Original No.{q['number']}Question:{q['question_zh']}",
        "",
        q["question"],
        "",
        f"Dataset standard answers:{q['answer']};Chinese:{q['answer_zh']}.T={q['answer_token_count']}.",
        "",
        f"Pair:**{b['pair_ids'][0]} × {b['pair_ids'][1]}**;**Actual I={b['interaction']:+.6f}**.",
        "",
        f"- **{b['pair_ids'][0]}**:{b['feature_explanations_zh'][0]}.",
        f"- **{b['pair_ids'][1]}**:{b['feature_explanations_zh'][1]}.",
        "",
        "| Indicators | 00 | 10 | 01 | 11 | Beyond the highest baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in q["table"]:
        values = " | ".join(f"{row['values_percent'][s]:.4f}%" for s in ["00", "10", "01", "11"])
        lines.append(
            f"| {row['label']} | {values} | {row['gain_percentage_points']:+.4f} Percentage |"
        )
    lines += [
        "",
        f"Maximum deviation from baseline:{b['baseline_spread'] * 100:.4f}percentage points;fixed background:{b['background']}.",
        f"Original measurement line (from 0 Count):{b['coalition_indices_zero_based']}.",
        "",
        "<details><summary>Full original input and options</summary>",
        "",
        "```text",
        q["full_prompt"],
        "```",
        "",
        "</details>",
        "",
    ]
(OUT / "five_datasets_pairs_by_interaction_zh.md").write_text("\n".join(lines) + "\n")
(OUT / "five_datasets_pairs_by_interaction_zh.json").write_text(
    json.dumps(
        {
            "ranking": "descending_observed_logprob_interaction",
            "minimum_probability": 0.5,
            "maximum_baseline_spread": 0.05,
            "extra_probability_gain_floor": None,
            "searched_questions": 50,
            "selected_questions": selected,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    "Verified all four original masks and g values for",
    len(selected),
    "selected questions; ranked by I.",
)
