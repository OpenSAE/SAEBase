"""Purpose: Each of the five data sets has been completed on 10 issues SAR Four-bar search results are organized into a subject-by-issue Chinese probability comparison..
Scope of application: Read raw measurements and existing searches JSON,Reservations are interacting with approximation; complete issues, explanations, two probabilities and increased co-delivery.
"""

import json
import math
from pathlib import Path


ROOT = Path("/workspace/SAEBase")
OUT = ROOT / "experiment/batches/capture/sar_qa_comparison"
DATASETS = ["GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"]
all_groups = []
lines = [
    "# 10 questions each of the five data sets: SAR Four Status Mode Search",
    "",
    "Model: Llama 3.1 8B Base,SAE L21;Each data set before following 100 Theme Filter, 10 Question SAR The results of the survey are as follows:.",
    "",
    "Strict conditions: 00/10/01 Average token It’s a high probability. 50%,Max.≤5 Percentage, 11 It’s at least higher than the three. 5 Percentage, measured logprob Interactively positive. Only the ten features selected for each topic are searched, all 45 One. pair;Four states must be fully consistent with the other characteristics..",
    "",
    "Strict hit. 0.When each issue has a weak candidate, show the first three baseline conditions met and measured interactively I>0,11 The largest increase in the number of candidates above the highest baseline; these approximations are not referred to as hard hits.",
    "",
    "First act of the Probability Form 100×exp(g),Second act 100×exp(Tg),T As the original standard answer token numbers;not included in line 2 EOS.Use this line for the last column. 11 Less the top three of this line, in percentage points. The answer follows the original data set and this search does not recheck the answer to the title..",
    "",
    "| Data Set | Number of topics | Full background four. | Number of questions with weak and positive interactive candidates | Strict hit. |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for dataset in DATASETS:
    run = ROOT / f"experiment/{dataset}/llama31_8b/capture/reference_additivity"
    raw = json.loads((run / "reports/pair_pattern_search_50_5_5/search.json").read_text())
    source = json.loads((run / "reports/question_features_pairs_zh.json").read_text())
    questions = []
    for q, original in zip(raw["questions"], source["questions"]):
        assert q["sample_id"] == original["sample_id"]
        eligible = [
            c
            for c in q["observed"]["candidates"]
            if c["interaction"] > 0 and c["both_gain_over_highest_baseline"] > 0
        ]
        chosen = eligible[0] if eligible else None
        record = {
            **{
                k: q[k]
                for k in [
                    "number",
                    "sample_id",
                    "question",
                    "question_zh",
                    "answer",
                    "answer_zh",
                    "answer_token_count",
                    "observed_probability_range",
                    "artifact",
                ]
            },
            "full_prompt": original["prompt"],
            "selected_feature_ids": q["selected_feature_ids"],
            "complete_observed_quartets": q["observed"]["complete_four_state_backgrounds"],
            "baseline_qualified_count": q["observed"]["baseline_qualified_count"],
            "strict_match_count": q["observed"]["strict_match_count"],
            "chosen_positive_candidate": chosen,
        }
        if chosen:
            joint = {
                state: 100 * math.exp(q["answer_token_count"] * chosen["g"][state])
                for state in ["00", "10", "01", "11"]
            }
            record["probability_rows"] = [
                {
                    "name": "Average token Probability",
                    "values_percent": {
                        s: chosen["probability"][s] * 100 for s in ["00", "10", "01", "11"]
                    },
                    "gain_percentage_points": chosen["both_gain_over_highest_baseline"] * 100,
                },
                {
                    "name": "Joint probability of answers",
                    "values_percent": joint,
                    "gain_percentage_points": joint["11"]
                    - max(joint[s] for s in ["00", "10", "01"]),
                },
            ]
        questions.append(record)
    group = {
        "dataset": dataset,
        "source_run": str(run),
        "questions": questions,
        "observed_quartets": sum(q["complete_observed_quartets"] for q in questions),
        "weak_candidate_questions": sum(
            q["chosen_positive_candidate"] is not None for q in questions
        ),
        "strict_matches": sum(q["strict_match_count"] for q in questions),
    }
    all_groups.append(group)
    lines.append(
        f"| {dataset} | {len(questions)} | {group['observed_quartets']} | {group['weak_candidate_questions']} | {group['strict_matches']} |"
    )
for group in all_groups:
    lines += ["", f"## {group['dataset']}:10 Question", ""]
    for q in group["questions"]:
        lines += [
            f"### {q['number']}. {q['question_zh']}",
            "",
            f"Title:{q['question']}",
            "",
            f"Dataset standard answers:{q['answer']};Chinese:{q['answer_zh']}.T={q['answer_token_count']}.",
            "",
        ]
        chosen = q["chosen_positive_candidate"]
        if chosen:
            lines += [
                f"Pair:**{chosen['pair_ids'][0]} × {chosen['pair_ids'][1]}**",
                "",
                f"- **{chosen['pair_ids'][0]}**:{chosen['feature_explanations_zh'][0]}.",
                f"- **{chosen['pair_ids'][1]}**:{chosen['feature_explanations_zh'][1]}.",
                "",
                "| Indicators | 00 | 10 | 01 | 11 | Beyond the highest baseline |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in q["probability_rows"]:
                numbers = " | ".join(
                    f"{row['values_percent'][s]:.4f}%" for s in ["00", "10", "01", "11"]
                )
                lines.append(
                    f"| {row['name']} | {numbers} | {row['gain_percentage_points']:+.4f} Percentage |"
                )
            lines += [
                "",
                f"Weak candidate, not achieved 5 Increase in percentage points. I={chosen['interaction']:+.9f}.",
                f"Fixed background status:{chosen['background']};Original measurement line (from 0 Count):{chosen['coalition_indices_zero_based']}.",
                "",
            ]
        else:
            maximum = q["observed_probability_range"][1] * 100
            formatted = f"{maximum:.4f}" if maximum >= 0.0001 else f"{maximum:.6g}"
            lines += [
                f"No positive and interactive candidates from the baseline were identified; original 384 Average of group measurements token The highest probability is **{formatted}%**.",
                "",
            ]
        lines += [
            "<details><summary>Full original input (with all options and mission statement))</summary>",
            "",
            "```text",
            q["full_prompt"],
            "```",
            "",
            "</details>",
            "",
        ]
(OUT / "five_datasets_pair_patterns_zh.md").write_text("\n".join(lines) + "\n")
(OUT / "five_datasets_pair_patterns_zh.json").write_text(
    json.dumps(
        {
            "thresholds": {
                "minimum_baseline_probability": 0.5,
                "maximum_baseline_spread": 0.05,
                "minimum_both_gain": 0.05,
            },
            "groups": all_groups,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
assert sum(len(g["questions"]) for g in all_groups) == 50
assert sum(g["observed_quartets"] for g in all_groups) == 6400
assert sum(g["weak_candidate_questions"] for g in all_groups) == 11
for group in all_groups:
    for q in group["questions"]:
        c = q["chosen_positive_candidate"]
        if not c:
            continue
        assert min(c["probability"][s] for s in ["00", "10", "01"]) > 0.5
        assert c["baseline_spread"] <= 0.05 and c["interaction"] > 0
        for row in q["probability_rows"]:
            assert (
                abs(
                    row["gain_percentage_points"]
                    - (
                        row["values_percent"]["11"]
                        - max(row["values_percent"][s] for s in ["00", "10", "01"])
                    )
                )
                < 1e-12
            )
print(
    "50 questions; 6400 observed quartets; 11 weak positive candidates; 0 strict matches; both probability-row gains verified."
)
