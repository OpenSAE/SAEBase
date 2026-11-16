"""目的：将五个数据集各十题的SAR四格搜索结果整理为逐题中文概率对照。
适用范围：读取原始测量及既有搜索JSON，保留正交互近似候选；完整问题、解释、两种概率和增幅共同交付。
"""

import json
import math
from pathlib import Path

ROOT = Path("/workspace/SAEBase")
OUT = ROOT / "experiment/central-maintenance/sar_fiveqa_ref100_20261114"
DATASETS = ["GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"]
all_groups = []
lines = [
    "# 五个数据集各十题：SAR四状态模式搜索",
    "",
    "模型：Llama 3.1 8B Base，SAE L21；沿用之前每个数据集100题筛选、10题SAR的实测结果。",
    "",
    "严格条件：00/10/01的平均token概率都严格高于50%，最大差≤5个百分点，11比三者最高值至少高5个百分点，且实测logprob交互为正。只搜索原先每题选定的十个特征，全部45个pair；四状态必须保持其他特征背景完全一致。",
    "",
    "严格命中为0。每题有弱候选时，展示满足前三个基线条件、实测交互I>0、11超过最高基线的候选中增幅最大的一个；这些近似候选不称为严格命中。",
    "",
    "概率表第一行为100×exp(g)，第二行为100×exp(Tg)，T为原标准答案token数；第二行不包含EOS。最后一列分别使用本行的11减去本行前三者最高值，单位为百分点。答案沿用原数据集标注，本次搜索不重新校验题目答案。",
    "",
    "| 数据集 | 题数 | 原实测完整背景四格 | 有弱正交互候选的题数 | 严格命中 |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for dataset in DATASETS:
    run = ROOT / f"experiment/{dataset}/llama31_8b/capture/qa5_L21_ref100_freq50_sar_20261114_42"
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
                    "name": "平均 token 概率",
                    "values_percent": {
                        s: chosen["probability"][s] * 100 for s in ["00", "10", "01", "11"]
                    },
                    "gain_percentage_points": chosen["both_gain_over_highest_baseline"] * 100,
                },
                {
                    "name": "答案联合概率",
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
    lines += ["", f"## {group['dataset']}：10题", ""]
    for q in group["questions"]:
        lines += [
            f"### {q['number']}. {q['question_zh']}",
            "",
            f"原题：{q['question']}",
            "",
            f"数据集标准答案：{q['answer']}；中文：{q['answer_zh']}。T={q['answer_token_count']}。",
            "",
        ]
        chosen = q["chosen_positive_candidate"]
        if chosen:
            lines += [
                f"Pair：**{chosen['pair_ids'][0]} × {chosen['pair_ids'][1]}**",
                "",
                f"- **{chosen['pair_ids'][0]}**：{chosen['feature_explanations_zh'][0]}。",
                f"- **{chosen['pair_ids'][1]}**：{chosen['feature_explanations_zh'][1]}。",
                "",
                "| 指标 | 00 | 10 | 01 | 11 | 超过最高基线 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in q["probability_rows"]:
                numbers = " | ".join(
                    f"{row['values_percent'][s]:.4f}%" for s in ["00", "10", "01", "11"]
                )
                lines.append(
                    f"| {row['name']} | {numbers} | {row['gain_percentage_points']:+.4f} 个百分点 |"
                )
            lines += [
                "",
                f"弱候选，未达到5个百分点增幅。实测交互I={chosen['interaction']:+.9f}。",
                f"固定背景状态：{chosen['background']}；原始测量行（从0计）：{chosen['coalition_indices_zero_based']}。",
                "",
            ]
        else:
            maximum = q["observed_probability_range"][1] * 100
            formatted = f"{maximum:.4f}" if maximum >= 0.0001 else f"{maximum:.6g}"
            lines += [
                f"未找到基线合格的正交互候选；原384组实测中平均token概率最高为 **{formatted}%**。",
                "",
            ]
        lines += [
            "<details><summary>完整原始输入（含全部选项与任务说明）</summary>",
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
