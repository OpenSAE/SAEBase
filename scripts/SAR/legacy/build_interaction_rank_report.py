"""目的：按实测interaction整理五数据集50题的合格pair，逐题展示两种概率及增幅。
适用范围：只读取已有SAR测量；保留原始问题和固定背景，取消额外的5个百分点增幅门槛。
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/SAEBase")
OUT = ROOT / "experiment/central-maintenance/sar_fiveqa_ref100_20261114"
selected = []
for dataset in ["GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"]:
    run = ROOT / f"experiment/{dataset}/llama31_8b/capture/qa5_L21_ref100_freq50_sar_20261114_42"
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
        for label, factor in [("平均 token 概率", 1), ("答案联合概率", q["answer_token_count"])]:
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
    "# 按实测interaction排序的SAR特征对",
    "",
    "Llama 3.1 8B Base，SAE L21；重新搜索五数据集各十题原始SAR测量，每题Top10全部45对。",
    "",
    "三个基线平均token概率均>50%，接近程度本次沿用最大差≤5个百分点；11高于三个基线最高值，实测I>0。取消额外的5个百分点提升门槛。每题保留I最大的一个候选，再跨题按I降序排列；不按概率增幅排序。",
    "",
    "I=g11−g10−g01+g00。四个g来自其他特征完全一致的固定背景。平均token概率=100×exp(g)，联合概率=100×exp(Tg)，不含EOS。最后一列分别按本行的11减去本行前三个最高值，使用未舍入数值。所有数据均为原实验实测，不使用二阶模型预测。",
    "",
    f"合格题目数：{len(selected)}；只展示合格项。",
    "",
]
for rank, q in enumerate(selected, 1):
    b = q["candidate"]
    lines += [
        f"## {rank}. {q['dataset']} 原第{q['number']}题：{q['question_zh']}",
        "",
        q["question"],
        "",
        f"数据集标准答案：{q['answer']}；中文：{q['answer_zh']}。T={q['answer_token_count']}。",
        "",
        f"Pair：**{b['pair_ids'][0]} × {b['pair_ids'][1]}**；**实测I={b['interaction']:+.6f}**。",
        "",
        f"- **{b['pair_ids'][0]}**：{b['feature_explanations_zh'][0]}。",
        f"- **{b['pair_ids'][1]}**：{b['feature_explanations_zh'][1]}。",
        "",
        "| 指标 | 00 | 10 | 01 | 11 | 超过最高基线 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in q["table"]:
        values = " | ".join(f"{row['values_percent'][s]:.4f}%" for s in ["00", "10", "01", "11"])
        lines.append(
            f"| {row['label']} | {values} | {row['gain_percentage_points']:+.4f} 个百分点 |"
        )
    lines += [
        "",
        f"基线最大差：{b['baseline_spread'] * 100:.4f}个百分点；固定背景：{b['background']}。",
        f"原测量行（从0计）：{b['coalition_indices_zero_based']}。",
        "",
        "<details><summary>完整原始输入与选项</summary>",
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
