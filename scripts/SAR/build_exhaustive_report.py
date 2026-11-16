"""Build the Chinese, per-question comparison from verified exhaustive scores."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from scripts.SAR.exhaustive import OUTPUT, ROOT, source_arrays, write_json
from tasks.capture.behavior import choose_pair, probabilities, rank_singles


def feature_rows(question, translations):
    features = {f["feature_id"]: dict(f) for f in question["active_features"]}
    reference = OUTPUT / "reference" / question["dataset"] / "capture_normalization.npz"
    if not reference.exists():
        reference = Path(question["source_run"]) / "metadata/capture_normalization.npz"
    with np.load(reference) as norm:
        raw = np.zeros(question["feature_count"])
        for fid, feature in features.items():
            raw[fid] = feature["activation"]
        valid = norm["std"] > 0
        zscore = np.zeros_like(raw)
        zscore[valid] = (raw[valid] - norm["mean"][valid]) / norm["std"][valid]
        ids = np.flatnonzero(valid)
        whiten_id = int(ids[np.lexsort((ids, -np.abs(zscore[ids])))][0])
        activation_id = int(np.argmax(np.abs(raw)))
        for fid in set(
            [whiten_id, activation_id, *[f["feature_id"] for f in question["top_features"]]]
        ):
            feature = features.setdefault(fid, {"feature_id": fid, "activation": float(raw[fid])})
            feature.update(
                zscore=float(zscore[fid]),
                mean=float(norm["mean"][fid]),
                std=float(norm["std"][fid]),
            )
    for feature in features.values():
        feature["explanation_zh"] = translations.get(str(feature["feature_id"]), "尚无中文解释")
    return features, whiten_id, activation_id


def analyze(question, translations):
    directory = OUTPUT / question["key"]
    completion = json.loads((directory / "complete.json").read_text())
    import hashlib

    if (
        hashlib.sha256((directory / "coalitions.npz").read_bytes()).hexdigest()
        != completion["coalitions_sha256"]
    ):
        raise ValueError("Exhaustive result checksum mismatch")
    with np.load(directory / "coalitions.npz") as arrays:
        ids = arrays["feature_ids"].tolist()
        scores = arrays["responses"].copy()
        token_count = len(arrays["answer_token_ids"])
        assert len(scores) == 1024 and len(ids) == 10 and np.isfinite(scores).all()
    original = source_arrays(question)
    features, whiten_id, activation_id = feature_rows(question, translations)
    pair, qualified, retained = choose_pair(ids, scores, token_count)
    for candidate in (pair, qualified, retained):
        if candidate is not None:
            candidate["features"] = [features[fid] for fid in candidate["pair_ids"]]
    full_state = len(scores) - 1
    measured_singles = rank_singles(
        ids, scores[full_state], [scores[full_state ^ (1 << i)] for i in range(len(ids))]
    )
    single_index = ids.index(measured_singles[0]["feature_id"])
    single_id = ids[single_index]
    single_g = {
        "00": float(scores[full_state]),
        "01": float(scores[full_state ^ (1 << single_index)]),
    }
    single = {
        "feature": features[single_id],
        "state": full_state ^ (1 << single_index),
        "baseline_state": full_state,
        "state_meaning": {
            "00": "unmodified full-state reference",
            "01": "only this feature zeroed",
        },
        "g": single_g,
        "probabilities": probabilities(single_g, token_count),
        "delta_g": single_g["01"] - single_g["00"],
        "definition": "Only this selected feature is zeroed; the other nine and all features outside Top10 remain native",
    }
    single_ranking = [
        {
            "feature_id": fid,
            "g01": float(scores[full_state ^ (1 << index)]),
            "state": full_state ^ (1 << index),
        }
        for index, fid in enumerate(ids)
    ]
    single_ranking.sort(key=lambda row: -row["g01"])
    original_all = float(original["full_response"])
    return {
        "key": question["key"],
        "dataset": question["dataset"],
        "number": question["number"],
        "sample_id": question["sample_id"],
        "question": question["question"],
        "question_zh": question["question_zh"],
        "prompt": question["prompt"],
        "answer": question["answer"],
        "answer_zh": question["answer_zh"],
        "answer_token_count": token_count,
        "selected_feature_ids": ids,
        "max_interaction_pair": pair,
        "baseline_qualified_max_interaction_pair": qualified,
        "all_other_selected_retained_max_interaction_pair": retained,
        "max_g01_single": single,
        "single_ranking": single_ranking,
        "whiten_top_feature": features[whiten_id],
        "sar_filtered_whiten_top_feature": features[ids[0]],
        "absolute_activation_top_feature": features[activation_id],
        "max_pair_context_best_single": {
            "feature_id": pair["pair_ids"][0 if pair["g"]["10"] >= pair["g"]["01"] else 1],
            "g": max(pair["g"]["10"], pair["g"]["01"]),
        },
        "all_retained_g": original_all,
        "all_retained_probabilities_percent": {
            "mean_token": 100 * math.exp(original_all),
            "joint": 100 * math.exp(token_count * original_all),
        },
        "verification": completion,
        "source_run": question["source_run"],
    }


def fmt(value):
    if value != 0 and abs(value) < 0.000001:
        return f"{value:.6e}"
    return f"{value:.6f}"


def explain(feature):
    return f"**{feature['feature_id']}**：{feature['explanation_zh']}"


def render(question):
    pair = question["max_interaction_pair"]
    single = question["max_g01_single"]
    lines = [
        f"## {question['key']}：{question['question_zh']}",
        "",
        "**完整原始输入（含任务说明与选项）**",
        "",
        "```text",
        question["prompt"],
        "```",
        "",
        f"标准答案：{question['answer']}；中文：{question['answer_zh']}。答案token数T={question['answer_token_count']}。",
        "",
        f"### 最大I的pair：{pair['pair_ids'][0]} × {pair['pair_ids'][1]}；I={pair['interaction']:+.9f}",
        "",
        *[
            f"- {explain(f)}。activation={fmt(f['activation'])}，z={fmt(f['zscore'])}。"
            for f in pair["features"]
        ],
        "",
        "其他八个候选的固定背景："
        + "；".join(
            f"{fid}={'保留' if state else '清零'}" for fid, state in pair["background"].items()
        )
        + "。",
        "",
        "| 指标 | 00 | 10 | 01 | 11 | 11超过前三项最高值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| g（平均log概率） | "
        + " | ".join(fmt(pair["g"][s]) for s in ("00", "10", "01", "11"))
        + " | "
        + fmt(pair["g"]["11"] - max(pair["g"][s] for s in ("00", "10", "01")))
        + " |",
    ]
    for row in pair["probabilities"]:
        lines.append(
            f"| {row['metric']} | "
            + " | ".join(fmt(row["percent"][s]) + "%" for s in ("00", "10", "01", "11"))
            + f" | {fmt(row['change_percentage_points'])}个百分点 |"
        )
    lines += [
        "",
        f"### 最大g01的single：{explain(single['feature'])}",
        "",
        "Top10中仅清零这一项，其余九项保持原样；基线为全部保留。",
        "",
        f"g01={fmt(single['g']['01'])}；原始g={fmt(single['g']['00'])}；Δg={fmt(single['delta_g'])}。",
        "",
        "| 指标 | 原始全保留 | 仅清零这一项（g01状态） | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in single["probabilities"]:
        lines.append(
            f"| {row['metric']} | {fmt(row['percent']['00'])}% | {fmt(row['percent']['01'])}% | {fmt(row['change_percentage_points'])}个百分点 |"
        )
    lines += [
        "",
        "### 四种选择的特征数值对比",
        "",
        "| 选择方式 | 特征ID | 中文解释 | activation | zscore | 参考mean | 参考std |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    choices = [
        ("最大I：第一项", pair["features"][0]),
        ("最大I：第二项", pair["features"][1]),
        ("最大g01 single", single["feature"]),
        ("whiten全SAE第一", question["whiten_top_feature"]),
        ("non-whiten全SAE第一", question["absolute_activation_top_feature"]),
        ("SAR过滤后whiten第一", question["sar_filtered_whiten_top_feature"]),
    ]
    for name, f in choices:
        lines.append(
            f"| {name} | {f['feature_id']} | {f['explanation_zh']} | "
            + " | ".join(fmt(f[k]) for k in ("activation", "zscore", "mean", "std"))
            + " |"
        )
    lines += [""]
    return lines


def main(*, report_name="report"):
    plan = json.loads((OUTPUT / "plan.json").read_text())
    translations = json.loads((ROOT / "feature_explanations_zh.json").read_text())
    questions = [
        analyze(q, translations)
        for q in plan["questions"]
        if (OUTPUT / q["key"] / "complete.json").exists()
    ]
    document = {
        "model": plan["model"],
        "sae": plan["sae"],
        "layer": plan["layer"],
        "requested_questions": 50,
        "completed_questions": len(questions),
        "pair_metric": "g11-g10-g01+g00; largest signed actual interaction over all pairs and fixed backgrounds of original Top10",
        "whiten_ranking": "largest absolute zscore over all SAE features with positive reference std",
        "nonwhiten_ranking": "largest absolute native activation over all SAE features",
        "questions": questions,
    }
    write_json(OUTPUT / f"{report_name}.json", document)
    summary = {
        "completed_questions": len(questions),
        "datasets": dict(Counter(q["dataset"] for q in questions)),
        "absolute_activation_winner_counts": dict(
            Counter(str(q["absolute_activation_top_feature"]["feature_id"]) for q in questions)
        ),
        "whiten_winner_counts": dict(
            Counter(str(q["whiten_top_feature"]["feature_id"]) for q in questions)
        ),
        "whiten_equals_best_single_count": sum(
            q["whiten_top_feature"]["feature_id"] == q["max_g01_single"]["feature"]["feature_id"]
            for q in questions
        ),
        "max_pair_11_highest_count": sum(
            q["max_interaction_pair"]["probabilities"][0]["change_percentage_points"] > 0
            for q in questions
        ),
        "baseline_qualified_question_count": sum(
            q["baseline_qualified_max_interaction_pair"] is not None for q in questions
        ),
        "max_interaction_range": [
            min(q["max_interaction_pair"]["interaction"] for q in questions),
            max(q["max_interaction_pair"]["interaction"] for q in questions),
        ]
        if questions
        else [],
    }
    summary_name = "summary.json" if report_name == "report" else f"{report_name}_summary.json"
    write_json(OUTPUT / summary_name, summary)
    lines = [
        "# SAR全遍历：50道题的pair、single、whiten与non-whiten对比",
        "",
        f"Llama 3.1 8B Base，Llama-Scope L21，原100题参考统计与SAR过滤规则；五数据集各10题。目前完成{len(questions)}/50。",
        "",
        "每题固定原先过滤后的Top10；实测其全部1024种保留/清零组合，穷举45对×256种固定背景，共11,520个四格交互。按有符号I最大选取，不取绝对值。这里的最大值是Top10范围内的条件交互最大值；不是全32768维SAE，也不是原SAR回归系数。其他Top10以外的特征和重构残差保持原值。",
        "",
        "g为标准答案teacher forcing的平均token log概率，不含EOS。平均token概率=exp(g)；答案联合概率=exp(Tg)。11超过前三项最高值及single的清零后−原始均按各自概率尺度计算，单位为百分点。最大正I未必意味着11概率最高；表中保留真实数值。",
        "",
        "single按Top10中仅清零一项、其余九项保持原样时最大的g01选择。whiten按全SAE的|zscore|选第一，non-whiten按全SAE的|activation|选第一；两项均不施加SAR专属过滤。另列SAR过滤后whiten第一，方便核对候选来源。参考mean/std来自各数据集原100题所有prompt token，未设置std下限，std=0排除。",
        "",
        "特征解释来自原SAE对应的Neuronpedia发布解释，中文为翻译；这些语义标签本身不构成因果验证。未观测值不使用SAR代理模型填充。每题原始结果、重放校验、源SHA与新增测量断点保存在同名子目录。",
        "",
    ]
    lines += [
        "## 总览",
        "",
        "表内为特征ID及对应选择指标；各项中文解释和完整数值在下方逐题给出。",
        "",
        "| 题号 | 最大I的pair（I） | 最大g01的single（g01） | whiten第一（zscore） | non-whiten第一（activation） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q in questions:
        pair = q["max_interaction_pair"]
        single = q["max_g01_single"]
        white = q["whiten_top_feature"]
        native = q["absolute_activation_top_feature"]
        lines.append(
            f"| {q['key']} | {pair['pair_ids'][0]} × {pair['pair_ids'][1]}（{pair['interaction']:.6f}） | {single['feature']['feature_id']}（{single['g']['01']:.6f}） | {white['feature_id']}（{white['zscore']:.6f}） | {native['feature_id']}（{native['activation']:.6f}） |"
        )
    lines += [""]
    for q in questions:
        lines.extend(render(q))
    target = OUTPUT / f"{report_name}_zh.md"
    target.write_text("\n".join(lines))
    print(f"Saved {len(questions)}/50 questions to {target}")


if __name__ == "__main__":
    main()
