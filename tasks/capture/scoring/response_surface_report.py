"""Build the Chinese, per-question comparison from verified exhaustive scores."""

from __future__ import annotations

from collections import Counter
import json
import math

import numpy as np

from tasks.capture.scoring.response_surface_reproduction import (
    reference_run,
    OUTPUT,
    source_arrays,
    write_json,
)
from tasks.capture.scoring.reproduction_paths import question_directory
from tasks.capture.behavior import choose_pair, probabilities, rank_singles


def feature_rows(question, translations):
    features = {f["feature_id"]: dict(f) for f in question["active_features"]}
    reference = OUTPUT / "reference" / question["dataset"] / "capture_normalization.npz"
    if not reference.exists():
        reference = reference_run(question["dataset"]) / "metadata/capture_normalization.npz"
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
        feature["explanation_zh"] = translations.get(
            str(feature["feature_id"]), "No explanation in Chinese."
        )
    return features, whiten_id, activation_id


def analyze(question, translations):
    directory = question_directory(OUTPUT, question["key"])
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
    }


def fmt(value):
    if value != 0 and abs(value) < 0.000001:
        return f"{value:.6e}"
    return f"{value:.6f}"


def explain(feature):
    return f"**{feature['feature_id']}**:{feature['explanation_zh']}"


def render(question):
    pair = question["max_interaction_pair"]
    single = question["max_g01_single"]
    lines = [
        f"## {question['key']}:{question['question_zh']}",
        "",
        "**Full original input (including mission description and options))**",
        "",
        "```text",
        question["prompt"],
        "```",
        "",
        f"Standard answer:{question['answer']};Chinese:{question['answer_zh']}.Answers token Number T={question['answer_token_count']}.",
        "",
        f"### Max I It’s... pair:{pair['pair_ids'][0]} × {pair['pair_ids'][1]};I={pair['interaction']:+.9f}",
        "",
        *[
            f"- {explain(f)}.activation={fmt(f['activation'])},z={fmt(f['zscore'])}."
            for f in pair["features"]
        ],
        "",
        "Fixed background for the other eight candidates:"
        + ";".join(
            f"{fid}={'Reservations' if state else 'Zero'}"
            for fid, state in pair["background"].items()
        )
        + ".",
        "",
        "| Indicators | 00 | 10 | 01 | 11 | 11 More than the top three |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| g(Average log Probability) | "
        + " | ".join(fmt(pair["g"][s]) for s in ("00", "10", "01", "11"))
        + " | "
        + fmt(pair["g"]["11"] - max(pair["g"][s] for s in ("00", "10", "01")))
        + " |",
    ]
    for row in pair["probabilities"]:
        lines.append(
            f"| {row['metric']} | "
            + " | ".join(fmt(row["percent"][s]) + "%" for s in ("00", "10", "01", "11"))
            + f" | {fmt(row['change_percentage_points'])}Percentage |"
        )
    lines += [
        "",
        f"### Max g01 It’s... single:{explain(single['feature'])}",
        "",
        "Top10 Only one item is cleared and the remaining nine items remain as they are; the baseline is retained in their entirety.",
        "",
        f"g01={fmt(single['g']['01'])};Original g={fmt(single['g']['00'])};Δg={fmt(single['delta_g'])}.",
        "",
        "| Indicators | Original All Reserved | Just one thing.(g01 Status) | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in single["probabilities"]:
        lines.append(
            f"| {row['metric']} | {fmt(row['percent']['00'])}% | {fmt(row['percent']['01'])}% | {fmt(row['change_percentage_points'])}Percentage |"
        )
    lines += [
        "",
        "### Four-option numeric contrasts",
        "",
        "| Selection Method | Features ID | English | activation | zscore | References mean | References std |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    choices = [
        ("Max I:Number one.", pair["features"][0]),
        ("Max I:Second", pair["features"][1]),
        ("Max g01 single", single["feature"]),
        ("whiten All SAE Number one.", question["whiten_top_feature"]),
        ("non-whiten All SAE Number one.", question["absolute_activation_top_feature"]),
        ("SAR After Filtering whiten Number one.", question["sar_filtered_whiten_top_feature"]),
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
    translations = json.loads((OUTPUT / "feature_explanations_zh.json").read_text())
    questions = [
        analyze(q, translations)
        for q in plan["questions"]
        if (question_directory(OUTPUT, q["key"]) / "complete.json").exists()
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
        "# SAR All over.: 50 It’s a question. pair,single,whiten with non-whiten Contrast",
        "",
        f"Llama 3.1 8B Base,Llama-Scope L21,Original 100 QUESTION OF ELIMINATION AND SAR filter rules;for each of the five data sets 10_Other Organiser{len(questions)}/50.",
        "",
        "I’m gonna fix each one of them after it’s been filtered. Top10;All of them. 1024 Reservations/Zero-sum, all over the place. 45 Yeah.×256 A fixed background, common 11,520 Four-bar interactive. Press a symbol I Most popular, not absolute. The biggest value here is Top10 range of conditions to interact maximum; not all 32768 V SAE,It’s not the original. SAR regression factor. Other Top10 Maintenance of original values for features and restructuring debris other than.",
        "",
        "g To Standard Answers teacher forcing Average token log Probability, excluding EOS.Average token Probability=exp(g);Joint probability of answers=exp(Tg).11 More than the top three and single After zero.−Originals are calculated at their respective probability scales in percentage points. Maximum positive I It doesn’t necessarily mean... 11 highest probability; retaining true values in tables.",
        "",
        "single Press Top10 Only one item was cleared and the remaining nine were the largest when they remained as they were. g01 Selection. whiten Press All SAE It’s...|zscore|Pick first., non-whiten Press All SAE It’s...|activation|Number one; neither is imposed SAR Exclusive filter. Separate column SAR After Filtering whiten First, it facilitates the verification of candidate sources. mean/std From original data sets 100 Threshold prompt token,Not set std Lower limit, std=0 Exclude.",
        "",
        "Characteristic interpretation from original SAE Corresponding Neuronpedia Issue an explanation; Chinese is translated; semantic labels do not themselves constitute cause or effect validation. No observations are used. SAR Proxy model fills. Original results, reruns, sources SHA Save with the new measurement breakpoint in the same name subdirectories.",
        "",
    ]
    lines += [
        "## Overview",
        "",
        "Features in Table ID and the corresponding selection indicator; each of the explanations and full values in Chinese is given below on a subject-by-issue basis.",
        "",
        "| Node | Max I It’s... pair(I) | Max g01 It’s... single(g01) | whiten Number one.(zscore) | non-whiten Number one.(activation) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q in questions:
        pair = q["max_interaction_pair"]
        single = q["max_g01_single"]
        white = q["whiten_top_feature"]
        native = q["absolute_activation_top_feature"]
        lines.append(
            f"| {q['key']} | {pair['pair_ids'][0]} × {pair['pair_ids'][1]}({pair['interaction']:.6f}) | {single['feature']['feature_id']}({single['g']['01']:.6f}) | {white['feature_id']}({white['zscore']:.6f}) | {native['feature_id']}({native['activation']:.6f}) |"
        )
    lines += [""]
    for q in questions:
        lines.extend(render(q))
    target = OUTPUT / f"{report_name}_zh.md"
    target.write_text("\n".join(lines))
    print(f"Saved {len(questions)}/50 questions to {target}")


if __name__ == "__main__":
    main()
