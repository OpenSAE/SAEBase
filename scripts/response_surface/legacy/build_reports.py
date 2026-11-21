"""Export the five completed QA experiments and verify the displayed evidence.

Scope: produce complete bilingual questions and Chinese single/pair comparisons
from formal Capture reports, retaining original inputs and canonical readouts.
"""

import hashlib
import json
from pathlib import Path


import numpy as np

from tasks.capture.task import CaptureTask


ROOT = Path("/workspace/SAEBase")
REPORT = Path(__file__).resolve().parent
TAG = "reference_additivity"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dataset_report(
    name, path, translations, question_translations, planned_ids=None, previous=False
):
    CaptureTask().export_report(path)
    payload = read(path / "reports/question_features_pairs.json")
    rows = {r["sample_id"]: r for r in CaptureTask().read_samples(path)}
    cohort = read(path / "metadata/sar_cohort_filter.json")
    semantic = read(path / "metadata/sar_feature_filter.json")
    blocked = set(cohort["blocked_ids"]) | set(map(int, semantic["blocked_features"]))
    assert len(rows) == 10 and all(r["status"] == "completed" for r in rows.values())
    assert cohort["identity"]["question_count"] == 100
    assert payload["pair_ranking"] == "signed_pairwise_coefficient_descending"
    if planned_ids is not None:
        assert [q["sample_id"] for q in payload["questions"]] == planned_ids
    localized = {q["sample_id"]: q for q in question_translations}
    lines = [
        f"# {name}:Full title, single feature and SAR Match",
        "",
        "100 Subject reference; 10 Question SAR;Llama 3.1 8B I don’t think so. 21 Layer SAE."
        "References end words z>1 Percentage of total 50%is excluded and filtered using original syntax and primary density."
        "single Press|z|Descending, pair Allows the endpoint to be repeated in order of the symbol coefficient.",
        "",
    ]
    visual_questions = []
    for index, q in enumerate(payload["questions"], 1):
        row = rows[q["sample_id"]]["result"]
        assert row["layer"] == 21
        canonical = row["sar"]["identity"]["behavior"]["reference"]["text"]
        q.update(localized[q["sample_id"]])
        q["source_answer"] = q["answer"]
        q["answer"] = canonical
        assert len(q["features"]) == 10 and len(q["pairs"]) == 5
        ids = {f["feature_id"] for f in q["features"]}
        assert not ids & blocked
        for artifact in row["required_artifacts"]:
            assert (
                hashlib.sha256((path / artifact["path"]).read_bytes()).hexdigest()
                == artifact["sha256"]
            )
        for f in q["features"]:
            f["explanation_zh"] = translations[str(f["feature_id"])]
            f["reference_z_exceedance_count"] = cohort["exceedance_counts"][f["feature_id"]]
            assert f["reference_z_exceedance_count"] < 50
        for pair in q["pairs"]:
            assert set(pair["feature_ids"]).issubset(ids)
            pair["explanation_zh"] = " × ".join(translations[str(k)] for k in pair["feature_ids"])
        with np.load(path / q["sar_artifact"]["path"]) as arrays:
            expected = sorted(arrays["pair_coefficients"].tolist(), reverse=True)[:5]
        assert np.allclose([p["coefficient"] for p in q["pairs"]], expected, atol=1e-12, rtol=0)
        lines += [
            f"## {index}．{q['question_zh']}",
            "",
            f"Dataset indicates the answer (translation)):{q['answer_zh']}",
            f"Actual scoring reference text:{canonical}",
            "",
            "**Full original input (with options and mission statement))**",
            "",
            "````text",
            q["prompt"],
            "````",
            "",
            "| Single ID | English | Raw activation | Z-score | Number of references above threshold |",
            "|---|---|---:|---:|---:|",
        ]
        for f in q["features"]:
            lines.append(
                f"| {f['feature_id']} | {f['explanation_zh']} | {f['activation']:.5f} | "
                f"{f['zscore']:.3f} | {f['reference_z_exceedance_count']}/100 |"
            )
        lines += ["", "| Pair | It’s a two-way Chinese explanation. | Interactive coefficient |", "|---|---|---:|"]
        for pair in q["pairs"]:
            pair_ids = " × ".join(map(str, pair["feature_ids"]))
            lines.append(f"| {pair_ids} | {pair['explanation_zh']} | {pair['coefficient']:+.5f} |")
        lines += ["", ";".join(f"{k}={q['sar_metrics'][k]:.6g}" for k in ("sar1", "sar2")), ""]
        visual_questions.append(
            {
                "id": q["sample_id"],
                "question": q["question"],
                "question_zh": q["question_zh"],
                "prompt": q["prompt"],
                "answer": canonical,
                "answer_zh": q["answer_zh"],
                "features": [
                    {
                        "id": f["feature_id"],
                        "text": f["explanation_zh"],
                        "raw": f["activation"],
                        "z": f["zscore"],
                        "frequency": f["reference_z_exceedance_count"],
                    }
                    for f in q["features"]
                ],
                "pairs": [
                    {
                        "ids": p["feature_ids"],
                        "text": p["explanation_zh"],
                        "coefficient": p["coefficient"],
                    }
                    for p in q["pairs"]
                ],
                "sar1": q["sar_metrics"]["sar1"],
                "sar2": q["sar_metrics"]["sar2"],
            }
        )
    payload["cohort_filter"] = cohort["identity"]
    write(path / "reports/question_features_pairs_zh.json", payload)
    (path / "reports/full_questions_singles_pairs_zh.md").write_text("\n".join(lines) + "\n")
    meta = {
        "dataset": name,
        "previous": previous,
        "completed": 10,
        "reference": 100,
        "frequency_blocked": len(cohort["blocked_ids"]),
        "additional_blocked": len(
            set(cohort["blocked_ids"]) - set(map(int, semantic["blocked_features"]))
        ),
        "combined_blocked": len(blocked),
        "wrote_frequency": cohort["exceedance_counts"][10723],
        "wrote_blocked": 10723 in blocked,
        "sar2_better_questions": sum(q["sar2"] < q["sar1"] for q in visual_questions),
        "questions": visual_questions,
    }
    return meta, "\n".join(lines)


def main():
    translations = read(REPORT / "feature_explanations_zh.json")
    questions = read(REPORT / "question_translations_zh.json")
    plan = read(REPORT / "plan.json")
    groups, documents = [], []
    for item in plan.values():
        name = item["dataset"]
        path = ROOT / "experiment" / name / "llama31_8b/capture" / TAG
        group, document = dataset_report(
            name, path, translations, questions[name], [q["sample_id"] for q in item["targets"]]
        )
        groups.append(group)
        documents.append(document)
    summaries = [{k: v for k, v in g.items() if k != "questions"} for g in groups]
    write(REPORT / "summary.json", summaries)
    write(REPORT / "five_datasets_results_zh.json", groups)
    (REPORT / "five_datasets_full_report_zh.md").write_text("\n\n".join(documents) + "\n")
    previous_path = (
        ROOT / "experiment/HotpotQA/llama31_8b/capture/reference_additivity"
    )
    previous_payload = read(previous_path / "reports/question_features_pairs_zh.json")
    previous_group, _ = dataset_report(
        "HotpotQA", previous_path, translations, previous_payload["questions"], previous=True
    )
    groups.append(previous_group)
    write(REPORT / "viewer_data.json", groups)
    write(
        REPORT / "verification.json",
        {
            "new_datasets": 5,
            "new_measured_questions": 50,
            "reference_questions_per_dataset": 100,
            "single_rows": 500,
            "pair_rows": 250,
            "all_selected_features_pass_both_filters": True,
            "all_artifact_hashes_verified": True,
            "signed_pair_rankings_verified": True,
            "all_fifty_full_original_prompts_present": True,
            "chinese_feature_explanations_complete": True,
            "previous_hotpotqa_questions_for_comparison": 10,
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
