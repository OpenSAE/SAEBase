"""Freeze the existing five-dataset, fifty-question SAR inputs for enumeration."""

import json
import hashlib


from tasks.capture.scoring.response_surface_reproduction import OUTPUT, reference_run, source_arrays, write_json


def main():
    OUTPUT.mkdir(exist_ok=True)
    questions = []
    references = []
    for dataset in ("GSM8K", "MMLUPro", "ARCChallenge", "TruthfulQA", "BBH"):
        run = reference_run(dataset)
        report = json.loads((run / "reports/question_features_pairs_zh.json").read_text())
        reference = OUTPUT / "reference" / dataset
        reference.mkdir(parents=True, exist_ok=True)
        for name in (
            "capture_normalization.npz",
            "capture_normalization.json",
            "sar_feature_filter.json",
            "sar_cohort_filter.json",
            "config.yaml",
        ):
            source = run / "metadata" / name
            if source.exists():
                content = source.read_bytes()
                (reference / name).write_bytes(content)
                references.append(
                    {
                        "dataset": dataset,
                        "filename": name,
                        "local": str((reference / name).relative_to(OUTPUT)),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        records = {
            row["sample_id"]: row["result"]
            for row in map(
                json.loads, (run / "reports/capture_samples.jsonl").read_text().splitlines()
            )
        }
        assert len(report["questions"]) == 10
        for number, q in enumerate(report["questions"], 1):
            record = records[q["sample_id"]]
            question = {
                "dataset": dataset,
                "number": number,
                "sample_id": q["sample_id"],
                **{
                    key: q[key]
                    for key in ("question", "question_zh", "prompt", "answer", "answer_zh")
                },
                "artifact": q["sar_artifact"],
                "sar_identity": record["sar"]["identity"],
                "feature_count": record["feature_count"],
                "active_features": record["active_features"],
                "top_features": q["features"],
                "native_input_identity": record["input_identity"],
                "key": f"{dataset}-{number:02d}",
            }
            source_arrays(question)
            questions.append(question)
    OUTPUT.mkdir(exist_ok=True)
    write_json(OUTPUT / "reference_sources.json", references)
    write_json(
        OUTPUT / "plan.json",
        {
            "model": "llama31_8b",
            "sae": "llama31_8b_lxr_8x",
            "layer": 21,
            "seed": 42,
            "questions": questions,
        },
    )
    print(f"Frozen {len(questions)} questions")


if __name__ == "__main__":
    main()
