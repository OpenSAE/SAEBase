"""Export the fifty verified questions into the user's eleven-column CSV."""

import csv
import json
from pathlib import Path


HEADERS = [
    "完整原题",
    "标准答案",
    "whiten top1",
    "non-whiten top1",
    "g(S) top 1",
    "interaction top 1",
    "whiten z score",
    "non whiten activation",
    "g(S)",
    "delta g(S)",
    "I",
]


def feature_label(feature):
    return f"{feature['feature_id']}｜{feature['explanation_zh']}"


def main():
    directory = Path(__file__).resolve().parent / "results"
    document = json.loads((directory / "report.json").read_text())
    assert document["completed_questions"] == 50
    rows = []
    for question in document["questions"]:
        white = question["whiten_top_feature"]
        raw = question["absolute_activation_top_feature"]
        single = question["max_g01_single"]
        pair = question["max_interaction_pair"]
        prompt = question["prompt"].rstrip()
        if prompt.endswith("Answer:"):
            prompt = prompt[: -len("Answer:")].rstrip()
        pair_label = f"{pair['pair_ids'][0]} × {pair['pair_ids'][1]}｜" + "；".join(
            f"{feature['feature_id']}：{feature['explanation_zh']}" for feature in pair["features"]
        )
        rows.append(
            [
                prompt,
                question["answer"],
                feature_label(white),
                feature_label(raw),
                feature_label(single["feature"]),
                pair_label,
                white["zscore"],
                raw["activation"],
                single["g"]["01"],
                single["delta_g"],
                pair["interaction"],
            ]
        )
    target = directory / "50_questions_four_top1.csv"
    # BOM lets Excel recognize Chinese; csv.writer quotes multiline questions.
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.reader(stream))
    assert saved[0] == HEADERS and len(saved) == 51
    assert all(len(row) == 11 for row in saved)
    assert saved[1:] == [[str(value) for value in row] for row in rows]
    print(f"Saved {len(rows)} questions × {len(HEADERS)} columns: {target}")


if __name__ == "__main__":
    main()
