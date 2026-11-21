"""Export the fixed semantic-preservation case without substituting unmatched features."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value, signed=False):
    if value == 0:
        return "0"
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def explanation(row):
    return cell(
        "; ".join(e["description"] for e in row["annotation"]["explanations"])
        or f"No public explanation obtained({row['annotation'].get('status', 'unknown')})"
    )


def feature(row):
    return f"[F{row['feature_id']}]({row['annotation']['url']})"


def values(row):
    return " | ".join(number(x, True) for x in row["delta"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--review", type=Path, default=ROOT / "assets/reproduction_inputs/stable_support/files/semantic_preservation_review.json"
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    case = json.loads((root / "rss_case.json").read_text())
    review = json.loads(args.review.read_text())
    selected = {r["feature_id"]: r for r in case["rss_features"]}
    food = [selected[f] for f in review["selected_food_ids"]]
    context = []
    for v, f in enumerate(review["context_match_ids"]):
        context.append(
            None
            if f is None
            else next(r for r in case["context_candidates"][v]["features"] if r["feature_id"] == f)
        )
    input_data = case["inputs"]
    counts = [len(g["features"]) for g in case["context_candidates"]]
    positions = [a["capture_token_index"] for a in case["alignment"]]
    target_token_count = len(case["alignment"][0]["target_token_indices"])
    lines = [
        "# RSS Semantic retention case: Fixed banana bread assignment has no prefixes",
        "",
        "**" + review.get("headline", review["outcome"]) + "**",
        "",
        "## Full Actual Input",
        "",
        "Below is the full string sent to the model, and the empty lines in the code block are kept in their original language.",
        "",
        "Clean",
        "",
        "```text",
        input_data["clean"],
        "```",
        "",
    ]
    for variant in input_data["variants"]:
        lines += [
            f"V{variant['variant']} · {variant['topic']}",
            "",
            "```text",
            variant["prompt"],
            "```",
            "",
        ]
    lines += [
        "## Experiment settings and alignment",
        "",
        "SAEBase v0.7.0;Llama 3.1 8B Base;llama31_8b_lxr_8x;layer 21;blocks.21.hook_resid_post;32,768 features.Same loaded model with SAE Capture the fullness of six entries native codes,None SAR Filtering or Standardization.",
        "",
        "RSS Use Open Original Selector, M=400,K=20,rho=kappa=0.8,tau=1e-5.Local File Git blob Yes. 08afc3929c26b3c0b3d94a629e99bae03f0e5d93,This is exactly the same as the public original document that was obtained. All value selections are determined and read later. Neuronpedia Explanation.",
        "",
        f"The text of the target sentence in the six inputs is identical and the target contains{target_token_count}One. token.The actual catch is the end of the period; 0 Absolute location of count by clean,V1–V5 Order is{positions}.Five. variants Keeps two line breaks between the prefix and the target., token Index and Input Alignment Full Fact Save In rss_case.json with run.json.",
        "",
        "## Single Consolidated Results Table",
        "",
        review.get(
            "table_note",
            "Select the rows to be displayed according to the current selection and original interpretation. The missing match clearly leaves empty and is not replaced by unrelated features. Non-zero changes show five decimal places, bottom JSON/NPZ Retain original precision.",
        ),
        "",
        "| RSS status | Feature | Clean activation | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in food:
        lines.append(
            f"| Selected {'↑' if row['direction'] > 0 else '↓'} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} |"
        )
    for v, row in enumerate(context):
        topic = input_data["variants"][v]["topic"]
        if row is None:
            lines.append(
                f"| Excluded {topic}:No semantic matching found | — | — | — | — | — | — | — | — | — |"
            )
        else:
            lines.append(
                f"| Excluded {topic} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} |"
            )
    lines += ["", "## Scientific interpretation", ""]
    titles = [
        "1. RSS Whether the feature is associated with the reserved cooking task?",
        "2. Whether an increase or decrease reflects the same information can explain the redistribution?",
        "3. Whether the excluded feature corresponds to each of the additional unrelated information?",
    ]
    for title, text in zip(titles, review["questions"]):
        lines += [f"### {title}", "", text, ""]
    lines += [
        "### This is a supportable conclusion.",
        "",
        review["conclusion"],
        "",
        "## Full Record Entry",
        "",
        "- [All 20 Article RSS Organisation](all_selected_features.md)",
        "- [Front 400 All the strict single variable candidates and original explanations in the pool](context_candidates.md)",
        "- [Complete input, feature, original explanation, alignment and source](rss_case.json)",
        "- [Six complete activated vectors and delta](rss_activations.npz)",
        "- [Model/SAE Identity and Configuration](config.yaml) · [Run Records](run.json)",
    ]
    (root / "rss_case.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# This time it’s raw. RSS All Selected 20 Article characteristics",
        "",
        "Retain the selected feature in the order in which the original selector returns, and no selection feature is deleted on the basis of an explanation. The direction is beyond tau..the dominant symbol cannot replace every one. variant Actual delta.",
        "",
        "| Feature | Clean activation | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | RSS direction | Consistency | Original Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in case["rss_features"]:
        lines.append(
            f"| {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {'+1 ↑' if row['direction'] > 0 else '-1 ↓'} | {row['consistency']:.1f} | {explanation(row)} |"
        )
    lines += [
        "",
        "Every one Feature Links all point to their original Neuronpedia page. Full values, all explanatory metadata and original sources are available rss_case.json Medium.",
    ]
    (root / "all_selected_features.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# All Strict Background Candidates and Unmatched Results",
        "",
        "I’ll be right back. RSS Before the average absolute change 400 Pool filter: Not available RSS Selection, Correlation variant It’s... delta>1e-5,The other four|delta|<=1e-5.Each recurrence Both. 0.2,It’s clear because... 0.2<rho=0.8 Excluded. Explanation in candidate ID Read after confirmation.",
        "",
        f"Numerical Candidates are in order{counts},Total{sum(counts)}The value single variable response itself does not guarantee that the explanation matches the corresponding background..",
        "",
        "| Variant | RSS status | Feature | Clean activation | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | Exclusion reason | Original Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in case["context_candidates"]:
        for row in group["features"]:
            lines.append(
                f"| V{group['variant']} {group['topic']} | Excluded | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | recurrence_below_rho(Yes. M=400 Ichichi.) | {explanation(row)} |"
            )
    lines += [
        "",
        review.get("context_summary", "Numerical uniqueness and semantic match to separate judgement; candidate not selected in the main list remains all."),
    ]
    (root / "context_candidates.md").write_text("\n".join(lines) + "\n")
    case["qualitative_review"] = review
    case["main_table_selected_feature_ids"] = [r["feature_id"] for r in food]
    case["main_table_context_feature_ids"] = review["context_match_ids"]
    (root / "rss_case.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    run = json.loads((root / "run.json").read_text())
    run.update(
        qualitative_report="rss_case.md",
        qualitative_outcome=review["outcome"],
        context_semantic_match_count=sum(r is not None for r in context),
        report_complete=True,
    )
    (root / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    archive = root.parent / f"{root.name}_reports.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for name in ["rss_case.md", "all_selected_features.md", "context_candidates.md"]:
            z.write(root / name, name)
    print(root / "rss_case.md")
    print(archive)


if __name__ == "__main__":
    main()
