"""Build eight literal-input qualitative reports from recorded features and explicit reviews."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.capture.rss_mode import feature_row

TOPICS = ["Basketball", "Space", "Music", "Chess", "Animals"]
HEADER = "| Status | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | Neuronpedia explanation |"
RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    inputs = json.loads((ROOT / "assets/reproduction_inputs/stable_support/files/qualitative_batch_inputs.json").read_text())
    review = json.loads((ROOT / "assets/reproduction_inputs/stable_support/files/qualitative_batch_review.json").read_text())
    write_json(root / "inputs.json", inputs)
    write_json(root / "review_decisions.json", review)
    cases = []
    for i, (inp, decision) in enumerate(zip(inputs["cases"], review["cases"]), 1):
        directory = root / f"case_{i:02d}"
        captured = json.loads((directory / "rss_case.json").read_text())
        candidates = json.loads((directory / "context_candidates.json").read_text())
        selected = {r["feature_id"]: r for r in captured["rss_features"]}
        target = [selected[f] for f in decision["rss_ids"]]
        contexts = []
        for v, f in enumerate(decision["context_ids"]):
            contexts.append(
                None
                if f is None
                else next(r for r in candidates["groups"][v]["features"] if r["feature_id"] == f)
            )
        match_count = sum(r is not None for r in contexts)
        lines = [
            f"# {inp['case_id']} {decision['title_zh']} · {inp['title']}",
            "",
            "## Full Input and Experiment Settings",
            "",
            f"Clean:{inp['clean']}",
            "",
        ]
        for variant in inp["variants"]:
            lines += [f"V{variant['number']} · {variant['topic']}:{variant['prompt']}", ""]
        last_ids = [ids[-1] for ids in captured["token_ids"]]
        lines += [
            "Model: Llama 3.1 8B Base;SAE:llama31_8b_lxr_8x;layer 21;hook:blocks.21.hook_resid_post;32,768 features.Capture the last of every complete input token(End period.), position=-1.",
            "",
            f"Actual final token IDs(clean,V1–V5):{last_ids}.RSS Use Original Selector, M=400,K=20,rho=kappa=0.8,tau=1e-5;Full raw codes,No semantic prefiltration.",
            "",
            "## Merge Feature Tables",
            "",
            "The following table retains an explanatory value: RSS Lines, and clearly label adjacent or broad labels. Do not treat semantics as identified target characteristics when they are inadequate; no line with clear background matches displays a gap, and no eight entries are filled with irrelevant features. 20 Article RSS See the original end of the text with all the values..",
            "",
            HEADER,
            RULE,
        ]
        for label, row in zip(decision["rss_labels"], target):
            lines.append(f"| {label} | " + feature_row(row)[2:])
        for v, row in enumerate(contexts):
            if row is None:
                lines.append(
                    f"| RSS excluded · V{v + 1} {TOPICS[v]}:No clear match found | — | — | — | — | — | — | — | — |"
                )
            else:
                lines.append(f"| RSS excluded · V{v + 1} {TOPICS[v]} | " + feature_row(row)[2:])
        lines += [
            "",
            "## Scientific interpretation and judgement",
            "",
            decision["interpretation"],
            "",
            decision["context_note"],
            "",
            f"**Objective relevance:** {decision['target_finding']}",
            "",
            f"**Directional relations:** {decision['opposition']}",
            "",
            f"**Background Match:** {match_count}/5.The figure indicates only how many contexts in this qualitative table have been clearly explained, not performance assessments..",
            "",
            f"**Show applicability:** {decision['suitability']}",
            "",
            "**User’s explanation pending examination:** "
            + inp["proposed_interpretation_to_inspect"]
            + " The proposal was not considered as known fact in this instalment.",
            "",
        ]
        if match_count:
            lines += ["### The exclusion basis of the background feature has been demonstrated", ""]
            for v, row in enumerate(contexts):
                if row:
                    lines.append(
                        f"- V{v + 1} F{row['feature_id']}:in_top_M={row['in_top_M']},recurrence={row['recurrence']:.1f},consistency={row['consistency']:.1f};Exclusions:{', '.join(row['exclusion_reasons'])}."
                    )
            lines += [""]
        lines += [
            "## Full original records",
            "",
            "- [All 20 Article RSS With All context Can not open message](all_candidates.md)",
            "- [Full clean/variants/delta Numeric](rss_activations.npz)",
            "- [Full Dictionary RSS Statistics and eligibility for election](rss_feature_statistics.npz)",
            "- [Original Capture Results and all 20 Article RSS](rss_case.json)",
            "- [All values have specific candidates and reasons for exclusion](context_candidates.json)",
            "- [Actual Configuration](config.yaml) · [Official mission return](task_result.json)",
            "- [This example shows the line and the manual reading judgement.](qualitative_case.json)",
        ]
        (directory / "case_report.md").write_text("\n".join(lines) + "\n")
        all_lines = [
            f"# {inp['case_id']} Check all candidates.",
            "",
            "## Original RSS All Selected Features",
            "",
            HEADER,
            RULE,
        ]
        for row in captured["rss_features"]:
            all_lines.append(
                f"| RSS {'↑' if row['direction'] > 0 else '↓'} | " + feature_row(row)[2:]
            )
        all_lines += [
            "",
            "## All values are candidates.",
            "",
            candidates["candidate_rule"],
            "",
            "Candidates are determined by activation and are publicly explained only for reading; a candidate is not a semantic acceptance. strict-recurrence And the candidate is still on the list. rss_case.json.",
            "",
        ]
        for v, group in enumerate(candidates["groups"]):
            all_lines += [
                f"### V{v + 1} {TOPICS[v]}",
                "",
                HEADER + " Exclusion reason |",
                RULE + " --- |",
            ]
            for row in group["features"]:
                reason = ", ".join(row["exclusion_reasons"])
                all_lines.append(f"| RSS excluded | " + feature_row(row)[2:] + f" {reason} |")
            if not group["features"]:
                all_lines += ["", "No character that meets the numerical candidacy criteria."]
            all_lines += [""]
        (directory / "all_candidates.md").write_text("\n".join(all_lines) + "\n")
        payload = {
            "case": inp,
            "review": decision,
            "context_match_count": match_count,
            "display_rss_rows": target,
            "display_context_rows": contexts,
            "capture_position": captured["capture_position"],
            "last_token_ids": last_ids,
            "identity": captured["identity"],
            "rss_parameters": captured["rss_parameters"],
            "neuronpedia": candidates["neuronpedia"],
        }
        write_json(directory / "qualitative_case.json", payload)
        cases.append(
            {
                "case_id": inp["case_id"],
                "title": decision["title_zh"],
                "target": decision["target_finding"],
                "opposition": decision["opposition"],
                "context_match_count": match_count,
                "suitability": decision["suitability"],
                "priority": decision["priority"],
                "report": f"case_{i:02d}/case_report.md",
                "rss_count": len(captured["rss_features"]),
                "capture_count": len(captured["token_ids"]),
            }
        )
    summary = [
        "# Eight. RSS Comparison of qualitative cases",
        "",
        "Assigned 8 Groups literal inputs Implemented: Llama 3.1 8B Base / Llama Scope I don’t think so. 21 Layer 32k SAE,Yes. clean With five variants The end of the catch. All right. 48 bar. Every case is saved in full. codes and delta,And original. RSS Actual election 20 Article characteristics. All of them. 8 Fill in reading judgement and sorting after example has been completed.",
        "",
        "**Cases in which three full presentation criteria were not met at the same time.** Some math., AI The responsibilities or security adjacent features are worth manual access, but have not yet been identified by two or three mechanisms, with a consistent two-way relationship, or a complete combination of five types of background in any one case.",
        "",
        "## Eight comparisons.",
        "",
        "| Cases | Goal-related RSS Features | Is there a consistent explanation for the opposite direction? | Clear Background Matches | OpenReview Applicability |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cases:
        summary.append(
            f"| [{c['case_id']} {c['title']}]({c['report']}) | {c['target']} | {c['opposition']} | {c['context_match_count']}/5 | {c['suitability']} |"
        )
    summary += [
        "",
        "## It’s recommended that we look first.",
        "",
        "1. **C06 Evidence and quotations**:AI Declined responsible use features, increased document format features, presence of content/The form of a reading trail; however, the public label cannot yet be specifically interpreted as evidence reliability and reference surface form. Background matches only one item in space..",
        "2. **C01 Decimal Comparison**:The two mathematical conceptual features have dropped together in five contexts; this is a clearer mathematical theme, but there is no distinction between decimal sizes, digital surface forms, or a two-way combination within the target theme.",
        "3. **C02 It’s safe.**:Background scores/The character of the game and space travel are the same. variant Response, clear basis for exclusion; only adjacent hackers on the target side/Malicious software tags, no substitute. API Evidence-based exposure mechanisms.",
        "",
        "These are the clues that this batch is worth reviewing, not recommended for entry. rebuttal. This is not the final draft. rebuttal.",
        "",
        "## How to read these judgments?",
        "",
        "The eight potential contrasts given by the user are retained as assumptions to be checked. The characteristic is explained as an open label and cannot be inferred from the proximity of the label or from the addition or decrease of the value only that the specific mechanism has been identified. C03,C07,C08 The consolidated table clearly labels a broad reference line as an unrecognized target and does not include it as a qualifying case.",
        "",
        "All background values candidate complete before searching for explanation delta Confirmed: Not RSS Selection, positive change in this variant>tau,Keep all eligible candidates in the order of magnitude of this variant. recurrence Failed to display as a priority condition, while recording other possible exclusions. Three actual contexts match(C02 Two., C06 It’s all here. M Candidatures, recurrence=0.2,The other four delta Zero, clear cause recurrence Insufficiently ruled out.",
        "",
        "No fine-tuning specified in this batch prompts,Model, layer or RSS Parameters to pursue preset explanations; the range of judgment is the measured activation and public interpretation of the end of these eight sets.",
        "",
        "## The product entrance.",
        "",
        "- [All original input and explanation to be checked](inputs.json)",
        "- [Individual AI Reading judgement](review_decisions.json) · [Machine-readable comparative summary](comparison.json)",
        "- [Original explanation and source of the characteristics of the instalment](batch_annotations.json)",
        "- [Run Parameters & Complete Status](run.json)",
        "- Every one of them. case_report.md Contains full input, a consolidated table, short scientific explanation and links to all original records.",
    ]
    (root / "summary.md").write_text("\n".join(summary) + "\n")
    write_json(
        root / "comparison.json",
        {
            "status": "eight_cases_measured_and_reviewed",
            "case_count": len(cases),
            "input_count": sum(c["capture_count"] for c in cases),
            "full_scientific_example_found": False,
            "cases": cases,
            "manual_review_order": ["C06", "C01", "C02"],
            "final_rebuttal_drafted": False,
        },
    )
    (root / "index.md").write_text(
        "# Eight qualitative cases\n\n[First look: eight comparative summaries and manual review recommendations](summary.md)\n\n"
        + "\n".join(f"- [{c['case_id']} {c['title']}]({c['report']})" for c in cases)
        + "\n"
    )
    print(root / "summary.md")


if __name__ == "__main__":
    main()
