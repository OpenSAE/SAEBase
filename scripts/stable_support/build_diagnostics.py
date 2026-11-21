"""Export all selected feature rows and bounded qualitative interpretations for E1–E4."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def descriptions(row):
    return "; ".join(e["description"] for e in row["annotation"]["explanations"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    review = json.loads((ROOT / "assets/reproduction_inputs/stable_support/files/diagnostic_review.json").read_text())
    (root / "review_decisions.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n"
    )
    summaries = []
    for decision in review["cases"]:
        eid = decision["id"]
        directory = root / eid
        case = json.loads((directory / "rss_case.json").read_text())
        inp = case["diagnostic_case"]
        lines = [
            f"# {eid} {decision['title_zh']}",
            "",
            f"Research issues:{' '.join(inp['scientific_question'].split())}",
            "",
            "## Full Input",
            "",
            "The following code block retains the internal line break that actually sent to the model.",
            "",
            "Clean",
            "",
            "```text",
            inp["clean"],
            "```",
            "",
        ]
        for v, prompt in enumerate(inp["variants"], 1):
            lines += [f"V{v}", "", "```text", prompt, "```", ""]
        lines += [
            "## Experiment Settings and Complete RSS Table",
            "",
            "Llama 3.1 8B Base;SAE llama31_8b_lxr_8x;layer 21;blocks.21.hook_resid_post;32,768 features.Catch the last full hint token(Target question end) , use full original SAE codes.",
            "",
            f"Actual end token(clean,V1–V5):{case['captured_last_tokens']}.All positions.-1.",
            "",
            "Original RSS Parameters: M=400,K=20,rho=kappa=0.8,tau=1e-5.Delta Yes. variant Less clean.In explanation. RSS. The following table maintains all selected features in their entirety, as is the case. RSS Order.",
            "",
            "| Feature ID | Clean activation | Delta V1 | Delta V2 | Delta V3 | Delta V4 | Delta V5 | Recurrence | RSS direction | Original Neuronpedia explanation | Neuronpedia URL |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in case["rss_features"]:
            fid = row["feature_id"]
            url = row["annotation"]["url"]
            vals = " | ".join(f"{x:+.6g}" for x in row["delta"])
            lines.append(
                f"| [F{fid}]({url}) | {row['clean_activation']:.6g} | {vals} | {row['recurrence']:.1f} | {'+1 ↑' if row['direction'] > 0 else '-1 ↓'} | {cell(descriptions(row))} | [Neuronpedia]({url}) |"
            )
        rows = {r["feature_id"]: r for r in case["rss_features"]}
        focus = [rows[f] for f in decision["focus_ids"]]
        lines += [
            "",
            "Recurrence Statistical non-zero change rate; RSS direction It’s over. tau. The original algorithm allows four of the five to be the same, with the exception of a table delta Reference.",
            "",
            "## Best-known candidate syntax relationship (up to three))",
            "",
            "Select this paragraph:"
            + ", ".join(f"F{f}" for f in decision["focus_ids"])
            + ".It’s an afterthought of the whole selection. It doesn’t change. RSS.",
            "",
        ]
        for i, paragraph in enumerate(decision["interpretation"], 1):
            lines += [f"{i}. {paragraph}", ""]
        lines += [
            f"**Specific useful rules/Whether or not the action appears:** {decision['useful_specific_rule']}",
            "",
            f"**Author reviews recommendations:** {decision['verdict']}",
            "",
            "## Input Range and Original Record",
            "",
            "This round is followed. literal Enter a line break within the retention attachment. The target question sequences and points of the variants are consistent, but the layout line change position is not fully consistent; therefore the observations contain these word change differences and cannot be attributed solely to new misleading information or exposure channels. No additional regularization, answer generation or quantitative evaluation is carried out.",
            "",
            "- Full raw clean/variants/delta:[rss_activations.npz](rss_activations.npz)",
            "- All 20 Article RSS,Original interpretation, direction, token IDs With full source:[rss_case.json](rss_case.json)",
            "- Candidature Explanations and 3 Shows the attention line:[interpretation.json](interpretation.json)",
            "- Model/SAE Identity, precision and software version:[config.json](../config.json)",
            "- All Actual Inputs:[inputs.json](../inputs.json)",
        ]
        (directory / "case_report.md").write_text("\n".join(lines) + "\n")
        payload = {
            "review": decision,
            "focus_rows": focus,
            "input_case": inp,
            "interpretation_status": "post_selection_qualitative_reading",
            "rss_selection_count": len(case["rss_features"]),
        }
        (directory / "interpretation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        summaries.append(
            {
                "id": eid,
                "title": decision["title_zh"],
                "finding": decision["finding"],
                "specific_rule": decision["useful_specific_rule"],
                "verdict": decision["verdict"],
                "priority": decision["priority"],
                "report": f"{eid}/case_report.md",
            }
        )
    lines = [
        "# E1–E4 Comparison of diagnostic cases",
        "",
        "All four groups are as follows: literal Enter execution, total 24 At the end of the second token Capture, complete and original for each group RSS Selected 20 bar feature. Model, SAE And the original RSS Parameters fixed, no answers generated, no quantification benchmark.Qualitative comparison of all four groups after completion.",
        "",
        "**E1 It is the relatively best lead to examine by the author; none of the four groups explicitly supports the prescribed specific correct rules, mechanisms or movement of actions.**",
        "",
        "| Cases | The actual semantic relationship of the added or subtracted feature | Whether the additioner contains more specific and useful rules | Author reviews judgement |",
        "| --- | --- | --- | --- |",
    ]
    for c in summaries:
        lines.append(
            f"| [{c['id']} {c['title']}]({c['report']}) | {c['finding']} | {c['specific_rule']} | {c['verdict']} |"
        )
    lines += [
        "",
        "## What does the author read first?",
        "",
        "Look first. E1 It’s... F21772(Mathematical functions and concepts, leading to reduction), F9736(Mathematics certificate, add)F29126(Logo or causal explanation, add. Its label supports the mathematical theme of coarse particle size./Differences in the way the argument is argued, but the reduction does not make it clear that the error is inspired, nor does the addition recognize the decimal bits or the correct comparison rule. The set of changes cannot be written as a proven ”from misleading to correct reasoning.””.",
        "",
        "E2 The increase is for general technical support questions and answers, not for revocation or rotation; E3 The increase is a broad mathematical concept, not a common cause or a mixed reasoning.; E4 The increase is in the configuration of objects and classes, not in an orderly order. The three cases are retained as unexplained cases and do not give them higher task value by giving them direction..",
        "",
        "Every report contains all of them. 20 bar instead of an extract sheet, before focusing on up to three candidate relationships, discuss whether the reduction, addition, meaning and label are sufficient to support the explanation. The final decision is to use the author to look at the original page. This is not drafted. rebuttal.",
        "",
        "## Literally input notes",
        "",
        "The main text of the annex contains an internal line break, with the four groups remaining as they are; the target question sequence and the points remain unchanged, but clean and variant. The report clearly documents this range and does not treat the result that contains a line change as a separate causal effect of a pure prefix.. E1/E2 They’re all caught at the question mark., E3/E4 They’re all caught at the end..",
        "",
        "## Download & Record",
        "",
        "- [Only four full case reports ZIP](../rss_diagnostic_four_reports.zip)",
        "- [Original input](inputs.json) · [Model/SAE Configure and Version](config.json) · [Run Finish State](run.json)",
        "- [AI Qualitative reading records](review_decisions.json) · [Machine Readable Comparison](comparison.json)",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "status": "four_cases_completed_and_reviewed",
                "cases": summaries,
                "preferred_partial_case": "E1",
                "specific_mechanism_transition_established": False,
                "final_rebuttal_drafted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    archive = root.parent / "rss_diagnostic_four_reports.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for c in summaries:
            z.write(root / c["id"] / "case_report.md", f"{c['id']}_report.md")
    print(root / "comparison.md")
    print(archive)


if __name__ == "__main__":
    main()
