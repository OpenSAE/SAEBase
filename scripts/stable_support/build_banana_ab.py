"""Export the two independent measured banana-bread tables and the A–E comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.stable_support.build_semantic_preservation import number, explanation, feature, values

COOKING = [20700, 12293, 9190]
CONTEXT = {"A": [None, None, None, None, None], "B": [None, None, None, 3044, None]}
TOPICS = ["Basketball", "Space", "Music", "Chess", "Animals"]
HEADER = "| RSS status | Feature | Clean activation | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | Original Neuronpedia explanation |"
RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    inputs = json.loads((root / "inputs.json").read_text())
    results = {}
    tables = {}
    selections = {}
    for name in ["A", "B"]:
        result = json.loads((root / name / "rss_case.json").read_text())
        results[name] = result
        selected = {r["feature_id"]: r for r in result["rss_features"]}
        cooking = [selected[f] for f in COOKING]
        bg = [
            None
            if f is None
            else next(
                r
                for r in result["all_pool_candidates"]
                if r["feature_id"] == f and not r["rss_selected"]
            )
            for f in CONTEXT[name]
        ]
        table = [HEADER, RULE]
        for row in cooking:
            table.append(
                f"| Selected {'↑' if row['direction'] > 0 else '↓'} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} |"
            )
        for v, row in enumerate(bg):
            if row is None:
                table.append(
                    f"| Excluded {TOPICS[v]}:No semantic match confirmed | — | — | — | — | — | — | — | — | — |"
                )
            else:
                table.append(
                    f"| Excluded {TOPICS[v]} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} |"
                )
        tables[name] = table
        selections[name] = {
            "selected_cooking_ids": COOKING,
            "display_context_ids": CONTEXT[name],
            "cooking_rows": cooking,
            "context_rows": bg,
            "confirmed_background_count": sum(r is not None for r in bg),
            "interpretation_status": "post_selection_AI_qualitative_review",
        }
        lines = [
            f"# Group {name}:{'Keep scene first sentence' if name == 'A' else 'Only delete scene first sentence'}",
            "",
            "Model: Llama 3.1 8B Base;SAE:llama31_8b_lxr_8x,L21,blocks.21.hook_resid_post,32768 V. RSS:M=400,K=20,rho=kappa=0.8,tau=1e-5.None SAR Screening or standardization.",
            "",
            "## Full Input",
            "",
            "Clean",
            "",
            "```text",
            inputs["clean"],
            "```",
            "",
        ]
        for variant in inputs["groups"][name]:
            lines += [
                f"{name}-V{variant['variant']} {variant['topic']}",
                "",
                "```text",
                variant["prompt"],
                "```",
                "",
            ]
        lines += [
            "## Independence RSS Results table",
            "",
            "All three of the cooking-related lines are from the group’s recapturing and independence. RSS Select, All recurrence=1,consistency=1.Background missing entries are not replaced by unrelated tabs. Non-zero values show five decimal places, with original accuracy remaining on JSON/NPZ.",
            "",
            *table,
            "",
            "## Semantic Interpretation",
            "",
            "F20700 The original explanation was cooking instructions and food-related operating procedures in five. variants Increase in average; F12293 Pointing to food preparation and food., F9190 It points to cooking, cooking and food content in five. variants All of them are reduced. The fixed problem still requires a description of the normal bakery practice for standard banana bread. All three labels are relevant to the retention of the mission and show the food./Cooking content changes in a system that is the opposite of the characteristics of the description. The addition cannot be interpreted automatically as better, more abstract or more correct, nor can the labels be used as information feature Proof of accurate relocation.",
            "",
        ]
        if name == "A":
            lines += [
                "No direct semantic matching of any of the five types of background has been identified. Strict monovariate candidates do not directly represent space travel or chess, although they have a neighbouring label such as ”Physics and Energy” and ”Puzzles and Text Game” in them; they are not used as clear background evidence..",
                "",
            ]
        else:
            lines += [
                "F3044 The original explanation clearly points to chess tactics and game mechanisms, only B-V4 Increase+0.86217,The other four delta Average 0,clean=0.It’s in the original. M=400 Candidature Number Two 89,recurrence=0.2,consistency=1,So it’s clear that recurrence♪ Not enough to be ♪RSS Excludes. Four other types of background are not clearly matched.",
                "",
            ]
        lines += [
            "## Full record",
            "",
            "- [All of this group 20 Article of the Treaty RSS Selected](all_selected_features.md)",
            "- [All Strict Single Variable Candidates and Original Explanations](context_candidates.md)",
            "- [Full result, front 400 Pools and all exclusions](rss_case.json)",
            "- [Full clean,variants,delta](rss_activations.npz)",
            "- [Two sets of input](../inputs.json) · [Common Configuration and Identity](../config.yaml) · [Comparison](../comparison.md)",
            "",
            "Effective measurement of active models and SAE Load only once, clean Only once. Five for each of the two. variants And run independently. RSS,then search for a uniform explanation. GPU There’s something else going on. The model first. CPU Load, actual embedded layer required, blocks0–21 with SAE Yes. GPU Calculating, leave the non-existent follow-up layer. CPU;Capture It’s in itself. layer21. Enter, weight and accuracy, original RSS And the capture position hasn’t changed..",
        ]
        (root / name / "rss_case.md").write_text("\n".join(lines) + "\n")
        full = ["# All 20 Article of the Treaty RSS Selected", "", HEADER, RULE]
        for row in result["rss_features"]:
            full.append(
                f"| Selected {'↑' if row['direction'] > 0 else '↓'} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} |"
            )
        (root / name / "all_selected_features.md").write_text("\n".join(full) + "\n")
        cand = [
            "# All Strict Monovariate Background Candidates",
            "",
            "In front of the original 400 Ichichi, not. RSS Selected, should variant It’s changing more than tau And the other four definitely don’t change. tau.Keeps all the candidates; whether to match the given background needs to read the original explanation. 400 All remaining candidates in the pool (including those who do not meet strict single variable conditions) are saved in rss_case.json.",
            "",
            HEADER + " Exclusion reason |",
            RULE + " --- |",
        ]
        for group in result["strict_context_candidates"]:
            for row in group["features"]:
                cand.append(
                    f"| Excluded V{group['variant']} {group['topic']} | {feature(row)} | {number(row['clean_activation'])} | {values(row)} | {row['recurrence']:.1f} | {explanation(row)} | {', '.join(row['exclusion_reasons'])} |"
                )
        (root / name / "context_candidates.md").write_text("\n".join(cand) + "\n")
        result["qualitative_display"] = selections[name]
        (root / name / "rss_case.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    overlap = sorted(
        set(COOKING)
        & {r["feature_id"] for r in results["A"]["rss_features"]}
        & {r["feature_id"] for r in results["B"]["rss_features"]}
    )
    answers = {
        "A": "Both groups are the same: they all found three clearly linked cooking tasks. RSS Characteristics, respectively. F20700,F12293,F9190.None of these groups have been able to tell more about culinary characteristics..",
        "B": "Both groups are in the opposite direction.: F20700 In five. variants Add All, F12293 with F9190 All reductions; three in two groups recurrence and consistency Both. 1.0.",
        "C": "B Group is more complete: yes 1/5 Confirm background match(Chess F3044),A Group 0/5.Checked All Before 400 Open labels and responses within the pool, with a strict candidate;/Energy or puzzles/Text games, such as broad proximity labels, are not forced into space or chess.",
        "D": "All three of the cooking features of the two groups overlap., ID Yes. F20700,F12293,F9190,And it’s the same direction. They’re all on their own. RSS Actual Selected.",
        "E": "B The group is a stronger local characterization case: a two-way change in three culinary-related characteristics after only the scenario guide has been deleted, with an additional semantic clarity, as a result recurrence Not sufficient to exclude chess features. Neither group has been able to obtain a complete cross-check of the five background categories, and should not claim that all five background categories have been filtered..",
    }
    lines = [
        "# Banana bread. A/B:Any independent comparison of the scene guide sentence",
        "",
        "Specifies that the input is executed as it is. Llama 3.1 8B Base / L21 SAE And once. clean Captured. Ten more. variant capture; two groups independently running the original RSS,An explanation is obtained only after two sets of value sets have been determined. The single space connection of the original file remains unchanged..",
        "",
        "**There are three two-way changes in cooking-related characteristics in both groups.; B The group also found a clear background exclusion feature.. B The group is a stronger local case, but has not yet reached all five background matches.**",
        "",
        "Common Clean:" + inputs["clean"],
        "",
        "## Group A:Keep scene first sentence",
        "",
        *tables["A"],
        "",
        "[A Group Complete Inputs and Records](A/rss_case.md)",
        "",
        "## Group B:Only remove scene first sentence",
        "",
        *tables["B"],
        "",
        "[B Group Complete Inputs and Records](B/rss_case.md)",
        "",
        "## Five comparative issues",
        "",
    ]
    for key, text in answers.items():
        lines += [f"### {key}", "", text, ""]
    lines += [
        "## Scope of semantic relations",
        "",
        "F12293 with F9190 The public label focuses on food, preparation and cooking in general., F20700 Focus on cooking instructions and operating procedures. SAE Characteristics continue to change systematically in both groups. This is a label-level relationship that can be discussed directly in this case; there is no assessment of the quality of the answer, nor does it equate the rise with a better or more abstract one..",
        "",
        "B-V4 It’s... F3044 In the same group B♪ New measurements, not with ♪A Could not close temporary folder: %s. B Group the rest of the background except for the match, A All background of the group, no direct semantics confirmed..",
        "",
        "Neuronpedia Cancellation 423 One. ID Medium 422 Public explanation; only explanation missing F177 All in two groups. delta Both. 0,Not a non-zero-background candidate, nor does it affect the integrity of the line presented..",
        "",
        "Original clean/variant Vector, delta,Each group 20 Article support,All for each group 400 Can not delete folder: %s: No such folder A/B Directory JSON/NPZ. No quantification benchmark,No designation modified prompts,Not drafted rebuttal.",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    source = ROOT / "tasks/capture/scoring/stable_support.py"
    data = source.read_bytes()
    comparison = {
        "answers": answers,
        "groups": selections,
        "cooking_overlap": overlap,
        "stronger_partial_case": "B",
        "complete_five_context_example": False,
        "selector_sha256": hashlib.sha256(data).hexdigest(),
        "selector_git_blob": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    (root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )
    with ZipFile(root.parent / "rss_banana_ab_reports.zip", "w", ZIP_DEFLATED) as z:
        z.write(root / "comparison.md", "comparison.md")
        for name in ["A", "B"]:
            z.write(root / name / "rss_case.md", f"{name}/rss_case.md")
    print(root / "comparison.md")
    print(root.parent / "rss_banana_ab_reports.zip")


if __name__ == "__main__":
    main()
