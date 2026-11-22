"""Export transparent explanation navigation and a same-case report from measured results."""

import argparse
import json
from pathlib import Path
import re

from saebase.runtime.experiment_results import historical_run_path
from tasks.capture.rss_mode import feature_row

ROOT = Path(__file__).resolve().parents[3]

# These patterns only navigate post-selection explanations, never select RSS features.
PATTERNS = [
    r"basket|sports?|athlet|ball",
    r"astronaut|space|orbit|cosmi|nasa|rocket|planet|galax|aeronaut|astronom",
    r"violin|music|symphon|orchestra|instrument|concert",
    r"chess|knight|board.game",
    r"elephant|animal|africa|savanna|wildlife|mammal",
]


def navigation(root, output):
    cases = []
    for path in sorted(root.glob("case_*/rss_case.json")):
        c = json.loads(path.read_text())
        groups = []
        for v, group in enumerate(c["context_candidates"]):
            matched = [
                r
                for r in group["features"]
                if re.search(
                    PATTERNS[v],
                    "; ".join(e["description"] for e in r["annotation"]["explanations"]),
                    re.I,
                )
            ]
            groups.append(
                {
                    "variant": v + 1,
                    "candidate_count": len(group["features"]),
                    "keyword_navigation_matches": matched,
                }
            )
        cases.append({"case": path.parent.name, "prompt": c["clean_prompt"], "groups": groups})
    report = {
        "purpose": "Explaining keywords is used only for positioning reading, does not mean semantic acceptance, nor does it affect RSS Select. All value candidates are saved in the original case.",
        "patterns": PATTERNS,
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "explanation_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write a separate report and explanation reviews")
    args = parser.parse_args(argv)
    original = historical_run_path(args.original)
    alternative = historical_run_path(args.alternative)
    archive = ROOT / "assets/history_experiment/local_runs/files"
    if args.output is None and any(path.is_relative_to(archive) for path in (original, alternative)):
        parser.error("--output is required when reviewing archived cases")
    output = args.output.resolve() if args.output else original
    navigation(original, output / "original" if args.output else original)
    navigation(alternative, output / "alternative" if args.output else alternative)
    c = json.loads((alternative / "case_04/rss_case.json").read_text())
    lines = [
        "# Second round: a comparison of the same real case",
        "",
        "**I’ve got three two-way changes in relation to the goal. RSS features; not yet found for five new contexts RSS-excluded Characteristics, complete acceptance condition not met.**",
        "",
        "GPT-2 Small,layer 8,SAE `gpt2small_fs_24576_l8`(Neuronpedia `gpt2-small/8-res_fs24576-jb`).Both forms come from this one. SAE,The same. clean 5 with the same group variants,In the last target. token(Period) Full original SAE codes.",
        "",
        f"Clean:{c['clean_prompt']}",
        "",
    ]
    lines += [f"- V{i}:{p}" for i, p in enumerate(c["variants"], 1)]
    lines += [
        "",
        "## Table A:Original RSS Selected three target-related features",
        "",
        "Full RSS support Yes. 20,Three of these articles are selected below by public explanation. recurrence and consistency Both. 1.0;Explanation of non-participation RSS Select. Keep original Neuronpedia explanation.",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    selected = []
    for fid in (3130, 8406, 15492):
        row = next(r for r in c["rss_features"] if r["feature_id"] == fid)
        lines.append(feature_row(row))
        selected.append(row)
    lines += [
        "",
        "F3130 In five. variants Decrease in average, F8406 and F15492 Increase in all; all three explanations point to food, recipe or cooking steps that are suitable for showing two-way changes in the characteristics associated with the retention of the target.",
        "",
        "## Table B:Results of screening in five contexts (five eligible) context features Not obtained)",
        "",
        "None of the candidates listed have been identified. RSS Selected and recurrence < 0.8;They can’t be marked as five context semantics that match.. V1 The nearest sports candidate in the semantics, V2,V5 Can not get folder: %s: %s; V3,V4 There are no candidates who meet the numerical filter criteria in the full dictionary.",
        "",
        "| Variant | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    partial = []
    for v, fid in [(1, 3959), (2, 15550), (3, None), (4, None), (5, 12314)]:
        if fid is None:
            lines.append(f"| V{v} | Unqualified candidate | — | — | — | — | — | — | — |")
        else:
            row = next(
                r for r in c["context_candidates"][v - 1]["features"] if r["feature_id"] == fid
            )
            lines.append(f"| V{v} | " + feature_row(row)[2:])
            partial.append({"variant": v, "row": row})
    lines += [
        "",
        "V1 It’s a broad-based sport, like football, and it’s just changed. +0.136;V2 The explanation is that it’s ready, and it doesn’t correspond directly to astronauts.; V5 The most unique response is interpreted as cooking. The numerical concentration here is not consistent with the syntax of the context.. V3,V4 The empty set is not an explanation of the missing.",
        "",
        "## Case selection process and scope",
        "",
        "1. First of all, it’s right there. SAE `gpt2small_res_jb_l8` Run banana bread and the rest 23 One. demo prompts.Banana bread. F2215,F567 It’s a clear cooking. RSS Characteristics reduced five times; no specific features were found for each of the five new contexts.",
        "2. In the same one. GPT-2 Small layer 8 Change to registered and available Neuronpedia Source `gpt2small_fs_24576_l8`,Run All Again 24 One. prompts.Banana bread gets a watch. A The three goals are related in two directions. RSS features, but tables B These gaps remain.",
        "3. Save Together 48 Full candidate, using the same five contexts, the same RSS Parameters and end position of the same target;no difference prompts,Different. SAE Or different. token Spell Location as a Full Case.",
        "4. context Numerical filter first all that is not available RSS Selected features, request recurrence < 0.8,Current variable increases > 1e-5,At least five times the maximum absolute change in the other variants, with all of them retained in order of change..",
        "5. To help read one by one,`explanation_review.json` Saves the candidates and original explanations for which five semantic keywords are located; the keyword matching is only navigation, and manual judgement still needs to look at the specifics. Additional view of the complete activation matrix found: two sets SAE None of the chess-related interpretative features appear.“V4 I’m changing. recurrence < 0.8”In the case of a five-fold rule, there’s no way to make up for that..",
        "",
        "## Conclusions currently available",
        "",
        "Table A Cases in which a two-way activity can change as a relevant feature of the reservation. The current result is not supported‘RSS At the same time, make a distinction between the five additional context-specific codings ’ , the complete conclusion. token The observations are not sufficient to indicate whether the model is coding these contexts at other locations; a change in the capture position belongs to another group of experiments and should be presented separately..",
        "",
        f"[Original SAE All cases]({original / 'index.md'}) · [Feature Splitting SAE All cases]({alternative / 'index.md'}) · [All of this RSS _Other Organiser]({alternative / 'case_04/rss_case.md'})",
    ]
    report_path = output / "contrast_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    (output / "case_selection.json").write_text(
        json.dumps(
            {
                "status": "acceptance_not_met",
                "searched_case_count": 48,
                "selected_partial_case": str(alternative / "case_04"),
                "target_features": selected,
                "context_partial_rows": partial,
                "missing": "No same case with five semantically aligned, RSS-excluded context features was found in these two SAEs and 24 targets.",
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if args.output is None:
        for directory in (original, alternative):
            index = directory / "index.md"
            text = index.read_text()
            marker = f"[Read first: Results and gaps in the current round]({report_path})"
            if marker not in text:
                text = text.replace("\n\n", "\n\n" + marker + "\n\n", 1)
            index.write_text(text)
    print(report_path)


if __name__ == "__main__":
    main()
