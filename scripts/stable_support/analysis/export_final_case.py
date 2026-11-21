"""Export the measured same-case target/context features selected for display."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tasks.capture.rss_mode import feature_row

TARGET_IDS = [12293, 28763, 9190]
CONTEXT_IDS = [17410, 5436, 32506, 3044, 30221]
TOPICS = ["Basketball", "Space", "Music", "Chess", "Animals"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    case_path = directory / "case_01/rss_case.json"
    case = json.loads(case_path.read_text())
    target = [next(r for r in case["rss_features"] if r["feature_id"] == f) for f in TARGET_IDS]
    context = [
        next(r for r in case["context_candidates"][v]["features"] if r["feature_id"] == f)
        for v, f in enumerate(CONTEXT_IDS)
    ]
    lines = [
        "# Llama 3.1:Re-allocation of objectives in the same case and new background features",
        "",
        "This example is the same. clean,One group of five variants,Same model and SAE Up, three goals are linked. RSS Features added to the five articles, not included RSS Selected features.",
        "",
        "Model: Llama 3.1 8B Base.SAE:`llama31_8b_lxr_8x`,layer 21,`blocks.21.hook_resid_post`,32,768 features.All input is the same at the last one. token(Period) Capture whole original SAE codes.",
        "",
        "RSS Parameters: M=400,K=20,rho=kappa=0.8,tau=1e-5.The change value is variant − clean.Explain reading after active selection is determined, not participating RSS Selection.",
        "",
        "## Full Input",
        "",
        f"Clean:{case['clean_prompt']}",
        "",
    ]
    for v, prompt in enumerate(case["variants"], 1):
        lines += [f"V{v} · {TOPICS[v - 1]}:{prompt}", ""]
    lines += [
        "## Table A:RSS Selected three cooking-related features",
        "",
        "From Original RSS It’s... 20 among the actual selected features, extract the following three by explanation. F12293,F28763 Reduction, F9190 Increase.",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Original Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [feature_row(row) for row in target]
    lines += [
        "",
        "## Table B:The five backgrounds corresponded to each other. RSS Excluded features",
        "",
        "Each feature is only reciprocated. variant Add the other four delta All zero; five features recurrence Both. 1/5=0.2,Less than rho=0.8.They’re all in. M=400 ♪ Candidate pool because it’s not enough ♪ recurrence Conditions not available RSS Eligibility for election.",
        "",
        "| Variant | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Original Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for topic, row in zip(TOPICS, context):
        lines.append(f"| {topic} | " + feature_row(row)[2:])
    lines += [
        "",
        "Basketball scene corresponds to team relationships, space scenes corresponds to space travel, violin scenes to musical instruments, chess games to chess tactics, and African elephant scenes to wildlife. Both the original interpretation and the original page link are kept in the table..",
        "",
        "## Design and selection records",
        "",
        "User Allow Changes prompts After, in Llama 3.1 No. No. 16,21 I’ve been exploring the floor. 96 A combination of background and regular cooking requests. backgrounds,Re-enforcement clean + Five. variants;All rows of this report are derived from the actual results of this group, and there are no aggregated tables of previous cases..",
        "",
        "The fixed target was changed from the original backgroundless baking problem to ’based on’ setting This is a context-related formulation rewriter case; it cannot be described as an original non-factual experiment. The presentation should use the full and true input mentioned above. prompts Universal Results.",
        "",
        "## Result Read",
        "",
        "- `case_01/rss_case.json`:All 20 Article RSS Characteristics, all values context Candidates, original explanations, recurrence,Candidatures and exclusions.",
        "- `case_01/rss_activations.npz`:Full clean,Five. variants with delta.",
        "- `case_01/config.yaml`,`case_01/task_result.json`,`run.json`:Actual running settings and official task return.",
        "- `selected_case.json`:It’s for the show. 8 Article features and complete input.",
    ]
    (directory / "case_report.md").write_text("\n".join(lines) + "\n")
    (directory / "selected_case.json").write_text(
        json.dumps(
            {
                "status": "matched",
                "selection_kind": "exploratory_case_study",
                "measurement": str(case_path),
                "identity": case["identity"],
                "clean": case["clean_prompt"],
                "variants": case["variants"],
                "rss_parameters": case["rss_parameters"],
                "table_a": target,
                "table_b": [
                    {"variant": v + 1, "topic": TOPICS[v], **row} for v, row in enumerate(context)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(directory / "case_report.md")
    for row in target:
        print("TARGET", row["feature_id"], row["delta"], row["recurrence"], row["consistency"])
    for row in context:
        print(
            "CONTEXT",
            row["feature_id"],
            row["delta"],
            row["recurrence"],
            "in_top_M",
            row["in_top_M"],
        )


if __name__ == "__main__":
    main()
