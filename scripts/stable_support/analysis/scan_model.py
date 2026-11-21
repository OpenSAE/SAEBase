"""Navigate measured context candidates and target features for a new model."""

import argparse
import json
from pathlib import Path
import re

PATTERNS = [
    r"basketball|sports|sporting|athlet|three.point|scoring|scores|NBA",
    r"astronaut|space|orbit|cosmi|NASA|rocket|planet|galax|aeronaut|astronom|celestial",
    r"violin|music|symphon|orchestra|instrument|concert|musician",
    r"chess|knight|board.game|chessboard",
    r"elephant|animal|Africa|savanna|wildlife|mammal|safari",
]
TARGETS = [
    r"sky|blue|color|light|atmospher",
    r"rainbow|weather|storm|rain|light|color",
    r"cook|food|recipe|tomato|sauce",
    r"bak|cook|recipe|ingredient|bread|banana",
    r"water|hydrat|health",
    r"sleep|memory",
    r"travel|pack|Tokyo|Japan",
    r"travel|coast|relax|weekend",
    r"Roman|history|empire",
    r"print|Europe|history",
    r"password|security|computer|data",
    r"solar|light|power|energy",
    r"financ|interest|saving|money",
    r"financ|budget|student|money",
    r"volleyball|sport",
    r"marathon|running|runner|athlet",
    r"bee|honey|insect",
    r"leaves|leaf|plant|autumn|color",
    r"paint|art|impression",
    r"jazz|music",
    r"stud|exam|learn|educat",
    r"read|child|literacy|learn",
    r"stain|clean|shirt|cloth|laundry",
    r"plant|garden|horticultur",
]


def description(row):
    return "; ".join(x["description"] for x in row.get("annotation", {}).get("explanations", []))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--target-pattern")
    args = parser.parse_args()
    cases = []
    for path in sorted(args.directory.glob("case_*/rss_case.json")):
        c = json.loads(path.read_text())
        case_id = int(path.parent.name.split("_")[-1])
        groups = []
        for v, group in enumerate(c["context_candidates"]):
            matches = [
                {**r, "description": description(r)}
                for r in group["features"]
                if re.search(PATTERNS[v], description(r), re.I)
            ]
            groups.append(matches)
        targets = [
            {**r, "description": description(r)}
            for r in c["rss_features"]
            if re.search(args.target_pattern or TARGETS[case_id - 1], description(r), re.I)
        ]
        row = {
            "case": path.parent.name,
            "prompt": c["clean_prompt"],
            "context_matches": groups,
            "target_matches": targets,
        }
        cases.append(row)
        print(
            path.parent.name,
            "context",
            [
                [(r["feature_id"], round(r["delta"][v], 3), r["description"]) for r in g]
                for v, g in enumerate(groups)
            ],
            "target",
            [(r["feature_id"], r["direction"], r["description"]) for r in targets],
        )
    (args.directory / "semantic_navigation.json").write_text(
        json.dumps(
            {
                "note": "Post-selection keyword navigation; semantic matches require reading. RSS never uses explanations.",
                "context_patterns": PATTERNS,
                "target_patterns": TARGETS,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
