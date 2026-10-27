"""Validate nested experiment analysis groups against authoritative Knowledge IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
CATEGORIES={"checkpoints","metrics","tables","figures","logs","metadata","reports","tmp"}


def validate(run: Path) -> dict:
    root = run.resolve(strict=True)
    manifest = json.loads((root / "metadata/run.json").read_text(encoding="utf-8"))
    graph = json.loads((ROOT / "doc/knowledge/context_graph.json").read_text(encoding="utf-8"))
    known = {node["id"] for node in graph["nodes"]}
    groupdir = root / "analysis"
    found = []
    if groupdir.is_dir():
        for path in sorted(groupdir.rglob("metadata/context.json")):
            if not path.resolve().is_relative_to(groupdir.resolve()):
                raise ValueError("Analysis metadata symlink escapes the run")
            rec=json.loads(path.read_text(encoding="utf-8"))
            group=path.parent.parent
            rel=group.relative_to(groupdir).as_posix()
            if rec.get("schema_version")!=1 or rec.get("kind")!="analysis_group" or rec.get("path")!=rel:
                raise ValueError(f"Malformed subgroup record: {path}")
            ids=rec.get("context_ids",[])
            if not isinstance(ids,list) or not ids or any(x not in known for x in ids):
                raise ValueError(f"Subgroup references unknown Knowledge ID: {rel}")
            text=rec.get("description","")
            if not isinstance(text,str) or not text.startswith("目标：") or len(text)<12:
                raise ValueError(f"Missing context-governed explanation: {rel}")
            for category in CATEGORIES:
                if not (group / category).is_dir():
                    raise ValueError(f"Missing typed artifact directory {category}: {rel}")
            found.append(rel)
    if set(found)!=set(manifest.get("analysis_groups",[])):
        raise ValueError("Run manifest analysis_groups differ from on-disk context manifests")
    return {"run":str(root),"analysis_groups":len(found),"validated":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_leaf")
    args=parser.parse_args()
    try:
        report=validate(Path(args.run_leaf))
    except (OSError,ValueError,KeyError) as exc:
        print(f"INVALID experiment: {exc}",file=sys.stderr)
        return 1
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

