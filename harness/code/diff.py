"""Compare independently generated code trees; validate planned vs observed structural changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FIELDS=("signature","bases","context_ids","core_ids","description")
def read(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema_version")!=1 or obj.get("kind")!="code":
        raise ValueError(f"Not a code tree: {path}")
    return obj

def compare(before,after):
    old={n["id"]:n for n in before["nodes"]}
    new={n["id"]:n for n in after["nodes"]}
    additions=sorted(set(new)-set(old))
    removals=sorted(set(old)-set(new))
    updates=[]
    for node_id in sorted(set(old)&set(new)):
        changed=[f for f in FIELDS if old[node_id].get(f)!=new[node_id].get(f)]
        if changed:
            updates.append({"id":node_id,"fields":changed,"before":old[node_id]["signature"],
                            "after":new[node_id]["signature"]})
    edgekey=lambda obj:(obj["source"],obj["type"],obj["target"])
    oe={edgekey(e) for e in before["edges"]}
    ne={edgekey(e) for e in after["edges"]}
    return {
       "kind":"code_diff","schema_version":1,
       "before_source":before.get("source_sha256"),"after_source":after.get("source_sha256"),
       "added_ids":additions,"removed_ids":removals,"changed":updates,
       "edges_added":[{"source":a,"type":b,"target":c} for a,b,c in sorted(ne-oe)],
       "edges_removed":[{"source":a,"type":b,"target":c} for a,b,c in sorted(oe-ne)]
    }

def check_plan(report,plan):
    expected=plan.get("expected",[])
    if not isinstance(expected,list):
        raise ValueError("Plan expected must be a list")
    observed=set([("add",x) for x in report["added_ids"]]+
                 [("remove",x) for x in report["removed_ids"]]+
                 [("change",x["id"]) for x in report["changed"]])
    planned=set()
    for row in expected:
        if row.get("operation") not in ("add","remove","change") or not isinstance(row.get("id"),str):
            raise ValueError(f"Bad planned change {row}")
        planned.add((row["operation"],row["id"]))
    return {"planned":len(planned),"matched":len(planned & observed),
            "missing":[{"operation":op,"id":key} for op,key in sorted(planned-observed)],
            "unexpected":[{"operation":op,"id":key} for op,key in sorted(observed-planned)]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",required=True)
    parser.add_argument("--after",required=True)
    parser.add_argument("--plan",help="Optional planned code-tree change manifest")
    parser.add_argument("--output",help="Write JSON result for visualization")
    parser.add_argument("--strict",action="store_true",help="Fail on plan mismatch")
    args=parser.parse_args()
    try:
        report=compare(read(args.before),read(args.after))
        if args.plan:
            p=json.loads(Path(args.plan).read_text(encoding="utf-8"))
            report["plan_check"]=check_plan(report,p)
        if args.output:
            target=Path(args.output)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({
          "added":len(report["added_ids"]),"removed":len(report["removed_ids"]),
          "changed":len(report["changed"]),"edges_added":len(report["edges_added"]),
          "edges_removed":len(report["edges_removed"]),"plan":report.get("plan_check")
        },ensure_ascii=False,indent=2))
        if args.strict and args.plan and (report["plan_check"]["missing"] or report["plan_check"]["unexpected"]):
            return 2
    except (OSError,ValueError,KeyError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 1
    return 0

if __name__=="__main__":
    sys.exit(main())
