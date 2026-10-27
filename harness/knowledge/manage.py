"""Create a reviewed knowledge-status proposal or apply it with explicit approval."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys
if __package__:
    from .validate import DEFAULT, validate, GraphError
    from .status import STATES, PRIMARY, refresh_status
else:
    from validate import DEFAULT, validate, GraphError
    from status import STATES, PRIMARY, refresh_status

ROOT = Path(__file__).resolve().parents[2]
PROPOSALS = ROOT / "harness/knowledge/proposals"


def atomic_write(target: Path, obj: dict):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix(target.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(target)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def propose(args):
    graph=json.loads(DEFAULT.read_text(encoding="utf-8"))
    validate(graph)
    matching=[n for n in graph["nodes"] if n["id"]==args.node]
    if len(matching)!=1:
        raise GraphError(f"Unknown knowledge node: {args.node}")
    if args.status not in STATES:
        raise GraphError("Unknown status")
    if len(args.reason.strip())<12 or not args.evidence:
        raise GraphError("Require specific reason (>=12 characters) and actual evidence")
    for item in args.evidence:
        if not item.strip():
            raise GraphError("Blank evidence")
    obj={"version":1,"node_id":args.node,"before":matching[0]["status"],
         "after":args.status,"reason":args.reason.strip(),
         "evidence":args.evidence,"created_at_utc":now(),"review_status":"proposed"}
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename=re.sub(r"[^a-zA-Z0-9_-]","_",args.node)
    target=PROPOSALS/f"{stamp}_{filename}.json"
    if target.exists():
        raise GraphError(f"Proposal collision: {target}")
    atomic_write(target,obj)
    print(f"PROPOSED {target.relative_to(ROOT)}; authoritative graph unchanged")


def apply(args):
    if not args.approve:
        raise GraphError("Applying knowledge changes requires explicit --approve by the authorized GitHub maintainer")
    source=Path(args.proposal).resolve(strict=True)
    if not source.is_relative_to(PROPOSALS.resolve(strict=False)):
        raise GraphError("Proposal must live inside harness/knowledge/proposals/")
    proposal=json.loads(source.read_text(encoding="utf-8"))
    if proposal.get("review_status")!="proposed":
        raise GraphError("Proposal was already handled")
    graph=json.loads(DEFAULT.read_text(encoding="utf-8"))
    validate(graph)
    matches=[n for n in graph["nodes"] if n["id"]==proposal.get("node_id")]
    if len(matches)!=1 or matches[0]["status"]!=proposal["before"]:
        raise GraphError("Graph status changed; review this proposal again")
    node=matches[0]
    node["status"]=proposal["after"]
    node["evidence"]=sorted(set(node.get("evidence",[])+proposal["evidence"]))
    node["last_change"]={"at_utc":now(),"reason":proposal["reason"],"evidence":proposal["evidence"]}
    # Parent status is a separate reviewed statement; never automatically
    # propagate this change along the scientific hierarchy.
    refresh_status(graph)
    validate(graph)
    atomic_write(DEFAULT,graph)
    proposal["review_status"]="applied"
    proposal["applied_at_utc"]=now()
    atomic_write(source,proposal)
    print(f"APPLIED reviewed knowledge change: {node['id']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("propose")
    p.add_argument("--node",required=True);p.add_argument("--status",required=True,choices=sorted(STATES))
    p.add_argument("--reason",required=True);p.add_argument("--evidence",required=True,action="append")
    a=sub.add_parser("apply")
    a.add_argument("proposal");a.add_argument("--approve",action="store_true")
    args=parser.parse_args()
    try:
        propose(args) if args.command=="propose" else apply(args)
    except (GraphError,OSError,ValueError,KeyError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
