"""Summarize one WP21 dispatch without converting NOT_RUN into PASS."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import BASE_COMMIT, canonical_hash, read_json, write_json


def summarize(dispatch: Path, reviews: Path) -> tuple[dict,str]:
    data=read_json(dispatch)
    if data.get("schema")!="saebase.wp21.dispatch.v1":
        raise ValueError("invalid dispatch")
    output=[]
    seen=set()
    for item in data["cases"]:
        cid=item["case_id"]
        if cid in seen:
            raise ValueError("duplicate dispatched case")
        seen.add(cid)
        assignment=read_json(dispatch.parent/"control"/cid/"assignment.json")
        if assignment.get("case_id")!=cid or assignment.get("run_id")!=data["run_id"]:
            raise ValueError("dispatch/assignment identity mismatch")
        final=reviews/cid/"review_final.json"
        if final.is_file():
            record=read_json(final)
            if record.get("schema")!="saebase.wp21.final_review.v1" or record.get("case_id")!=cid or record.get("run_id")!=data["run_id"]:
                raise ValueError("review final identity mismatch")
            status=record["status"]
            score=record.get("score",{}).get("total")
            blockers=record.get("blockers",[])
            attribution=record.get("attribution")
            failed_check_ids=record.get("failed_check_ids",[])
            evidence_rejection=record.get("candidate_evidence_rejection_reason")
            dimensions=record.get("score",{})
        else:
            status="NOT_RUN" if assignment["state"]!="assigned" else "ASSIGNED_NOT_REVIEWED"
            score=None
            blockers=[]
            attribution=None
            failed_check_ids=[]
            evidence_rejection=None
            dimensions=None
        output.append({"case_id":cid,"status":status,"score":score,
                       "workspace":item["workspace"],"review_final":str(final) if final.is_file() else None,
                       "blockers":blockers,"attribution":attribution,
                       "failed_check_ids":failed_check_ids,"candidate_evidence_rejection_reason":evidence_rejection,
                       "frozen_score_dimensions":dimensions})
    counts={status:sum(item["status"]==status for item in output) for status in sorted({x["status"] for x in output})}
    result={"schema":"saebase.wp21.summary.v1","run_id":data["run_id"],"base_commit":BASE_COMMIT,
            "cases":output,"counts":counts,"created_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "note":"Only independent final reviews may record PASS; missing results remain NOT_RUN or ASSIGNED_NOT_REVIEWED."}
    lines=["# WP21 独立考试结果","",f"Run ID: `{data['run_id']}`","",
           "| Case | 最终状态 | 冻结分数 | 归因 | 未完成检查 |", "| --- | --- | ---: | --- | --- |"]
    for x in output:
        ids = ", ".join(x["failed_check_ids"]) or "—"
        cause = x["attribution"] or "—"
        lines.append(f"| {x['case_id']} | {x['status']} | {x['score'] if x['score'] is not None else '—'} | {cause} | {ids} |")
    lines.extend(["","状态计数："+", ".join(f"{k}={v}" for k,v in counts.items()),"",
                  "只有真实独立复跑完成、所有硬门槛满足的题目才标记 PASS。"])
    return result,"\n".join(lines)+"\n"


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dispatch",required=True,type=Path)
    p.add_argument("--reviews-dir",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    try:
        result,md=summarize(a.dispatch.resolve(strict=True),a.reviews_dir.resolve())
        target=a.output_dir.absolute()
        if target.is_relative_to(a.dispatch.parent):
            raise ValueError("summary must be stored outside candidate/dispatch workspace")
        target.mkdir(parents=True,exist_ok=True)
        write_json(target/"summary.json",result,exclusive=True)
        (target/"summary.md").write_text(md,encoding="utf-8")
        print(json.dumps({"status":"SUMMARIZED","counts":result["counts"],"output":str(target)},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"SUMMARY_ERROR","reason":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=="__main__":raise SystemExit(main())
