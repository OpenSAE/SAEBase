"""Trusted WP21 reviewer workspace, independent replay log, and hard-gate verdict.

This tool never auto-awards PASS from candidate evidence or from an assertion
label. A reviewer must execute each frozen P/N check in a separately created
worktree and produce independently computed assertion artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import BASE_COMMIT, IGNORED_DIRS, ROOT, SUITE, allowed_source, canonical_hash, contained, get_case, read_json, sha256, snapshot, source_changes, write_json
from manage import check_spec, scope_audit
from record import record
from audit_evidence import audit
from preflight import verify_assets


def checklist(case: dict) -> list[dict]:
    return ([{"id":f"P{i:02d}","kind":"positive","text":text,"status":"NOT_RUN","attempts":[]} for i,text in enumerate(case["positive_checks"],1)]
         + [{"id":f"N{i:02d}","kind":"negative","text":text,"status":"NOT_RUN","attempts":[]} for i,text in enumerate(case["negative_checks"],1)])


def _state(review_dir: Path) -> dict:
    path=review_dir/"review.json"
    result=read_json(path)
    if result.get("schema") != "saebase.wp21.independent_review.v1" or result.get("base_commit") != BASE_COMMIT:
        raise ValueError("unknown reviewer record")
    return result


def init_review(candidate: Path, control: Path, destination: Path) -> dict:
    check_spec()
    candidate=candidate.resolve(strict=True)
    control=control.resolve(strict=True)
    assignment=read_json(control/"assignment.json")
    cid=assignment["case_id"]
    case=get_case(cid)
    if assignment.get("state") != "assigned":
        raise ValueError("an unassigned prepared case cannot be independently graded")
    scope=scope_audit(candidate,control)
    if scope["violations"]:
        raise ValueError("cannot review source scope violation: "+repr(scope["violations"]))
    destination=destination.absolute()
    if destination.exists() or destination.resolve().is_relative_to(ROOT) or destination.resolve().is_relative_to(candidate) or candidate.is_relative_to(destination.resolve()):
        raise ValueError("new external reviewer destination required")
    destination.mkdir(parents=True)
    reviewer=destination/"workspace"
    def ignore(folder,names):
        skipped={n for n in names if n in IGNORED_DIRS or n.endswith((".pyc",".pyo"))}
        if Path(folder)==ROOT:
            skipped.update({"experiment","experiments"}&set(names))
        if Path(folder)==SUITE:
            skipped.update({"runs","acceptance"}&set(names))
        return skipped
    shutil.copytree(ROOT,reviewer,ignore=ignore)
    seed=case.get("seed_fixture")
    if seed:
        seedroot=SUITE/"fixtures"/seed
        for entry in read_json(seedroot/"manifest.json")["files"]:
            src=contained(seedroot,entry["source"],exists=True)
            dest=contained(reviewer,entry["destination"])
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
    (reviewer/"experiment").mkdir()
    # Copy only observed, allowed production changes. Candidate self-test code
    # is NOT transferred as the independent examiner's test implementation.
    transferred=[]
    excluded=[]
    for change in scope["changes"]:
        rel=change["path"]
        if not allowed_source(rel,case):
            raise ValueError("candidate diff escaped frozen scope")
        if rel.startswith("tests/wp21/"):
            excluded.append(rel)
            continue
        target=contained(reviewer,rel)
        source=contained(candidate,rel)
        if change["operation"] == "delete":
            if target.exists():target.unlink()
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        transferred.append(rel)
    state={"schema":"saebase.wp21.independent_review.v1","case_id":cid,"run_id":assignment["run_id"],
           "base_commit":BASE_COMMIT,"assignment_sha256":canonical_hash(assignment),
           "frozen_case_sha256":canonical_hash(case),"source_snapshot_sha256":scope["source_snapshot_sha256"],
           "reviewer_workspace_source_sha256":canonical_hash(snapshot(reviewer)),
           "candidate_workspace":str(candidate),"control":str(control),"workspace":str(reviewer),
           "transferred_product_paths":transferred,"excluded_candidate_test_paths":excluded,
           "created_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"checks":checklist(case),"status":"NOT_RUN",
           "grade":None,"note":"Only candidate product changes transferred. Reviewer authors and runs independent checks."}
    write_json(destination/"review.json",state,exclusive=True)
    return {"status":"REVIEW_WORKSPACE_PREPARED","case_id":cid,"positive_checks":len(case["positive_checks"]),
            "negative_checks":len(case["negative_checks"]),"workspace":str(reviewer),
            "excluded_candidate_test_paths":excluded,"grade":None}


def _get_check(state: dict, check_id: str) -> dict:
    matches=[check for check in state["checks"] if check["id"]==check_id]
    if len(matches)!=1:
        raise ValueError("unknown frozen check ID")
    return matches[0]


def run_check(review_dir: Path, check_id: str, argv: list[str], timeout: int, assertions_name: str | None) -> dict:
    review_dir=review_dir.resolve(strict=True)
    state=_state(review_dir)
    if (review_dir/"review_final.json").exists():
        raise ValueError("review already finalized; create a new reviewer attempt")
    check=_get_check(state,check_id)
    workspace=Path(state["workspace"])
    if workspace != review_dir/"workspace" or not workspace.is_dir():
        raise ValueError("independent workspace missing")
    case=get_case(state["case_id"])
    if not 1<=timeout<=case["budget"]["max_seconds"]:
        raise ValueError("review command timeout exceeds frozen case limit")
    # Detect edits to reviewer product source after init, excluding reports and experiment.
    if canonical_hash(snapshot(workspace)) != state["reviewer_workspace_source_sha256"]:
        raise ValueError("reviewer product source changed; create a fresh review attempt")
    attempt=len(check["attempts"])+1
    directory=review_dir/"records"/check_id
    command_path=directory/f"{attempt:04d}.json"
    assertions_path=directory/f"{attempt:04d}.assertions.json" if assertions_name else None
    script_proof = None
    if assertions_name:
        # Only an independent reviewer-owned script may produce a verified
        # result. A candidate's own tests or inline Python are not an oracle.
        scripts=[Path(arg).resolve(strict=True) for arg in argv if arg.endswith(".py") and Path(arg).is_file()]
        if len(scripts)!=1 or scripts[0].is_relative_to(workspace) or scripts[0].is_relative_to(Path(state["candidate_workspace"])):
            raise ValueError("independent assertion checker must be one external Python script, not candidate code")
        script_proof={"path":str(scripts[0]),"sha256":sha256(scripts[0])}
        # Pass the path to the independent script as a conventional explicit
        # argument. The independent process must create this file itself.
        if assertions_name != "yes":
            raise ValueError("use --assertions yes to require independent assertions")
        if assertions_path.exists():
            raise ValueError("assertion output already exists")
        argv=argv+["--wp21-assertions-output",str(assertions_path),"--wp21-check-id",check_id]
    result=record(argv,workspace,command_path,timeout=timeout)
    digest=sha256(command_path)
    attempt_record={"path":command_path.relative_to(review_dir).as_posix(),"sha256":digest,
                    "exit_code":result["exit_code"],"timed_out":result["timed_out"]}
    if script_proof is not None:
        attempt_record["independent_script"] = script_proof
    if assertions_path is not None:
        if assertions_path.is_file():
            assertion=read_json(assertions_path)
            rows=assertion.get("assertions")
            if assertion.get("schema") != "saebase.wp21.reviewer_assertions.v1" or assertion.get("check_id")!=check_id or not isinstance(rows,list) or not rows:
                raise ValueError("malformed independent assertions")
            for row in rows:
                if not isinstance(row,dict) or not isinstance(row.get("name"),str) or not row["name"] or type(row.get("passed")) is not bool or not isinstance(row.get("observed"), (str,int,float,bool,list,dict)):
                    raise ValueError("assertion rows require name, boolean passed, and observed result")
            attempt_record.update(assertions_path=assertions_path.relative_to(review_dir).as_posix(),
                                  assertions_sha256=sha256(assertions_path),assertion_count=len(rows))
            passed=(result["exit_code"]==0 and not result["timed_out"] and all(x["passed"] for x in rows))
            attempt_record["status"]="PASS_VERIFIED" if passed else "FAIL"
        else:
            attempt_record["status"]="FAIL"
            attempt_record["reason"]="reviewer script did not produce independent assertions"
    else:
        attempt_record["status"]="EXECUTED_UNASSESSED"
    check["attempts"].append(attempt_record)
    check["status"]=attempt_record["status"]
    state["status"]="REPLAY_IN_PROGRESS"
    write_json(review_dir/"review.json",state)
    return {"status":attempt_record["status"],"case_id":state["case_id"],"check_id":check_id,
            "attempt":attempt,"process_exit_code":result["exit_code"],"grade":None}


def _validate_review_records(review_dir: Path, state: dict) -> None:
    if [{"id":c["id"],"kind":c["kind"],"text":c["text"]} for c in state["checks"]] != [
       {"id":c["id"],"kind":c["kind"],"text":c["text"]} for c in checklist(get_case(state["case_id"]))]:
        raise ValueError("frozen positive/negative review checklist was modified")
    for check in state["checks"]:
        for attempt in check["attempts"]:
            path=contained(review_dir,attempt["path"],exists=True)
            if sha256(path)!=attempt["sha256"]:
                raise ValueError("reviewer command record was modified")
            command=read_json(path)
            if command.get("execution_claim")!="process_only" or command.get("exit_code")!=attempt["exit_code"] or command.get("timed_out")!=attempt["timed_out"]:
                raise ValueError("reviewer command status inconsistent")
            for stream in ("stdout","stderr"):
                file=contained(path.parent,command[stream]["path"],exists=True)
                if sha256(file)!=command[stream]["sha256"]:
                    raise ValueError("reviewer stdout/stderr changed")
            if "independent_script" in attempt:
                script=Path(attempt["independent_script"]["path"])
                if script.is_relative_to(Path(state["candidate_workspace"])) or script.is_relative_to(Path(state["workspace"])) or not script.is_file() or sha256(script)!=attempt["independent_script"]["sha256"]:
                    raise ValueError("independent reviewer script changed or belongs to candidate")
            if "assertions_path" in attempt:
                assertions=contained(review_dir,attempt["assertions_path"],exists=True)
                if sha256(assertions)!=attempt["assertions_sha256"]:
                    raise ValueError("reviewer assertions changed")
                report=read_json(assertions)
                if report.get("check_id")!=check["id"] or not report.get("assertions"):
                    raise ValueError("reviewer assertion identity invalid")
                passed=all(type(x.get("passed")) is bool and x["passed"] for x in report["assertions"])
                if (attempt["status"]=="PASS_VERIFIED") != (passed and command["exit_code"]==0 and not command["timed_out"]):
                    raise ValueError("review status differs from observed assertions")
        if check["status"] != (check["attempts"][-1]["status"] if check["attempts"] else "NOT_RUN"):
            raise ValueError("review check status inconsistent with latest attempt")


def finalize(review_dir: Path, *, assets_lock: Path | None, probe_report: Path | None, attribution: str | None = None, owner: str | None = None) -> dict:
    review_dir=review_dir.resolve(strict=True)
    if (review_dir/"review_final.json").exists():
        raise ValueError("review already finalized")
    state=_state(review_dir)
    candidate=Path(state["candidate_workspace"])
    control=Path(state["control"])
    assignment=read_json(control/"assignment.json")
    scope=scope_audit(candidate,control)
    _validate_review_records(review_dir,state)
    errors=[]
    if scope["violations"] or scope["source_snapshot_sha256"]!=state["source_snapshot_sha256"]:
        errors.append("candidate source changed since reviewer snapshot")
    if canonical_hash(snapshot(Path(state["workspace"]))) != state["reviewer_workspace_source_sha256"]:
        errors.append("reviewer workspace product source changed")
    if assignment.get("state")!="assigned" or canonical_hash(assignment)!=state["assignment_sha256"]:
        errors.append("assignment modified or unassigned")
    asset_ok=False
    if assets_lock and probe_report:
        lock=verify_assets(assets_lock)
        probe=read_json(probe_report)
        profiles={"gemma_inference","gemma_training"}
        same_profile=get_case(probe["case_id"])["profile"]==get_case(state["case_id"])["profile"]
        if get_case(probe["case_id"])["profile"] in profiles and get_case(state["case_id"])["profile"] in profiles:
            same_profile=True
        asset_ok=(probe.get("schema")=="saebase.wp21.real_probe.v1" and
                  probe.get("status")=="REAL_MODEL_PROBE_OK_NOT_EXAM_PASS" and probe.get("model_executed") is True
                  and probe.get("asset_lock_sha256")==lock["sha256"] and same_profile)
        if state["case_id"]=="R17" and probe.get("device")!="cuda":
            asset_ok=False
    if not asset_ok:
        errors.append("missing valid, compatible offline real-model probe and unchanged frozen assets")
    try:
        candidate_evidence=audit(candidate,control)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        candidate_evidence={"status":"EVIDENCE_REJECTED","reason":str(exc)}
    evidence_ok=candidate_evidence["status"]=="EVIDENCE_READY_FOR_REPLAY"
    if not evidence_ok:
        errors.append("candidate evidence is not independently auditable")
    case=get_case(state["case_id"])
    causes={"CONTRACT_GAP","SELECTION_GAP","IMPLEMENTATION_ERROR","ENFORCEMENT_GAP","EVALUATOR_SCOPE_GAP","ENV_BLOCKED"}
    owners={"baseline_product","candidate","governance","environment","evaluator"}
    if attribution is not None and attribution not in causes:
        raise ValueError("unknown frozen failure attribution")
    if owner is not None and owner not in owners:
        raise ValueError("unknown failure responsibility owner")
    positives=[c for c in state["checks"] if c["kind"]=="positive"]
    negatives=[c for c in state["checks"] if c["kind"]=="negative"]
    p_pass=sum(c["status"]=="PASS_VERIFIED" for c in positives)
    n_pass=sum(c["status"]=="PASS_VERIFIED" for c in negatives)
    completeness={"positive":p_pass,"positive_total":len(positives),"negative":n_pass,"negative_total":len(negatives)}
    score={"real_functionality":round(40*p_pass/len(positives),2),
           "compatibility_and_negative":round(25*n_pass/len(negatives),2),
           "evidence_reproducibility":20 if evidence_ok and asset_ok else 0,
           "scope_and_documentation":15 if not scope["violations"] and evidence_ok else 0}
    score["total"]=round(sum(score.values()),2)
    all_checks=(p_pass==len(positives) and n_pass==len(negatives))
    status="PASS" if all_checks and not errors else ("FAIL" if any(c["status"]=="FAIL" for c in state["checks"]) or scope["violations"] else "INCOMPLETE")
    if candidate_evidence.get("status")=="ENV_BLOCKED" and not all_checks:
        status="ENV_BLOCKED"
    result={"schema":"saebase.wp21.final_review.v1","case_id":case["id"],"run_id":state["run_id"],
            "status":status,"score":score,"hard_gates":{"frozen_spec_valid":True,
            "candidate_scope_valid":not bool(scope["violations"]),"candidate_evidence_ready":evidence_ok,
            "real_model_probe_valid":asset_ok,"all_positive_replayed":p_pass==len(positives),
            "all_negative_replayed":n_pass==len(negatives)},"checks":completeness,"blockers":errors,
            "attribution":attribution or ("NONE" if status=="PASS" else "UNCLASSIFIED"), "attribution_owner":owner,"attribution_stage":"independent_replay",
            "review_checklist_sha256":canonical_hash(state["checks"]),"candidate_source_sha256":state["source_snapshot_sha256"],
            "reviewer_source_sha256":state["reviewer_workspace_source_sha256"],
            "candidate_evidence_status":candidate_evidence["status"],
            "reviewed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "note":"Automatic hard gates only; independent reviewer scripts are responsible for genuine real-model checks. Human/Atlas approvals remain separate."}
    write_json(review_dir/"review_final.json",result,exclusive=True)
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    init=subs.add_parser("init")
    init.add_argument("--candidate",required=True,type=Path)
    init.add_argument("--control",required=True,type=Path)
    init.add_argument("--destination",required=True,type=Path)
    run=subs.add_parser("run")
    run.add_argument("--review",required=True,type=Path)
    run.add_argument("--check",required=True)
    run.add_argument("--timeout",type=int,default=900)
    run.add_argument("--assertions",choices=["yes"],help="append independent assertion output flags to the checker command")
    run.add_argument("argv",nargs=argparse.REMAINDER)
    final=subs.add_parser("finalize")
    final.add_argument("--review",required=True,type=Path)
    final.add_argument("--assets-lock",type=Path)
    final.add_argument("--probe-report",type=Path)
    final.add_argument("--attribution",choices=["CONTRACT_GAP","SELECTION_GAP","IMPLEMENTATION_ERROR","ENFORCEMENT_GAP","EVALUATOR_SCOPE_GAP","ENV_BLOCKED"])
    final.add_argument("--owner",choices=["baseline_product","candidate","governance","environment","evaluator"])
    a=p.parse_args()
    try:
        if a.command=="init":
            result=init_review(a.candidate,a.control,a.destination)
        elif a.command=="run":
            argv=a.argv[1:] if a.argv and a.argv[0]=="--" else a.argv
            result=run_check(a.review,a.check,argv,a.timeout,a.assertions)
        else:
            result=finalize(a.review,assets_lock=a.assets_lock,probe_report=a.probe_report,attribution=a.attribution,owner=a.owner)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result["status"] not in {"FAIL","INCOMPLETE","ENV_BLOCKED"} else 3
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        print(json.dumps({"status":"REVIEW_ERROR","reason":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=="__main__":raise SystemExit(main())
