"""WP21.1 independent exam gate tests; test-only fixtures, never model passes."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import shutil

import pytest
from test_wp21_local_suite import T, ROOT, _json, _text

TOOLS=ROOT/"harness/tests/wp21/tools"


def _modules():
    import importlib.util
    import sys
    loaded={}
    previous={}
    try:
        for name in ("summary", "probe"):
            previous[name]=sys.modules.get(name)
            spec=importlib.util.spec_from_file_location("_wp21_exam_test_"+name,TOOLS/(name+".py"))
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
            loaded[name]=module
    finally:
        for name,old in previous.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old
    return loaded


@pytest.fixture
def dispatched(tmp_path):
    # Explicit synthetic authorization is only valid in the unit-test fixture.
    root=tmp_path/"dispatch"
    T.manage.prepare(root,["R01"],"synthetic_exam", "SYNTHETIC_TEST_ASSIGNMENT_ONLY")
    return {"root":root,"candidate":root/"candidates/R01", "control":root/"control/R01"}


@pytest.mark.parametrize("relative", ["cases/R02.md", "prompts/R02.md", "fixtures/common/prompts.jsonl"])
def test_freeze_detects_mutated_prose_and_input_in_temporary_tree(tmp_path,relative):
    spec=tmp_path/"spec"
    files=T.common.read_json(ROOT/"harness/tests/wp21/suite_lock.json")["files"]
    for rel in list(files)+["harness/tests/wp21/suite_lock.json"]:
        target=spec/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    (spec/"harness/tests/wp21"/relative).write_text("tampered by unit test",encoding="utf-8")
    with pytest.raises(ValueError,match="frozen exam content changed"):
        T.manage.check_spec(spec)


def test_all_public_review_checklists_bind_the_actual_twenty_cases():
    cases=T.common.load_spec()["cases"]
    assert len(cases)==20
    for case in cases:
        checks=T.review.checklist(case)
        assert len(checks)==len(case["positive_checks"])+len(case["negative_checks"])
        assert [x["id"] for x in checks if x["kind"]=="positive"]==[f"P{i:02d}" for i in range(1,len(case["positive_checks"])+1)]
        assert [x["id"] for x in checks if x["kind"]=="negative"]==[f"N{i:02d}" for i in range(1,len(case["negative_checks"])+1)]
        assert all(x["status"]=="NOT_RUN" for x in checks)


def test_python_version_and_empty_snapshot_must_block_without_running_model(tmp_path):
    empty=tmp_path/"empty"
    empty.mkdir()
    result=T.preflight.preflight("R01",model_snapshot=empty,sae_snapshot=empty)
    assert "model_snapshot_missing_or_empty" in result["blockers"]
    assert result["status"]=="ENV_BLOCKED" and result["executed_model"] is False
    if sys.version_info[:2] not in {(3,11),(3,12)}:
        assert "unsupported_python_requires_3_11_or_3_12" in result["blockers"]


def test_locked_cache_verification_detects_bytes_changed(tmp_path):
    resource=tmp_path/"dummy-not-a-checkpoint"
    resource.mkdir()
    _text(resource/"file.bin","SYNTHETIC_LOCK_TEST_ONLY")
    lock=tmp_path/"lock.json"
    result=T.preflight.freeze_assets([("model",resource)],lock)
    assert T.preflight.verify_assets(lock)["sha256"]==result["sha256"]
    _text(resource/"file.bin","CHANGED")
    with pytest.raises(ValueError,match="bytes changed"):
        T.preflight.verify_assets(lock)


def test_candidate_recorder_requires_assigned_external_control(dispatched,tmp_path):
    x=dispatched
    candidate,control=x["candidate"],x["control"]
    output=candidate/"harness/tests/wp21/runs/synthetic_exam/R01/commands/0001.json"
    argv=[sys.executable,"-c","print('synthetic recorder auth test')"]
    with pytest.raises(ValueError,match="requires --control"):
        T.record.record(argv,candidate,output,timeout=10)
    with pytest.raises(ValueError,match="under assigned evidence root"):
        T.record.record(argv,candidate,tmp_path/"wrong.json",timeout=10,control=control)
    with pytest.raises(ValueError,match="timeout exceeds"):
        T.record.record(argv,candidate,output,timeout=901,control=control)
    result=T.record.record(argv,candidate,output,timeout=20,control=control)
    assert result["exit_code"]==0
    assert result["case_id"]=="R01" and result["time_budget_enforced_per_record"]
    assert result["assignment_sha256"]==T.common.canonical_hash(T.common.read_json(control/"assignment.json"))
    assert T.common.read_json(output)["exit_code"]==0


def test_prepared_worktree_must_never_run_record_or_review(tmp_path):
    root=tmp_path/"prepared"
    T.manage.prepare(root,["R01"],"synthetic_prepared",None)
    candidate=root/"candidates/R01"
    control=root/"control/R01"
    command=candidate/"harness/tests/wp21/runs/synthetic_prepared/R01/commands/0001.json"
    with pytest.raises(ValueError,match="assigned dispatcher"):
        T.record.record([sys.executable,"-c","print('would run')"],candidate,command,timeout=10,control=control)
    with pytest.raises(ValueError,match="unassigned prepared"):
        T.review.init_review(candidate,control,tmp_path/"should_not_exist")


def test_external_reviewer_transfers_no_candidate_selftest(dispatched,tmp_path):
    x=dispatched
    new_script=x["candidate"]/"tests/wp21/R01/candidate_faked_pass.py"
    _text(new_script,"print('candidate says PASS but reviewer must not execute')\n")
    review_dir=tmp_path/"review"
    state=T.review.init_review(x["candidate"],x["control"],review_dir)
    assert state["status"]=="REVIEW_WORKSPACE_PREPARED"
    assert "tests/wp21/R01/candidate_faked_pass.py" in state["excluded_candidate_test_paths"]
    assert not (review_dir/"workspace/tests/wp21/R01/candidate_faked_pass.py").exists()
    assert T.common.read_json(review_dir/"review.json")["checks"][0]["status"]=="NOT_RUN"
    assert not (review_dir/"review_final.json").exists()


def test_fake_single_successful_assertion_cannot_pass_remaining_checklist(dispatched,tmp_path):
    x=dispatched
    review_dir=tmp_path/"review"
    T.review.init_review(x["candidate"],x["control"],review_dir)
    script=tmp_path/"independent_test_only.py"
    _text(script,'''import json,sys\nfrom pathlib import Path\nargs=sys.argv\nout=Path(args[args.index("--wp21-assertions-output")+1])\ncheck=args[args.index("--wp21-check-id")+1]\nout.parent.mkdir(parents=True,exist_ok=True)\nout.write_text(json.dumps({"schema":"saebase.wp21.reviewer_assertions.v1","check_id":check,"assertions":[{"name":"synthetic tool plumbing check","passed":True,"observed":"synthetic fixture only"}]}))\n''')
    result=T.review.run_check(review_dir,"P01",[sys.executable,str(script)],20,"yes")
    assert result["status"]=="PASS_VERIFIED"
    final=T.review.finalize(review_dir,assets_lock=None,probe_report=None)
    assert final["status"]=="INCOMPLETE"
    assert final["hard_gates"]["all_positive_replayed"] is False
    assert final["hard_gates"]["real_model_probe_valid"] is False
    assert final["score"]["total"]<100
    with pytest.raises(ValueError,match="already finalized"):
        T.review.finalize(review_dir,assets_lock=None,probe_report=None)


def test_numerical_independent_validators_reject_wrong_metrics_and_shapes():
    b=T.behavior
    assert b.reconstruction_metrics([2,4],[1,2],measured_error=5**.5,measured_norm=5**.5)
    with pytest.raises(AssertionError):b.reconstruction_metrics([2,4],[1,2],measured_error=5.0,measured_norm=5**.5)
    assert b.mse_metrics([2,4],[1,2],measured_mse=2.5)
    with pytest.raises(AssertionError):b.mse_metrics([2,4],[1,2],measured_mse=1.25)
    with pytest.raises(AssertionError):b.close_vectors([1,2],[1])
    with pytest.raises(AssertionError):b.close_vectors([float("nan")],[0])
    assert b.intervention_invariants(old_codes=[1,2],new_codes=[0,2],replacements={0:0},
        original_activation=[1,2],edited_activation=[0,2],old_reconstruction=[1,1],new_reconstruction=[0,1])
    with pytest.raises(AssertionError):b.intervention_invariants(old_codes=[1,2],new_codes=[0,2],replacements={0:0},
        original_activation=[1,2],edited_activation=[1,2],old_reconstruction=[1,1],new_reconstruction=[0,1])


def test_lifecycle_search_and_resume_independent_invariants():
    b=T.behavior
    assert b.component_lifecycle(["old","keep"],["keep"],retired=["old"],retained=["keep"])
    with pytest.raises(AssertionError):b.component_lifecycle(["old","keep"],["old","keep"],retired=["old"],retained=["keep"])
    grid={"a":[0,1],"b":[2,3]}
    trials=[{"trial_id":str(i),"parameters":{"a":a,"b":v},"status":"completed","exit_code":0} for i,(a,v) in enumerate([(0,2),(0,3),(1,2),(1,3)])]
    assert b.grid_complete(grid=grid,trials=trials)["grid_size"]==4
    with pytest.raises(AssertionError):b.grid_complete(grid=grid,trials=trials[:-1])
    assert b.argmin_mean(trials=[{"id":"x","J":0.5},{"id":"y","J":0.5}],group_by="id",value_key="J",tie_break=["y","x"])["best"]=="y"
    with pytest.raises(AssertionError):b.resume_integrity([{"path":"x","sha256":"a"}],[{"path":"x","sha256":"b"}],reused=["x"],identity_before={"s":1},identity_after={"s":1})


def test_filesystem_and_pipeline_checker_detect_tamper(tmp_path):
    b=T.behavior
    root=tmp_path/"artifacts"
    root.mkdir()
    _text(root/"a.json","REAL_BYTES_FIXTURE_NOT_MODEL")
    items=[{"path":"a.json","size_bytes":(root/"a.json").stat().st_size,"sha256":T.common.sha256(root/"a.json")}]
    assert b.file_manifest(root,items)["files"]==1
    assert b.pipeline_integrity(root,[{"id":"A","status":"completed","path":"a.json","sha256":items[0]["sha256"]}])["completed"]==1
    with pytest.raises(AssertionError):b.pipeline_integrity(root,[{"id":"A","status":"failed"},{"id":"B","status":"completed","parents":["A"],"path":"a.json","sha256":items[0]["sha256"]}])
    _text(root/"a.json","TAMPER")
    with pytest.raises(AssertionError):b.file_manifest(root,items)


def test_summary_never_manufactures_passes(dispatched,tmp_path):
    summary=_modules()["summary"]
    result,body=summary.summarize(dispatched["root"]/"dispatch.json",tmp_path/"empty_reviews")
    assert result["counts"]=={"ASSIGNED_NOT_REVIEWED":1}
    assert "| R01 | PASS |" not in body
    assert result["cases"][0]["score"] is None
