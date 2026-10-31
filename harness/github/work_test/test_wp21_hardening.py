"""WP21.2 offline regression tests. Every simulated case is explicitly synthetic."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile

import pytest
from test_wp21_local_suite import T, ROOT, TOOLS, _json, _text


def load_module(name):
    # Normal module import so archive_review can import the trusted modules.
    sys.path.insert(0, str(TOOLS))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("dirname", ["native_training_logs/", "benchmark_output/"])
def test_directory_artifacts_require_real_nonempty_directory(tmp_path, dirname):
    evidence = tmp_path / "evidence"
    experiment = tmp_path / "experiment"
    evidence.mkdir()
    experiment.mkdir()
    name = dirname.rstrip("/")
    validate = T.audit_evidence.required_artifact_present
    _text(evidence / (name + "_backup") / "log.txt", "unrelated")
    assert validate(dirname, evidence=evidence, workspace=tmp_path, experiment_files=[]) is False
    _text(evidence / name / "notes.txt", "real bytes, synthetic fixture")
    assert validate(dirname, evidence=evidence, workspace=tmp_path, experiment_files=[]) is True
    (evidence / name / "notes.txt").unlink()
    assert validate(dirname, evidence=evidence, workspace=tmp_path, experiment_files=[]) is False
    _text(tmp_path / "experiment/run_01" / name / "trace.json", '{"kind":"synthetic"}')
    files = T.audit_evidence.artifacts(tmp_path)
    assert validate(dirname, evidence=evidence, workspace=tmp_path, experiment_files=files) is True
    assert (
        validate(name + "_short/", evidence=evidence, workspace=tmp_path, experiment_files=files)
        is False
    )


def test_directory_artifact_rejects_symlink_even_when_staying_inside_root(tmp_path):
    evidence = tmp_path / "evidence"
    (evidence / "native_training_logs").mkdir(parents=True)
    _text(evidence / "raw.txt", "synthetic fixture")
    (evidence / "native_training_logs" / "log.txt").symlink_to(evidence / "raw.txt")
    (tmp_path / "experiment").mkdir()
    with pytest.raises(ValueError, match="symlink"):
        T.audit_evidence.required_artifact_present(
            "native_training_logs/", evidence=evidence, workspace=tmp_path, experiment_files=[]
        )


@pytest.mark.parametrize("bad", ["../native_training_logs/", "/etc/", ".//", "./output/", "C:/x/"])
def test_required_artifact_rejects_bad_paths(tmp_path, bad):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "experiment").mkdir()
    with pytest.raises(ValueError):
        T.audit_evidence.required_artifact_present(
            bad, evidence=tmp_path / "evidence", workspace=tmp_path, experiment_files=[]
        )


def test_nonzero_child_is_recorded_without_aborting_cli_loop(tmp_path, monkeypatch, capsys):
    output = tmp_path / "command.json"
    args = [
        "record.py",
        "--cwd",
        str(tmp_path),
        "--output",
        str(output),
        "--timeout",
        "15",
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert T.record.main() == 0
    assert T.common.read_json(output)["exit_code"] == 7
    other = tmp_path / "legacy.json"
    monkeypatch.setattr(
        sys,
        "argv",
        args[:1] + ["--propagate-exit"] + args[1:5] + ["--output", str(other)] + args[7:],
    )
    # Explicit legacy mode preserves old shell behavior.
    assert T.record.main() == 7
    assert T.common.read_json(other)["exit_code"] == 7


def test_preflight_all_distinguishes_pythia_and_model_only(tmp_path, monkeypatch):
    calls = []

    def fake_preflight(case_id, *, model_snapshot=None, sae_snapshot=None):
        calls.append((case_id, model_snapshot, sae_snapshot))
        blocked = case_id == "R08" and model_snapshot is None
        return {
            "status": "ENV_BLOCKED" if blocked else "PREREQUISITES_PRESENT_UNVERIFIED",
            "blockers": ["missing_pythia"] if blocked else [],
            "model_snapshot": str(model_snapshot) if model_snapshot else None,
            "sae_snapshot": str(sae_snapshot) if sae_snapshot else None,
        }

    monkeypatch.setattr(T.preflight, "preflight", fake_preflight)
    gemma = tmp_path / "gemma"
    sae = tmp_path / "sae"
    result = T.preflight.preflight_all(model_snapshot=gemma, sae_snapshot=sae)
    assert result["counts"] == {"ready_for_probe": 19, "env_blocked": 1}
    assert result["blocked_cases"] == ["R08"]
    by_id = {c[0]: c for c in calls}
    assert by_id["R08"] == ("R08", None, None)
    assert by_id["R09"] == ("R09", gemma, None)
    assert by_id["R17"] == ("R17", gemma, None)
    assert by_id["R01"] == ("R01", gemma, sae)
    calls.clear()
    result = T.preflight.preflight_all(
        model_snapshot=gemma, sae_snapshot=sae, pythia_snapshot=tmp_path / "pythia"
    )
    assert result["counts"]["env_blocked"] == 0


def test_probe_import_path_works_from_arbitrary_cwd(tmp_path):
    script = """import runpy,sys; runpy.run_path(sys.argv[1]); print('ROOT_IN_PATH',sys.argv[2] in sys.path)"""
    out = subprocess.run(
        [sys.executable, "-c", script, str(TOOLS / "probe.py"), str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "ROOT_IN_PATH True" in out.stdout


@pytest.fixture
def review_for_synthetic_archive(tmp_path):
    dispatch = tmp_path / "dispatch"
    T.manage.prepare(dispatch, ["R01"], "synthetic_archive", "SYNTHETIC_TEST_AUTHORIZATION_ONLY")
    candidate = dispatch / "candidates/R01"
    control = dispatch / "control/R01"
    review = tmp_path / "review"
    T.review.init_review(candidate, control, review)
    script = tmp_path / "examiner_owned.py"
    _text(
        script,
        """import json,sys\nfrom pathlib import Path\na=sys.argv\np=Path(a[a.index('--wp21-assertions-output')+1])\np.write_text(json.dumps({'schema':'saebase.wp21.reviewer_assertions.v1','check_id':a[a.index('--wp21-check-id')+1],'assertions':[{'name':'synthetic plumbing','passed':True,'observed':{'fixture':1}}]}))\n""",
    )
    T.review.run_check(review, "P01", [sys.executable, str(script)], 20, "yes")
    final = T.review.finalize(review, assets_lock=None, probe_report=None)
    assert final["status"] == "INCOMPLETE"
    return review, script


def test_review_archiving_contains_scripts_assertions_logs_and_excludes_workspace(
    review_for_synthetic_archive, tmp_path
):
    review, script = review_for_synthetic_archive
    arch = load_module("archive_review")
    dst = tmp_path / "proof.zip"
    result = arch.create_archive(review, dst)
    assert result["independent_scripts"] == 1
    assert arch.verify_archive(dst)["status"] == "REVIEW_ARCHIVE_INTEGRITY_VALID_NOT_REPLAYED"
    with zipfile.ZipFile(dst) as z:
        names = set(z.namelist())
        assert "review_final.json" in names and "review.json" in names
        assert any(x.startswith("reviewer_scripts/") for x in names)
        assert "records/P01/0001.assertions.json" in names
        assert "records/P01/0001.stdout.log" in names
        assert not any(x.startswith("workspace/") for x in names)
    _text(script, "# mutated after the exam")
    with pytest.raises(ValueError, match="script"):
        arch.create_archive(review, tmp_path / "new_proof.zip")


def test_review_archive_rejects_undocumented_members(review_for_synthetic_archive, tmp_path):
    review, _ = review_for_synthetic_archive
    arch = load_module("archive_review")
    good = tmp_path / "good.zip"
    arch.create_archive(review, good)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("unexpected.txt", "added after archiving")
    with pytest.raises(ValueError, match="unlisted"):
        arch.verify_archive(bad)


def test_additive_multi_cause_diagnostics_keep_frozen_grade(review_for_synthetic_archive, tmp_path):
    review, _ = review_for_synthetic_archive
    arch = load_module("archive_review")
    diagnosis = load_module("review_diagnostics")
    path = review / "review_final.json"
    original = path.read_bytes()
    report = tmp_path / "archive.zip"
    arch.create_archive(review, report)
    findings = tmp_path / "findings.json"
    _json(
        findings,
        {
            "schema": "saebase.wp21.review_findings.v1",
            "case_id": "R01",
            "findings": [
                {
                    "cause": "IMPLEMENTATION_ERROR",
                    "owner": "baseline_product",
                    "description": "Synthetic example of a distinct product issue",
                    "evidence_refs": ["synthetic training log id 01"],
                },
                {
                    "cause": "EVALUATOR_SCOPE_GAP",
                    "owner": "evaluator",
                    "description": "Synthetic example of a distinct audit issue",
                    "evidence_refs": ["synthetic audit error id 02"],
                },
            ],
        },
    )
    result = diagnosis.diagnostics(path, findings_path=findings, reviewer_archive=report)
    assert len(result["additional_findings"]) == 2
    assert result["frozen_total"] < 100
    assert path.read_bytes() == original
    assert result["reviewer_provenance"]["case_id"] == "R01"


def test_training_repro_computes_frozen_tolerance_and_failing_elements(tmp_path):
    torch = pytest.importorskip("torch")
    tool = load_module("training_repro")
    a, b = tmp_path / "seed7a.pt", tmp_path / "seed7b.pt"
    torch.save({"W_dec": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, a)
    torch.save({"W_dec": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, b)
    good = tool.compare_checkpoints(a, b)
    assert good["status"] == "WITHIN_FROZEN_TOLERANCE"
    torch.save({"W_dec": torch.tensor([[1.0, 2.0], [3.0, 4.083]])}, b)
    wrong = tool.compare_checkpoints(a, b)
    assert wrong["status"] == "OUTSIDE_FROZEN_TOLERANCE"
    assert wrong["comparison"]["failing_elements"] == 1
    assert wrong["comparison"]["max_abs"] > 0.08
    assert wrong["rtol"] == 1e-4 and wrong["atol"] == 1e-5


def test_frozen_exam_materials_untouched():
    assert T.manage.check_spec()["status"] == "SPEC_VALID"


def test_device_truth_does_not_confuse_requested_cuda_with_actual_cpu():
    value = T.behavior.observed_device(configured="cuda", actual="cpu", requires_cuda=False)
    assert value["configured_matches_actual"] is False
    with pytest.raises(AssertionError, match="actually"):
        T.behavior.observed_device(configured="cuda", actual="cpu", requires_cuda=True)
    assert T.behavior.observed_device(configured="cuda:0", actual="cuda:0", requires_cuda=True)[
        "configured_matches_actual"
    ]


def test_optional_engineering_quality_is_separate_from_frozen_score(
    review_for_synthetic_archive, tmp_path
):
    review, _ = review_for_synthetic_archive
    analysis = load_module("review_diagnostics")
    quality = tmp_path / "quality.json"
    _json(
        quality,
        {
            "schema": "saebase.wp21.engineering_quality.v1",
            "case_id": "R01",
            "dimensions": {
                "code_maintainability": {"status": "UNREVIEWED"},
                "robustness": {
                    "status": "REVIEWED_WITH_FINDINGS",
                    "evidence_refs": ["synthetic source review #1"],
                },
                "documentation_accuracy": {"status": "UNREVIEWED"},
            },
        },
    )
    result = analysis.diagnostics(review / "review_final.json", quality_assessment_path=quality)
    assert (
        result["engineering_quality"]["dimensions"]["robustness"]["status"]
        == "REVIEWED_WITH_FINDINGS"
    )
    assert result["frozen_dimensions"]["functionality"] == 8.0
    assert result["frozen_total"] == 8.0
    quality_data = json.loads(quality.read_text())
    quality_data["dimensions"]["robustness"]["evidence_refs"] = []
    _json(quality, quality_data)
    with pytest.raises(ValueError, match="evidence references"):
        analysis.diagnostics(review / "review_final.json", quality_assessment_path=quality)


def test_summary_exposes_frozen_dimensions_and_failure_details_without_inventing_pass(tmp_path):
    summary = load_module("summary")
    dispatch = tmp_path / "dispatch"
    T.manage.prepare(dispatch, ["R01"], "synthetic_summary", "SYNTHETIC_TEST_AUTHORIZATION_ONLY")
    reviews = tmp_path / "reviews"
    final = reviews / "R01/review_final.json"
    _json(
        final,
        {
            "schema": "saebase.wp21.final_review.v1",
            "case_id": "R01",
            "run_id": "synthetic_summary",
            "status": "INCOMPLETE",
            "score": {"total": 65, "real_functionality": 40},
            "attribution": "EVALUATOR_SCOPE_GAP",
            "blockers": ["synthetic missing evidence"],
            "failed_check_ids": ["N02"],
            "candidate_evidence_rejection_reason": "synthetic directory audit",
        },
    )
    result, body = summary.summarize(dispatch / "dispatch.json", reviews)
    row = result["cases"][0]
    assert row["status"] == "INCOMPLETE" and row["score"] == 65
    assert row["failed_check_ids"] == ["N02"]
    assert row["frozen_score_dimensions"]["real_functionality"] == 40
    assert "EVALUATOR_SCOPE_GAP" in body and "N02" in body
    assert "| R01 | PASS |" not in body
