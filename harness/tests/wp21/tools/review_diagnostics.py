"""Write additive WP21 review diagnostics without modifying frozen grades.

Final reviews remain immutable. Multiple independently supported findings can
coexist (e.g. a training precision failure and a directory evidence audit bug).
The optional findings input must come from a reviewer, not the candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_json, sha256, write_json
from archive_review import verify_archive

CAUSES = {"CONTRACT_GAP", "SELECTION_GAP", "IMPLEMENTATION_ERROR",
          "ENFORCEMENT_GAP", "EVALUATOR_SCOPE_GAP", "ENV_BLOCKED"}
OWNERS = {"baseline_product", "candidate", "governance", "environment", "evaluator"}


def diagnostics(final_path: Path, *, findings_path: Path | None = None,
                reviewer_archive: Path | None = None,
                quality_assessment_path: Path | None = None) -> dict:
    record = read_json(final_path)
    if record.get("schema") != "saebase.wp21.final_review.v1":
        raise ValueError("only a finalized WP21 review may be diagnosed")
    findings = []
    if findings_path:
        input_data = read_json(findings_path)
        if input_data.get("schema") != "saebase.wp21.review_findings.v1" or input_data.get("case_id") != record.get("case_id"):
            raise ValueError("findings case/schema mismatch")
        findings = input_data.get("findings")
        if not isinstance(findings,list):
            raise ValueError("findings list required")
        for row in findings:
            if (not isinstance(row,dict) or row.get("cause") not in CAUSES or row.get("owner") not in OWNERS
                    or not isinstance(row.get("description"),str) or len(row["description"].strip())<10
                    or not isinstance(row.get("evidence_refs"),list) or not row["evidence_refs"]
                    or not all(isinstance(ref,str) and ref.strip() for ref in row["evidence_refs"])):
                raise ValueError("each finding requires a supported cause, owner, description and evidence_refs")
    quality = None
    if quality_assessment_path:
        quality = read_json(quality_assessment_path)
        if quality.get("schema") != "saebase.wp21.engineering_quality.v1" or quality.get("case_id") != record["case_id"]:
            raise ValueError("engineering quality assessment identity mismatch")
        metrics = quality.get("dimensions")
        if not isinstance(metrics,dict) or set(metrics) != {"code_maintainability","robustness","documentation_accuracy"}:
            raise ValueError("engineering review requires all three independent dimensions")
        for value in metrics.values():
            if not isinstance(value,dict) or value.get("status") not in {"UNREVIEWED","REVIEWED_SATISFACTORY","REVIEWED_WITH_FINDINGS"}:
                raise ValueError("invalid engineering review dimension status")
            if value["status"] != "UNREVIEWED" and (not isinstance(value.get("evidence_refs"),list) or not value["evidence_refs"]):
                raise ValueError("reviewed dimension requires concrete evidence references")
    source_archive = None
    if reviewer_archive:
        import zipfile
        source_archive=verify_archive(reviewer_archive)
        if source_archive.get("case_id") != record.get("case_id") or source_archive.get("run_id") != record.get("run_id"):
            raise ValueError("review provenance archive does not match final review")
        with zipfile.ZipFile(reviewer_archive) as archive:
            import hashlib
            if hashlib.sha256(archive.read("review_final.json")).hexdigest() != sha256(final_path):
                raise ValueError("review provenance archives a different final review")
    checks=record.get("checks",{})
    score=record.get("score",{})
    has_failed_checks=(checks.get("positive",0)!=checks.get("positive_total",0) or
                       checks.get("negative",0)!=checks.get("negative_total",0))
    has_evidence_gap=(record.get("candidate_evidence_status")!="EVIDENCE_READY_FOR_REPLAY")
    return {"schema":"saebase.wp21.review_diagnostics.v1",
            "case_id":record["case_id"],"run_id":record["run_id"],
            "frozen_review_sha256":sha256(final_path),"frozen_status":record["status"],
            "frozen_total":score.get("total"),"frozen_primary_attribution":record.get("attribution"),
            "observed_gates":{"incomplete_positive_or_negative_replay":has_failed_checks,
                              "evidence_gate_rejected":has_evidence_gap,
                              "reviewer_provenance_archive_present":source_archive is not None},
            "reviewer_provenance":source_archive,
            "additional_findings":findings,
            "frozen_dimensions":{"functionality":score.get("real_functionality"),
                                 "negative_and_compatibility":score.get("compatibility_and_negative"),
                                 "evidence_reproducibility":score.get("evidence_reproducibility"),
                                 "scope_and_documentation_gate":score.get("scope_and_documentation")},
            "engineering_quality":quality if quality is not None else {"status":"NOT_REVIEWED_SEPARATELY"},
            "interpretation":"The 100-point total measures frozen-check completion, not general code quality or scientific validity. The 15-point scope/documentation gate is not a manual assessment of documentation accuracy. Findings are additive and never revise a previous PASS/FAIL or its original score.",
            "limitations":["A reviewer archive is an integrity aid, not proof of genuine model computation.",
                           "Evidence not provided here, including raw GPU tensors, must be independently replayed in the execution environment."]}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-final",type=Path,required=True)
    p.add_argument("--findings",type=Path)
    p.add_argument("--archive",type=Path)
    p.add_argument("--quality-assessment",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        result=diagnostics(a.review_final,findings_path=a.findings,reviewer_archive=a.archive,
                           quality_assessment_path=a.quality_assessment)
        write_json(a.output,result,exclusive=True)
        print(json.dumps({"status":"DIAGNOSTICS_WRITTEN_NOT_REGRADED","path":str(a.output),
                          "case_id":result["case_id"],"findings":len(result["additional_findings"])},ensure_ascii=False))
        return 0
    except (OSError,KeyError,TypeError,ValueError) as exc:
        print(json.dumps({"status":"DIAGNOSTICS_ERROR","reason":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=="__main__":raise SystemExit(main())
