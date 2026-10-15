"""Portable and checksum-verified M4 audit reporting; attaches to M3 HTML."""
from __future__ import annotations
import csv,hashlib,html,json
from pathlib import Path
from typing import Any

from saebase.benchmarks.results import atomic_json

AUDIT_SCHEMA="saebase.m4.audit.v1"
MANIFEST_SCHEMA="saebase.m4.audit.manifest.v1"

def _json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")

def write_audit(root:str|Path,report:dict[str,Any])->Path:
    root=Path(root)
    m3=json.loads((root/"metrics_manifest.json").read_text())
    if report.get("schema")!=AUDIT_SCHEMA or report.get("m3_run_id")!=m3["run_id"]:
        raise ValueError("M4 report is not attached to this M3 benchmark")
    data=_json_bytes(report)
    # Two-phase atomic writing. A stale manifest is always rejected on read.
    from tempfile import NamedTemporaryFile
    import os
    root.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(mode="wb",dir=root,prefix=".m4-",delete=False) as temp:
        tmp=Path(temp.name)
        temp.write(data)
        temp.flush()
        os.fsync(temp.fileno())
    try:
        os.replace(tmp,root/"reliability.json")
    finally:
        tmp.unlink(missing_ok=True)
    atomic_json(root/"reliability_manifest.json",{
        "schema":MANIFEST_SCHEMA,"m3_run_id":report["m3_run_id"],
        "m3_metrics_sha256":m3["sha256"],
        "audit_sha256":hashlib.sha256(data).hexdigest()
    })
    with (root/"reliability_summary.csv").open("w",newline="",encoding="utf-8") as f:
        fields=("model_id","sae_id","metric_id","condition_id","status",
                "n_groups","point_estimate","ci_low","ci_high","reason")
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in list(report.get("metric_audits",[]))+list(report.get("localization_audits",[])):
            stat=row.get("audit") or {}
            interval=stat.get("interval") or (None,None)
            writer.writerow({**{k:row.get(k) for k in fields},
                "n_groups":stat.get("n_groups"),
                "point_estimate":stat.get("point_estimate"),
                "ci_low":interval[0],"ci_high":interval[1]})
    (root/"reliability.html").write_text(render_audit_html(report),encoding="utf-8")
    return root/"reliability.json"

def read_audit(root:str|Path)->dict:
    root=Path(root)
    manifest=json.loads((root/"reliability_manifest.json").read_text())
    m3=json.loads((root/"metrics_manifest.json").read_text())
    if manifest.get("schema")!=MANIFEST_SCHEMA or manifest.get("m3_run_id")!=m3["run_id"]:
        raise ValueError("M4 audit identity or schema mismatch")
    if manifest.get("m3_metrics_sha256")!=m3["sha256"]:
        raise ValueError("M3 metrics changed since audit; rerun M4")
    data=(root/"reliability.json").read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest["audit_sha256"]:
        raise ValueError("M4 audit integrity mismatch")
    report=json.loads(data)
    if report.get("schema")!=AUDIT_SCHEMA or report.get("m3_run_id")!=m3["run_id"]:
        raise ValueError("M4 audit content identity mismatch")
    truth=report.get("ground_truth")
    if truth is not None:
        from saebase.benchmarks.adapters.papers import source_identity
        for paper in ("sar","rss"):
            source=truth["source_provenance"][paper]
            if source_identity(source["path"],paper)!=source["sha256"]:
                raise ValueError(f"M4 ground truth {paper} source changed since audit")
    return report

def render_audit_fragment(report)->str:
    headers=("metric_id","sae_id","condition_id","status","n_groups",
             "point_estimate","95% or configured CI","reason")
    cells=[]
    for row in report.get("metric_audits",[]):
        stat=row.get("audit") or {}
        interval=stat.get("interval")
        values=(row.get("metric_id"),row.get("sae_id"),row.get("condition_id"),
                row.get("status"),stat.get("n_groups"),stat.get("point_estimate"),
                interval,row.get("reason"))
        cells.append("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in values)+"</tr>")
    localize_audits=report.get("localization_audits",[])
    localize_comparisons=report.get("localization_comparisons",[])
    localization_html=""
    if localize_audits:
        localization_html="<h3>Input localization reliability</h3>"
        localization_html+=("<p>One complete controlled grid is one evaluation unit. "
                            "Finite paired-input resampling across seeds measures estimator "
                            "stability conditional on that fixed grid, not a confidence interval "
                            "over independent prompts or evidence that one SAE is better.</p>")
        localization_html+="<table><thead><tr><th>Metric</th><th>SAE</th><th>Status</th>"
        localization_html+="<th>Exact budget</th><th>Paired sampling status</th></tr></thead><tbody>"
        for row in localize_audits:
            audit=row.get("audit") or {}
            sampling=audit.get("sampling_stability") or {}
            local_cells=(row.get("metric_id"),row.get("sae_id"),row.get("status"),
                         audit.get("point_estimate"),sampling.get("status"))
            localization_html+="<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in local_cells)+"</tr>"
        localization_html+="</tbody></table>"
        localization_html+="<p>Complete seed results are stored in reliability.json.</p>"
    if localize_comparisons:
        localization_html+="<h3>Matched Localize SAE comparisons</h3>"
        localization_html+=("<p>Contextual dependency-budget differences only; "
                             "feature counts and design identity are retained. No SAE quality ranking.</p>")
        localization_html+="<pre>"+html.escape(json.dumps(localize_comparisons,indent=2,allow_nan=False))+"</pre>"
    truth=report.get("ground_truth")
    extra=""
    if truth:
        fields=[]
        for label,details in sorted(truth.get("factor_cases",{}).items()):
            fields.append((f"Factors: {label}",details["status"]))
        for label,details in sorted(truth.get("known_interaction_cases",{}).items()):
            fields.append((f"SAR order: {label}",details["status"]))
        fields.append(("RSS direction and held-out",truth["known_rss_direction_case"]["status"]))
        for label,details in sorted(truth.get("negative_invariance_controls",{}).items()):
            fields.append((f"Control: {label}",details["status"]))
        extra="<h3>Synthetic ground truth and negative controls</h3>"
        extra+="<p>Analytical fixtures, independent calibration and test. Not a claim about real SAE quality.</p>"
        extra+="<table><tr><th>Known-truth case</th><th>Status</th></tr>"
        extra+="".join("<tr><td>"+html.escape(x)+"</td><td>"+html.escape(y)+"</td></tr>" for x,y in fields)
        extra+="</table><p>Full seeds, effect sizes, intervals and original source hashes are in reliability.json.</p>"
    return ("<section id='reliability-audit'><h2>M4 Reliability Audit</h2>"
            "<p>Paired/grouped evaluation resampling of frozen M3 measurements. "
            "Intervals are descriptive, not causal claims. No new model forwards.</p>"
            "<table><thead><tr>"+"".join("<th>"+html.escape(x)+"</th>" for x in headers)
            +"</tr></thead><tbody>"+"".join(cells)+"</tbody></table>"+extra+
            "<h3>Paired comparisons and rank stability</h3><pre>"
            +html.escape(json.dumps({"comparisons":report.get("paired_comparisons",[]),
                   "rank_stability":report.get("rank_stability",[])},indent=2,allow_nan=False))
            +"</pre>"+localization_html+"</section>")

def render_audit_html(report):
    return ("<!doctype html><html><meta charset='utf-8'>"
            "<style>body{font:14px system-ui;margin:2rem;line-height:1.5}"
            "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.5rem}"
            "pre{white-space:pre-wrap}</style>"
            "<title>SAEBase M4 Reliability Audit</title><h1>SAEBase M4 Reliability Audit</h1>"
            +render_audit_fragment(report)+"</html>")
