"""Portable CSV and single-file HTML; no network or CDN dependencies."""
from __future__ import annotations
import csv,html,json
from pathlib import Path
from .results import atomic_json

COLUMNS=("metric_id","model_id","sae_id","dataset_id","split","condition_id",
         "status","value","n_requested","n_valid","sample_unit","metric_version","protocol_id",
         "source_commit","source_sha256","reason")

def write_report(folder,config,plan,results,*,provenance=None):
    root=Path(folder);root.mkdir(parents=True,exist_ok=True)
    rows=[r.to_dict() if hasattr(r,"to_dict") else r for r in results]
    atomic_json(root/"resolved_config.json",config.to_dict())
    atomic_json(root/"plan.json",plan)
    atomic_json(root/"provenance.json",provenance or {})
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    heads="".join(f"<th>{html.escape(x)}</th>" for x in COLUMNS)
    body="".join("<tr>"+"".join("<td>"+html.escape(str(row.get(k,"")))+"</td>" for k in COLUMNS)+"</tr>" for row in rows)
    title=html.escape(config.name)
    document=("<!doctype html><html lang='en'><meta charset='utf-8'>"
              "<meta name='viewport' content='width=device-width'>"
              "<style>body{font:14px system-ui;margin:3rem;line-height:1.5}"
              "table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #999;"
              "padding:6px;text-align:left}th{position:sticky;top:0;background:#eee}"
              " .scroll{overflow:auto}</style>"
              f"<title>{title} | SAEBase Benchmark</title><h1>{title}</h1>"
              f"<p>Run ID: <code>{html.escape(config.run_id)}</code></p>"
              "<p>Each metric retains its native units and interpretation. No composite SAE score.</p>"
              f"<p>{len(rows)} metric rows. Status and failed/unsupported reasons are preserved.</p>"
              f"<div class='scroll'><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div></html>")
    (root/"report.html").write_text(document,encoding="utf-8")
    return root/"report.html"
