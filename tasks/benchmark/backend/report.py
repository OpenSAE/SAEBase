"""Portable CSV and single-file HTML; no network or CDN dependencies."""

from __future__ import annotations
import csv
import html
import json
from pathlib import Path
from .results import atomic_json

COLUMNS = (
    "metric_id",
    "model_id",
    "sae_id",
    "dataset_id",
    "split",
    "condition_id",
    "status",
    "value",
    "n_requested",
    "n_valid",
    "sample_unit",
    "metric_version",
    "protocol_id",
    "source_commit",
    "source_sha256",
    "reason",
)


def write_report(folder, config, plan, results, *, provenance=None):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    atomic_json(root / "resolved_config.json", config.to_dict())
    atomic_json(root / "plan.json", plan)
    if provenance is None and (root / "provenance.json").exists():
        provenance = json.loads((root / "provenance.json").read_text())
    atomic_json(root / "provenance.json", provenance or {})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    heads = "".join(f"<th>{html.escape(x)}</th>" for x in COLUMNS)
    body = "".join(
        "<tr>"
        + "".join("<td>" + html.escape(str(row.get(k, ""))) + "</td>" for k in COLUMNS)
        + "</tr>"
        for row in rows
    )
    title = html.escape(config.name)
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width'>"
        "<style>body{font:14px system-ui;margin:3rem;line-height:1.5}"
        "table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #999;"
        "padding:6px;text-align:left}th{position:sticky;top:0;background:#eee}"
        " .scroll{overflow:auto}</style>"
        f"<title>{title} | SAEBase Benchmark</title><h1>{title}</h1>"
        f"<p>Run ID: <code>{html.escape(config.run_id)}</code></p>"
        "<p>Each metric retains its native units and interpretation. No composite SAE score.</p>"
        f"<p>{len(rows)} metric rows. Status and failed/unsupported reasons are preserved.</p>"
        f"<div class='scroll'><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div></html>"
    )
    # Full exact Localize curves are stored as checksum-bound JSON artifacts.
    # The report exposes only run-local links to verified completed results.
    localize_refs = sorted(
        {
            ref
            for row in rows
            if row["status"] == "completed" and row["metric_id"].startswith("localization.")
            for ref in row.get("measurement_refs", [])
        }
    )
    if localize_refs:
        from .adapters.localization import read_localization_artifact

        entries = []
        for ref in localize_refs:
            artifact = read_localization_artifact(root, ref)
            entries.append(
                "<li><a href='"
                + html.escape(ref, quote=True)
                + "'>"
                + html.escape(
                    artifact["model_id"] + " / " + artifact["sae_id"] + " / full exact scope curve"
                )
                + "</a>"
                + " — budget @ 90%: "
                + str(artifact["budget_at_90pct"])
                + ", fractional budget: "
                + format(artifact["budget_fraction_at_90pct"], ".5g")
                + ", features: "
                + str(artifact["n_features"])
                + "</li>"
            )
        document = document.replace(
            "</html>",
            "<section><h2>Input Localization Curves</h2><p>Verified full curves, "
            "assignments, controlled input designs and source provenance; "
            "each result describes one complete grid, not independent input states.</p><ul>"
            + "".join(entries)
            + "</ul></section></html>",
        )
    # M4 appends a checksum-verified audit without changing existing M3 rows.
    if (root / "reliability_manifest.json").exists():
        from tasks.LLM.benchmark.reliability.report import read_audit, render_audit_fragment

        attached = read_audit(root)
        document = document.replace("</html>", render_audit_fragment(attached) + "</html>")
    (root / "report.html").write_text(document, encoding="utf-8")
    return root / "report.html"
