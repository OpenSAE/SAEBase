"""Capture-owned question, feature and interaction reports.

Reads Dataset-owned questions and Capture's existing annotation-aware reader.
Exports derived views only; no weights, activations or measurements are changed.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value
from tasks.capture.configuration import capture_section

from collections import Counter
import html
import json
from pathlib import Path


def explanation_text(feature):
    annotation = feature.get("neuronpedia", {})
    descriptions = [
        item["description"]
        for item in annotation.get("explanations", [])
        if item.get("description")
    ]
    return ";".join(descriptions) if descriptions else annotation.get("reason", "No public explanation obtained")


def model_answer_text(answer):
    if answer is None:
        return "The old results have yet to record the model answers"
    if answer["status"] != "completed":
        return "Generate incomplete:" + (
            answer.get("reason") or answer.get("error", {}).get("message", "The reason is in the original record.")
        )
    return answer.get("text") or "(Model ends without generating text)"


def export_report(folder: str | Path) -> dict:
    from omegaconf import OmegaConf
    from saebase.datasets.selection import read_selection
    from saebase.runtime.experiment_results import write_json
    from tasks.capture.annotations import read_samples

    root = Path(folder).resolve()
    manifest = json.loads((root / "metadata/manifest.json").read_text())
    cfg = OmegaConf.create(manifest["configuration"])
    summary = (manifest.get("result") or {}).get("summary", {})
    selected = read_selection(cfg.dataset, str(cfg.model.name))
    observations = {row["sample_id"]: row for row in read_samples(root)}
    is_single = task_value(cfg.task, "mode") == "g_s"
    is_exhaustive = task_value(cfg.task, "mode") == "I" or (
        task_value(cfg.task, "mode") == "sar"
        and capture_section(cfg.task, "sar").get("algorithm") == "exhaustive"
    )
    questions = []
    empty_candidates = set()
    for sample in selected:
        recorded = observations.get(sample["sample_id"], {})
        result = recorded.get("result", {})
        if result.get("candidate_shortfall") and not result["candidate_feature_ids"]:
            empty_candidates.add(sample["sample_id"])
        source = sample.get("source_record", {})
        questions.append(
            {
                "sample_id": sample["sample_id"],
                "question": source.get("question", sample["prompt"]),
                "answer": (result.get("g_s") or result.get("sar") or {})
                .get("identity", {})
                .get("behavior", {})
                .get("reference", {})
                .get("text", source.get("answer", sample.get("reference_answer"))),
                "model_answer": result.get("model_answer"),
                "prompt": sample["prompt"],
                "split": sample.get("split"),
                "status": recorded.get("status", "pending"),
                "error": recorded.get("error") or recorded.get("reason"),
                "features": result.get("top_features", []),
                "pairs": result.get("sar", {}).get("pairwise", {}).get("top_pairs", []),
                "sar_metrics": result.get("sar", {}).get("metrics", {}),
                "pair_fit": {
                    k: v
                    for k, v in result.get("sar", {}).get("pairwise", {}).items()
                    if k != "top_pairs"
                },
                "sar_artifact": result.get("sar", {}).get("artifact"),
                "g_s": result.get("g_s"),
                "measured_states": (result.get("g_s") or result.get("sar") or {}).get(
                    "measured_states"
                ),
            }
        )
    payload = {
        "schema": "saebase.capture.question_report.v1",
        "run_dir": str(root),
        "dataset": str(cfg.dataset.name),
        "model": cfg.model.name,
        "sae": cfg.task.get("sae_key") or (cfg.model.get("default_saes") or [None])[0],
        "layer": summary.get(
            "layer",
            cfg.task.get("layer")
            if cfg.task.get("layer") is not None
            else next(iter(cfg.model.get("default_layers", {}).values()), None),
        ),
        "requested": len(questions),
        "recorded": len(observations),
        "statuses": dict(Counter(q["status"] for q in questions)),
        "feature_ranking": "signed_exhaustive_interaction_descending"
        if is_exhaustive
        else "descending_deleted_answer_mean_logprob"
        if is_single
        else "absolute_zscore"
        if task_value(cfg.task, "mode") == "whiten"
        or (
            task_value(cfg.task, "mode") == "sar"
            and capture_section(cfg.task, "selection").get("feature_selection") == "whiten"
        )
        else "absolute_native_activation",
        "pair_ranking": "signed_exhaustive_interaction_descending"
        if is_exhaustive
        else "signed_pairwise_coefficient_descending"
        if capture_section(cfg.task, "sar").get("pair_ranking", "absolute") == "signed"
        else "absolute_pairwise_coefficient",
        "explanation_sources": {
            "features": "Neuronpedia Public explanation, retention of original language and source field",
            "pairs": "Four-tempered, solid background."
            if is_exhaustive
            else "An interactive direction statement for the proposed second stage of combined intervention; no impersonation Neuronpedia The character against the interpretation",
        },
        "questions": questions,
    }
    output = root / "reports"
    output.mkdir(exist_ok=True)
    write_json(output / "question_features_pairs.json", payload)
    from tasks.capture.selection import settings

    selected_count = settings(cfg)["capture_count"]
    feature_title = (
        "After Delete g(S)"
        if is_single
        else "Whiten"
        if payload["feature_ranking"] == "absolute_zscore"
        else "Native Activation"
    ) + f" Top {selected_count} Features"
    if is_exhaustive:
        feature_title = "Highest interactive pair Endpoint Character"
    pair_title = f"{'I' if is_exhaustive else 'SAR'} Top {selected_count if is_exhaustive else capture_section(cfg.task, 'sar').get('top_pairs', 0)} Match"
    pair_order = (
        "Actual I Original value reduction order"
        if is_exhaustive
        else "Factor Decline Order"
        if payload["pair_ranking"] == "signed_pairwise_coefficient_descending"
        else "Decline in absolute value of coefficient"
    )
    sections = []
    markdown = [
        "# Title, Whiten Characteristics and SAR Match",
        "",
        f"Model:{payload['model']};SAE:{payload['sae']};Layer:{payload['layer']}.",
        f"Recorded {payload['recorded']}/{payload['requested']} Question.:{payload['statuses']}.",
        "",
        f"Characteristic interpretation from Neuronpedia,Keeps its original public language. Sorting in pairs:{pair_order}."
        + ("I=g11−g10−g01+g00,Maximum value after fixed background for each pair." if is_exhaustive else "")
        + ("g(S)To average the answer after only deleting the feature log Probability; Δg Relative original total reservation." if is_single else ""),
        "",
    ]

    def number(value):
        return "Undefined" if value is None else f"{value:.6g}"

    def cell(value):
        return html.escape(str(value))

    displayed = (
        question for question in questions if question["sample_id"] not in empty_candidates
    )
    for index, q in enumerate(displayed, 1):
        feature_rows = []
        pair_rows = []
        markdown.extend(
            [
                f"## {index}. {q['question']}",
                "",
                f"Sample:{q['sample_id']};Status:{q['status']}.",
                *([f"Reason:{q['error']}"] if q["error"] else []),
                f"Standard answer:{q['answer']}",
                "",
                "Model Original Answer:",
                "",
                *(("> " + line) for line in model_answer_text(q.get("model_answer")).splitlines()),
                "",
                f"Answer Status:{(q.get('model_answer') or {}).get('status', 'not_recorded')};Quantity:{(q.get('model_answer') or {}).get('finish_reason', 'Unrecorded')}.",
                "",
                "| Features ID | Original Activated | Whiten Score | g(S) | Δg | Neuronpedia Explanation |"
                if is_single
                else "| Features ID | Original Activated | Whiten Score | Neuronpedia Explanation |",
                "|---|---:|---:|---:|---:|---|" if is_single else "|---|---:|---:|---|",
            ]
        )
        for feature in q["features"]:
            values = [
                feature["feature_id"],
                number(feature.get("activation")),
                number(feature.get("zscore")),
                *([number(feature.get("g")), number(feature.get("delta_g"))] if is_single else []),
                explanation_text(feature),
            ]
            feature_rows.append(
                "<tr>" + "".join("<td>" + cell(v) + "</td>" for v in values) + "</tr>"
            )
            markdown.append(
                "| "
                + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values)
                + " |"
            )
        markdown.extend(
            [
                "",
                "| Characteristics Yes | Actual I | Open interpretation of both-end features |"
                if is_exhaustive
                else "| Characteristics Yes | Interactive coefficient | Open interpretation of both-end features |",
                "|---|---:|---|",
            ]
        )
        for pair in q["pairs"]:
            explanations = ";".join(
                f"{f['feature_id']}:{explanation_text(f)}" for f in pair.get("features", [])
            )
            values = [
                " × ".join(map(str, pair["feature_ids"])),
                number(pair["coefficient"]),
                explanations,
            ]
            pair_rows.append("<tr>" + "".join("<td>" + cell(v) + "</td>" for v in values) + "</tr>")
            markdown.append(
                "| "
                + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values)
                + " |"
            )
        metrics = ";".join(
            f"{k}={number(q['sar_metrics'].get(k))}" for k in ("sar1", "sar2", "rho2", "var")
        )
        if is_single or is_exhaustive:
            metrics = f"Number of measured combinations:{q['measured_states']}"
        error = f"<p>{cell(q['error'])}</p>" if q["error"] else ""
        sections.append(
            f"<details><summary>{index}. {cell(q['question'])} · {cell(q['status'])}</summary>"
            f"<p>Sample:{cell(q['sample_id'])};Standard answer:{cell(q['answer'])}</p>{error}"
            f"<h3>Model Original Answer</h3><pre>{cell(model_answer_text(q.get('model_answer')))}</pre>"
            f"<p>Answer Status:{cell((q.get('model_answer') or {}).get('status', 'not_recorded'))};Quantity:{cell((q.get('model_answer') or {}).get('finish_reason', 'Unrecorded'))}</p>"
            f"<h3>{cell(feature_title)}</h3><table><tr><th>ID</th><th>Original Activated</th><th>Whiten Score</th>{'<th>g(S)</th><th>Δg</th>' if is_single else ''}<th>Neuronpedia Public explanation</th></tr>{''.join(feature_rows)}</table>"
            f"<h3>{cell(pair_title)}</h3><table><tr><th>Characteristics Yes</th><th>{'Actual I' if is_exhaustive else 'Interactive coefficient'}</th><th>Both ends of the public explanation</th></tr>{''.join(pair_rows)}</table>"
            f"<p>{cell(q['pair_fit'].get('reason', ''))}</p>"
            f"<p>{cell(metrics)}"
            + (
                ""
                if is_single or is_exhaustive
                else f";It’s about time.:{cell(q['pair_fit'].get('fit_rank'))}/{cell(q['pair_fit'].get('fit_parameters'))}"
            )
            + "</p>"
            f"<details><summary>Full Model Input</summary><pre>{cell(q['prompt'])}</pre></details></details>"
        )
        markdown.extend(["", metrics, ""])
    page = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Whiten Characteristics and SAR Match</title>"
        "<style>body{font:16px system-ui;margin:2rem;line-height:1.6;max-width:1400px}details{margin:1rem 0;padding:.7rem;border:1px solid #ccc}summary{cursor:pointer}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem;text-align:left}pre{white-space:pre-wrap}</style>"
        f"<h1>{cell(payload['dataset'])}:Characteristics and SAR Match</h1><p>Model:{cell(payload['model'])};SAE:{cell(payload['sae'])};Layer:{cell(payload['layer'])}.Recorded {len(observations)}/{len(questions)} Question.</p>"
        f"<p>Characteristic interpretation from Neuronpedia,Keeps the original. Sorting pairs:{pair_order};g(S)Average answer after deletion for single feature log Probability.</p>"
        + "".join(sections)
        + "</html>"
    )
    (output / "question_features_pairs.html").write_text(page)
    (output / "question_features_pairs.md").write_text("\n".join(markdown))
    return {
        "requested": len(questions),
        "recorded": len(observations),
        "statuses": payload["statuses"],
        "html": str(output / "question_features_pairs.html"),
        "json": str(output / "question_features_pairs.json"),
        "markdown": str(output / "question_features_pairs.md"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(export_report(parser.parse_args().run_dir), ensure_ascii=False, indent=2))
