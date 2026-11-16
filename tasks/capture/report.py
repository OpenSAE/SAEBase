"""Capture-owned question, feature and interaction reports.

Reads Dataset-owned questions and Capture's existing annotation-aware reader.
Exports derived views only; no weights, activations or measurements are changed.
"""

from __future__ import annotations

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
    return "；".join(descriptions) if descriptions else annotation.get("reason", "未取得公开解释")


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
    is_single = cfg.task.get("mode") == "g_s"
    is_exhaustive = cfg.task.get("mode") == "I" or (
        cfg.task.get("mode") == "sar" and cfg.task.get("sar", {}).get("algorithm") == "exhaustive"
    )
    questions = []
    for sample in selected:
        recorded = observations.get(sample["sample_id"], {})
        result = recorded.get("result", {})
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
        if cfg.task.get("mode") == "whiten"
        or (
            cfg.task.get("mode") == "sar"
            and cfg.task.get("sar", {}).get("feature_selection") == "whiten"
        )
        else "absolute_native_activation",
        "pair_ranking": "signed_exhaustive_interaction_descending"
        if is_exhaustive
        else "signed_pairwise_coefficient_descending"
        if cfg.task.get("sar", {}).get("pair_ranking", "absolute") == "signed"
        else "absolute_pairwise_coefficient",
        "explanation_sources": {
            "features": "Neuronpedia 公开解释，保留原文和来源字段",
            "pairs": "固定背景实测四格交互"
            if is_exhaustive
            else "组合干预二阶拟合的交互方向说明；不冒充 Neuronpedia 的特征对解释",
        },
        "questions": questions,
    }
    output = root / "reports"
    output.mkdir(exist_ok=True)
    write_json(output / "question_features_pairs.json", payload)
    from tasks.capture.selection import settings

    selected_count = settings(cfg)["capture_count"]
    feature_title = (
        "删除后g(S)"
        if is_single
        else "Whiten"
        if payload["feature_ranking"] == "absolute_zscore"
        else "原生激活"
    ) + f" Top {selected_count} 特征"
    if is_exhaustive:
        feature_title = "最高交互pair的端点特征"
    pair_title = f"{'I' if is_exhaustive else 'SAR'} Top {selected_count if is_exhaustive else cfg.task.get('sar', {}).get('top_pairs', 0)} 配对"
    pair_order = (
        "实测I原值降序"
        if is_exhaustive
        else "系数值降序"
        if payload["pair_ranking"] == "signed_pairwise_coefficient_descending"
        else "系数绝对值降序"
    )
    sections = []
    markdown = [
        "# 题目、Whiten 特征与 SAR 配对",
        "",
        f"模型：{payload['model']}；SAE：{payload['sae']}；层：{payload['layer']}。",
        f"已记录 {payload['recorded']}/{payload['requested']} 题。状态：{payload['statuses']}。",
        "",
        f"特征解释来自 Neuronpedia，保留其公开原文。配对排序：{pair_order}。"
        + ("I=g11−g10−g01+g00，每对遍历固定背景后取最大值。" if is_exhaustive else "")
        + ("g(S)为仅删除该特征后的答案平均log概率；Δg相对原始全保留。" if is_single else ""),
        "",
    ]

    def number(value):
        return "未定义" if value is None else f"{value:.6g}"

    def cell(value):
        return html.escape(str(value))

    for index, q in enumerate(questions, 1):
        feature_rows = []
        pair_rows = []
        markdown.extend(
            [
                f"## {index}. {q['question']}",
                "",
                f"样本：{q['sample_id']}；状态：{q['status']}。",
                f"标准答案：{q['answer']}",
                "",
                "| 特征 ID | 原始激活 | Whiten 分数 | g(S) | Δg | Neuronpedia 解释 |"
                if is_single
                else "| 特征 ID | 原始激活 | Whiten 分数 | Neuronpedia 解释 |",
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
                "| 特征对 | 实测I | 两端特征的公开解释 |"
                if is_exhaustive
                else "| 特征对 | 交互系数 | 两端特征的公开解释 |",
                "|---|---:|---|",
            ]
        )
        for pair in q["pairs"]:
            explanations = "；".join(
                f"{f['feature_id']}：{explanation_text(f)}" for f in pair.get("features", [])
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
        metrics = "；".join(
            f"{k}={number(q['sar_metrics'].get(k))}" for k in ("sar1", "sar2", "rho2", "var")
        )
        if is_single or is_exhaustive:
            metrics = f"实测组合数：{q['measured_states']}"
        error = f"<p>{cell(q['error'])}</p>" if q["error"] else ""
        sections.append(
            f"<details><summary>{index}. {cell(q['question'])} · {cell(q['status'])}</summary>"
            f"<p>样本：{cell(q['sample_id'])}；标准答案：{cell(q['answer'])}</p>{error}"
            f"<h3>{cell(feature_title)}</h3><table><tr><th>ID</th><th>原始激活</th><th>Whiten 分数</th>{'<th>g(S)</th><th>Δg</th>' if is_single else ''}<th>Neuronpedia 公开解释</th></tr>{''.join(feature_rows)}</table>"
            f"<h3>{cell(pair_title)}</h3><table><tr><th>特征对</th><th>{'实测I' if is_exhaustive else '交互系数'}</th><th>两端公开解释</th></tr>{''.join(pair_rows)}</table>"
            f"<p>{cell(q['pair_fit'].get('reason', ''))}</p>"
            f"<p>{cell(metrics)}"
            + (
                ""
                if is_single or is_exhaustive
                else f"；拟合秩：{cell(q['pair_fit'].get('fit_rank'))}/{cell(q['pair_fit'].get('fit_parameters'))}"
            )
            + "</p>"
            f"<details><summary>完整模型输入</summary><pre>{cell(q['prompt'])}</pre></details></details>"
        )
        markdown.extend(["", metrics, ""])
    page = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Whiten 特征与 SAR 配对</title>"
        "<style>body{font:16px system-ui;margin:2rem;line-height:1.6;max-width:1400px}details{margin:1rem 0;padding:.7rem;border:1px solid #ccc}summary{cursor:pointer}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem;text-align:left}pre{white-space:pre-wrap}</style>"
        f"<h1>{cell(payload['dataset'])}：特征与 SAR 配对</h1><p>模型：{cell(payload['model'])}；SAE：{cell(payload['sae'])}；层：{cell(payload['layer'])}。已记录 {len(observations)}/{len(questions)} 题。</p>"
        f"<p>特征解释来自 Neuronpedia，保留原文。配对排序：{pair_order}；g(S)为单特征删除后的答案平均log概率。</p>"
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
