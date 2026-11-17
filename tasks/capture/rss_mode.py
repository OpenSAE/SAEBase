"""Capture one clean target and its context variants with the original RSS selector."""

from pathlib import Path
import hashlib
import runpy

import numpy as np
import torch

from saebase.api import capture_features
from saebase.api.neuronpedia import annotate_features
from saebase.runtime.experiment_results import write_json

DEFAULTS = {"M": 400, "K": 20, "rho": 0.8, "kappa": 0.8, "tau": 1e-5}


def settings(cfg):
    from tasks.sae_runtime import input_path

    raw = cfg.task.get("rss", {})
    source = raw.get("source")
    contexts = list(raw.get("contexts", []))
    if (
        not source
        or len(contexts) < 2
        or not all(isinstance(x, str) and x.strip() for x in contexts)
    ):
        raise ValueError(
            "mode=rss requires task.rss.source and at least two nonempty task.rss.contexts"
        )
    if not str(cfg.task.get("prompt", "") or "").strip():
        raise ValueError("mode=rss requires an explicit task.prompt for the shared target")
    if (
        cfg.task.get("filter", "all") != "all"
        or cfg.task.get("output_scope", "last_token") != "last_token"
    ):
        raise ValueError(
            "RSS captures complete last-token codes; set task.filter=all and output_scope=last_token"
        )
    return {
        "source": input_path(source),
        "contexts": contexts,
        "parameters": {k: raw.get(k, v) for k, v in DEFAULTS.items()},
        "dominance_ratio": float(raw.get("dominance_ratio", 5.0)),
    }


def measure(pair, prompt, contexts, source, parameters=None, dominance_ratio=5.0, variants=None):
    """Select purely from activations before any explanation lookup."""
    parameters = dict(DEFAULTS if parameters is None else parameters)
    source = Path(source)
    selector = runpy.run_path(str(source))["rss"]
    variants = (
        [f"{context} {prompt}" for context in contexts] if variants is None else list(variants)
    )
    raw_model = getattr(pair.model.model, "_model", pair.model.model)
    raw_model.eval()
    with torch.inference_mode():
        snapshots = [
            capture_features(pair.model, pair.sae, site=pair.site, prompt=p)
            for p in [prompt, *variants]
        ]
    codes = np.stack([s.codes.detach().float().cpu().numpy() for s in snapshots])
    delta = codes[1:] - codes[0]
    support, direction, mean_rss = selector(delta, **parameters)
    magnitude = np.abs(delta)
    moved = magnitude > parameters["tau"]
    recurrence = moved.mean(axis=0)
    up = (delta > parameters["tau"]).sum(axis=0)
    down = (delta < -parameters["tau"]).sum(axis=0)
    strength_order = np.argsort(-magnitude.mean(axis=0), kind="stable")
    pool = set(strength_order[: parameters["M"]].tolist())
    selected = set(support.tolist())

    def row(f):
        f = int(f)
        return {
            "feature_id": f,
            "clean_activation": float(codes[0, f]),
            "variant_activations": codes[1:, f].tolist(),
            "delta": delta[:, f].tolist(),
            "mean_absolute_change": float(magnitude[:, f].mean()),
            "recurrence": float(recurrence[f]),
            "moved_count": int(moved[:, f].sum()),
            "consistency": float(max(up[f], down[f]) / max(up[f] + down[f], 1)),
            "in_top_M": f in pool,
            "rss_selected": f in selected,
        }

    groups = []
    for v in range(len(contexts)):
        other = np.delete(magnitude, v, axis=0).max(axis=0)
        ids = [
            int(f)
            for f in np.argsort(-magnitude[v], kind="stable")
            if f not in selected
            and recurrence[f] < parameters["rho"]
            and delta[v, f] > parameters["tau"]
            and magnitude[v, f] >= dominance_ratio * other[f]
        ]
        features = []
        for rank, f in enumerate(ids, 1):
            features.append(
                {
                    **row(f),
                    "candidate_rank": rank,
                    "other_max_change": float(other[f]),
                    "dominance_ratio": float(magnitude[v, f] / other[f]) if other[f] > 0 else None,
                    "rss_exclusion_reason": "recurrence_below_rho",
                }
            )
        groups.append({"variant": v + 1, "context": contexts[v], "features": features})
    result = {
        "status": "executed",
        "operation": "capture",
        "mode": "rss",
        "clean_prompt": prompt,
        "contexts": list(contexts),
        "variants": variants,
        "capture_position": pair.site.position,
        "identity": pair.identity.to_dict(),
        "token_ids": [s.tokens.detach().cpu().reshape(-1).tolist() for s in snapshots],
        "rss_parameters": parameters,
        "dominance_ratio": dominance_ratio,
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "mean_rss": mean_rss if np.isfinite(mean_rss) else None,
        "rss_features": [dict(row(f), direction=int(d)) for f, d in zip(support, direction)],
        "context_candidates": groups,
    }
    return result, codes, delta


def annotate(pair, result, options):
    rows = result["rss_features"] + [
        r for group in result["context_candidates"] for r in group["features"]
    ]
    info, descriptions = annotate_features(pair, sorted({r["feature_id"] for r in rows}), options)
    for row in rows:
        row["annotation"] = descriptions[row["feature_id"]]
    result["neuronpedia"] = info
    return result


def markdown(result):
    lines = ["# RSS 对照案例", "", f"Clean：{result['clean_prompt']}", ""]
    lines += [f"- V{i}: {p}" for i, p in enumerate(result["variants"], 1)]
    lines += [
        "",
        "## 表 A：原始 RSS 选中的全部特征",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [feature_row(r) for r in result["rss_features"]]
    lines += [
        "",
        "## 表 B 候选：激活集中于一个变体且 recurrence < 0.8",
        "",
        "所有未选中特征中按该变体的绝对变化排序，要求本变体增加 > tau 且至少为其他变体最大变化的五倍。候选尚需根据解释判断语义；微小变化照实保留，不能仅凭比例当成强响应。",
        "",
    ]
    for group in result["context_candidates"]:
        lines += [
            f"### V{group['variant']}：{group['context']}",
            "",
            "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [feature_row(r) for r in group["features"]]
        if not group["features"]:
            lines += ["", "无符合数值条件的候选。"]
        lines += [""]
    return "\n".join(lines) + "\n"


def feature_row(row):
    annotation = row.get("annotation", {})
    descriptions = (
        "; ".join(e["description"] for e in annotation.get("explanations", [])) or "暂无解释"
    )
    descriptions = descriptions.replace("|", "\\|").replace("\n", " ")
    values = " | ".join(f"{x:+.6g}" for x in row["delta"])
    url = annotation.get("url", "")
    return (
        f"| [F{row['feature_id']}]({url}) | {values} | {row['recurrence']:.1f} | {descriptions} |"
    )


def execute(cfg):
    from tasks.sae_runtime import _pair, _prompt
    from tasks.capture.operation import neuronpedia_settings

    options = settings(cfg)
    pair = _pair(cfg)
    result, codes, delta = measure(pair, _prompt(cfg), **options)
    np.savez_compressed("rss_activations.npz", clean=codes[0], variants=codes[1:], delta=delta)
    write_json(Path("rss_case.json"), result)
    annotate(pair, result, neuronpedia_settings(cfg))
    write_json(Path("rss_case.json"), result)
    Path("rss_case.md").write_text(markdown(result))
    from saebase.results import completion_summary

    info = result["neuronpedia"]
    pending = info["requested"] - info["explained"]
    result["completion"] = completion_summary(
        [
            {"status": "completed"},
            {
                "status": "blocked",
                "reason": info.get(
                    "reason", f"{pending} RSS/candidate feature explanations unavailable"
                ),
            }
            if pending
            else {"status": "completed"},
        ],
        unit="capture and annotation",
    )
    result["required_artifacts"] = [
        {"path": p} for p in ("rss_case.json", "rss_case.md", "rss_activations.npz")
    ]
    return result
