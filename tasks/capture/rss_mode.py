"""Capture one clean target and its context variants with the original RSS selector."""

from saebase.runtime.configuration.task_options import task_value

from tasks.capture.configuration import capture_section
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
    from saebase.runtime.sae_operations import input_path

    raw = capture_section(cfg.task, "rss")
    source = raw.get("source")
    contexts = list(raw.get("contexts", []))
    if (
        not source
        or len(contexts) < 2
        or not all(isinstance(x, str) and x.strip() for x in contexts)
    ):
        raise ValueError(
            "mode=rss requires task.capture.rss.source and at least two nonempty task.capture.rss.contexts"
        )
    if not str(cfg.task.get("prompt", "") or "").strip():
        raise ValueError("mode=rss requires an explicit task.prompt for the shared target")
    if (
        capture_section(cfg.task, "selection").get("candidate_count", "all") != "all"
        or task_value(cfg.task, "output_scope", "last_token") != "last_token"
    ):
        raise ValueError(
            "RSS captures complete last-token codes; set task.capture.selection.candidate_count=all and output_scope=last_token"
        )
    return {
        "source": input_path(source),
        "contexts": contexts,
        "parameters": {k: raw.get(k, v) for k, v in DEFAULTS.items()},
        "dominance_ratio": float(raw.get("report", {}).get("dominance_ratio", 5.0)),
    }


def measure(
    pair,
    prompt,
    contexts,
    source,
    parameters=None,
    dominance_ratio=5.0,
    variants=None,
    clean_snapshot=None,
):
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
        base = (
            clean_snapshot
            if clean_snapshot is not None
            else capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
        )
        snapshots = [base] + [
            capture_features(pair.model, pair.sae, site=pair.site, prompt=p) for p in variants
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
    lines = ["# RSS Contrast cases", "", f"Clean:{result['clean_prompt']}", ""]
    lines += [f"- V{i}: {p}" for i, p in enumerate(result["variants"], 1)]
    lines += ["", "Model Original Answer:", "", result.get("model_answer", {}).get("text", "Not Generated")]
    for i, answer in enumerate(result.get("variant_model_answers", []), 1):
        lines += [
            "",
            f"V{i} Model Original Answer({answer['status']}):",
            "",
            answer.get("text", str(answer.get("error", ""))),
        ]
    lines += [
        "",
        "## Table A:Original RSS All Selected Features",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [feature_row(r) for r in result["rss_features"]]
    lines += [
        "",
        "## Table B Candidate: Activation focused on one variant and recurrence < 0.8",
        "",
        "All unselected features are sorted by the absolute change of the variant, requiring this variant to be added > tau The candidate needs to judge the semantics based on the interpretation..",
        "",
    ]
    for group in result["context_candidates"]:
        lines += [
            f"### V{group['variant']}:{group['context']}",
            "",
            "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [feature_row(r) for r in group["features"]]
        if not group["features"]:
            lines += ["", "No candidate with numerical criteria."]
        lines += [""]
    return "\n".join(lines) + "\n"


def feature_row(row):
    annotation = row.get("annotation", {})
    descriptions = (
        "; ".join(e["description"] for e in annotation.get("explanations", [])) or "There’s no explanation."
    )
    descriptions = descriptions.replace("|", "\\|").replace("\n", " ")
    values = " | ".join(f"{x:+.6g}" for x in row["delta"])
    url = annotation.get("url", "")
    return (
        f"| [F{row['feature_id']}]({url}) | {values} | {row['recurrence']:.1f} | {descriptions} |"
    )


def execute(cfg):
    from saebase.runtime.sae_operations import _pair, _prompt
    from tasks.capture.operation import neuronpedia_settings

    options = settings(cfg)
    pair = _pair(cfg)
    result, codes, delta = measure(pair, _prompt(cfg), **options)
    from saebase.runtime.sae_operations import pair_state_fingerprints, _fingerprint_prompt

    model_sha, _ = pair_state_fingerprints(pair)
    result["input_identity"] = {"model_sha256": model_sha}
    result["prompt_sha256"] = _fingerprint_prompt(result["clean_prompt"])
    from tasks.capture.model_answer import answers, attach

    continuations = answers(cfg, pair, [result["clean_prompt"], *result["variants"]])
    result["variant_model_answers"] = continuations[1:]
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
    attach(result, continuations[0], extra_answers=continuations[1:])
    from tasks.capture.model_answer import summary as answer_summary

    result["stages"]["variant_model_answers"] = answer_summary(continuations[1:])
    write_json(Path("rss_case.json"), result)
    Path("rss_case.md").write_text(markdown(result))
    result["required_artifacts"] = [
        {"path": p} for p in ("rss_case.json", "rss_case.md", "rss_activations.npz")
    ]
    return result
