"""Capture RSS and variant-specific features for the 24 public demo prompts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.RSS.inputs import FACTS, PROMPTS
from scripts.RSS.rss import rss
from saebase.api import LoadedResearchPair, ModelAdapter, SAEIdentity, capture_features
from saebase.api.sae_loading import load_sae_for_profile
from omegaconf import OmegaConf
from transformer_lens import HookedTransformer
from saebase.api.neuronpedia import annotate_features
from tasks.capture.operation import neuronpedia_settings

PARAMETERS = {"M": 400, "K": 20, "rho": 0.8, "kappa": 0.8, "tau": 1e-5}
UNIQUE = {"candidate_pool": 100, "dominance_ratio": 5.0, "per_variant": 3}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def select_unique(delta, support):
    groups = []
    for v in range(len(delta)):
        strength = np.abs(delta[v])
        other = np.max(np.abs(np.delete(delta, v, axis=0)), axis=0)
        candidates = np.argsort(-strength, kind="stable")[: UNIQUE["candidate_pool"]]
        groups.append(
            [
                int(f)
                for f in candidates
                if f not in support
                and strength[f] > PARAMETERS["tau"]
                and strength[f] >= UNIQUE["dominance_ratio"] * other[f]
            ][: UNIQUE["per_variant"]]
        )
    return groups


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def explanation(row):
    annotation = row.get("annotation", {})
    descriptions = annotation.get("explanations", [])
    return cell(
        "; ".join(x["description"] for x in descriptions) or annotation.get("reason", "暂无解释")
    )


def feature_link(row):
    url = row.get("annotation", {}).get(
        "url", f"https://www.neuronpedia.org/gpt2-small/8-res-jb/{row['feature_id']}"
    )
    return f"[F{row['feature_id']}]({url})"


def render(case):
    lines = [
        f"# 案例 {case['case_id']}",
        "",
        f"目标：{case['clean_prompt']}",
        "",
        "GPT-2 Small，layer 8，blocks.8.hook_resid_pre，完整原始 SAE codes，最后一个目标 token。",
        "",
        "变化值 = 变体激活 − clean 激活。解释来自 Neuronpedia，保留原文，语义对应关系需要人工阅读判断。",
        "",
    ]
    lines.extend(f"- V{i + 1}：{fact}" for i, fact in enumerate(FACTS))
    lines += [
        "",
        "## 表 A：RSS 共同响应特征",
        "",
        "| 特征 | 方向 | Clean | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in case["rss_features"]:
        values = " | ".join(f"{x:+.4f}" for x in row["delta"])
        lines.append(
            f"| {feature_link(row)} | {'增加' if row['direction'] > 0 else '减少'} | {row['clean_activation']:.4f} | {values} | {explanation(row)} |"
        )
    if not case["rss_features"]:
        lines += ["", "当前参数下没有 RSS 特征。"]
    lines += [
        "",
        "## 表 B：变体特有特征",
        "",
        "从各变体绝对变化前 100 名中，取变化 > 1e-5 且 ≥ 其他四个变体最大变化五倍的特征；排除 RSS 选集，每个变体最多三个。",
        "",
        "| 变体 | 特征 | Clean | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in case["variant_specific"]:
        for row in group["features"]:
            values = " | ".join(f"{x:+.4f}" for x in row["delta"])
            lines.append(
                f"| V{group['variant']} | {feature_link(row)} | {row['clean_activation']:.4f} | {values} | {explanation(row)} |"
            )
        if not group["features"]:
            lines.append(f"| V{group['variant']} | 无符合条件的特征 | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiment/RSS" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source = json.loads((Path(__file__).parent / "source.json").read_text())
    write_json(
        out / "run.json",
        {
            "status": "loading",
            "source": source,
            "rss": PARAMETERS,
            "unique": UNIQUE,
            "command": sys.argv,
        },
    )
    print(f"输出目录：{out}", flush=True)
    # Match the published RSS demo and this SAE's model_from_pretrained_kwargs.
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", center_writing_weights=True, device=args.device
    )
    sae, metadata = load_sae_for_profile("gpt2small_res_jb_l8", 8, device=args.device)
    pair = LoadedResearchPair(
        model=ModelAdapter(model),
        sae=sae,
        identity=SAEIdentity.from_loaded(sae=sae, metadata=metadata, model_key="gpt2small"),
    )
    pair.model.model.eval()
    identity = pair.identity.to_dict()
    identity["model_loading"] = {
        "method": "from_pretrained_no_processing",
        "center_writing_weights": True,
        "dtype": str(next(model.parameters()).dtype),
    }
    cases = []
    all_ids = set()
    with torch.inference_mode():
        for i, (opening, verb, target) in enumerate(PROMPTS[: args.limit], 1):
            clean_prompt = f"{opening} {verb} {target}"
            variants = [f"{fact} {clean_prompt}" for fact in FACTS]
            snapshots = [
                capture_features(pair.model, pair.sae, site=pair.site, prompt=p)
                for p in [clean_prompt, *variants]
            ]
            codes = np.stack([s.codes.detach().float().cpu().numpy() for s in snapshots])
            delta = codes[1:] - codes[0]
            support, directions, mean_rss = rss(delta, **PARAMETERS)
            unique = select_unique(delta, set(support.tolist()))
            directory = out / f"case_{i:02d}"
            directory.mkdir()
            np.savez_compressed(
                directory / "rss_activations.npz", clean=codes[0], variants=codes[1:], delta=delta
            )

            def row(feature_id):
                f = int(feature_id)
                moved = np.abs(delta[:, f]) > PARAMETERS["tau"]
                up = int((delta[:, f] > PARAMETERS["tau"]).sum())
                down = int((delta[:, f] < -PARAMETERS["tau"]).sum())
                return {
                    "feature_id": f,
                    "clean_activation": float(codes[0, f]),
                    "variant_activations": codes[1:, f].tolist(),
                    "delta": delta[:, f].tolist(),
                    "mean_absolute_change": float(np.abs(delta[:, f]).mean()),
                    "recurrence": float(moved.mean()),
                    "consistency": max(up, down) / max(up + down, 1),
                }

            case = {
                "case_id": i,
                "clean_prompt": clean_prompt,
                "target": target,
                "contexts": FACTS,
                "variants": variants,
                "token_ids": [s.tokens.detach().cpu().reshape(-1).tolist() for s in snapshots],
                "capture_position": -1,
                "identity": identity,
                "source": source,
                "rss_parameters": PARAMETERS,
                "unique_parameters": UNIQUE,
                "mean_rss": mean_rss if np.isfinite(mean_rss) else None,
                "rss_features": [
                    dict(row(f), direction=int(d)) for f, d in zip(support, directions)
                ],
                "variant_specific": [
                    {"variant": v + 1, "context": FACTS[v], "features": [row(f) for f in ids]}
                    for v, ids in enumerate(unique)
                ],
            }
            all_ids.update(support.tolist())
            all_ids.update(f for group in unique for f in group)
            write_json(directory / "rss_case.json", case)
            (directory / "rss_case.md").write_text(render(case))
            cases.append(case)
            print(
                f"{i:02d} RSS={len(support)}, unique={[len(g) for g in unique]}: {clean_prompt}",
                flush=True,
            )
    print(f"获取 {len(all_ids)} 个特征的 Neuronpedia 解释", flush=True)
    info, annotations = annotate_features(
        pair, sorted(all_ids), neuronpedia_settings(OmegaConf.create({"task": {}}))
    )
    write_json(out / "neuronpedia.json", {"info": info, "features": annotations})
    index = [
        "# RSS 案例总览",
        "",
        f"本轮 {len(cases)} 个候选均保留，以下按公开 demo 原顺序排列。",
        "",
        "| 案例 | 目标 | RSS 特征 | V1–V5 特有特征数 |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        directory = out / f"case_{case['case_id']:02d}"
        for row in case["rss_features"] + [
            f for g in case["variant_specific"] for f in g["features"]
        ]:
            row["annotation"] = annotations[row["feature_id"]]
        case["neuronpedia"] = info
        write_json(directory / "rss_case.json", case)
        (directory / "rss_case.md").write_text(render(case))
        sizes = [len(g["features"]) for g in case["variant_specific"]]
        index.append(
            f"| [{case['case_id']:02d}](case_{case['case_id']:02d}/rss_case.md) | {case['clean_prompt']} | {len(case['rss_features'])} | {sizes} |"
        )
    (out / "index.md").write_text("\n".join(index) + "\n")
    write_json(
        out / "run.json",
        {
            "status": "complete",
            "case_count": len(cases),
            "identity": identity,
            "device": args.device,
            "source": source,
            "rss": PARAMETERS,
            "unique": UNIQUE,
            "neuronpedia": info,
            "command": sys.argv,
        },
    )
    print(f"完成：{out / 'index.md'}；解释状态：{info['status']}", flush=True)


if __name__ == "__main__":
    main()
