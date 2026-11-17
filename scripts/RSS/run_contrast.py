"""Run the five distinct contexts through capture mode=rss, banana bread first."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from transformer_lens import HookedTransformer

from conf import load_group
from saebase.api import LoadedResearchPair, ModelAdapter, SAEIdentity, load_registered
from saebase.api.sae_loading import load_sae_for_profile
from saebase.api.neuronpedia import annotation_session
from scripts.RSS.inputs import PROMPTS
from tasks.capture.task import CaptureTask
from tasks.sae_runtime import loaded_pair_session

CONTEXTS = [
    "A basketball player scored a three-point shot.",
    "An astronaut floated outside a spaceship.",
    "A violinist performed a classical symphony.",
    "A chess player moved a knight across the board.",
    "An elephant walked across the African savanna.",
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiment/RSS"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_contrast"),
    )
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--sae-key", default="gpt2small_res_jb_l8")
    parser.add_argument("--model-key", default="gpt2small")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--contexts-json", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_key == "gpt2small":
        model = HookedTransformer.from_pretrained_no_processing(
            "gpt2", center_writing_weights=True, device=device
        )
        sae, metadata = load_sae_for_profile(args.sae_key, args.layer, device=device)
        pair = LoadedResearchPair(
            model=ModelAdapter(model),
            sae=sae,
            identity=SAEIdentity.from_loaded(sae=sae, metadata=metadata, model_key=args.model_key),
        )
        loading = {"method": "from_pretrained_no_processing", "center_writing_weights": True}
    else:
        pair = load_registered(
            model_key=args.model_key,
            sae_key=args.sae_key,
            layer=args.layer,
            device=device,
            require_patch=False,
        )
        loading = {
            "method": "load_registered",
            "dtype": str(next(pair.model.model.parameters()).dtype),
        }
    cfg = OmegaConf.merge(
        load_group("model", args.model_key),
        load_group("dataset", "ioi"),
        OmegaConf.load(ROOT / "conf/task/capture.yaml"),
    )
    cfg.task.update(
        {
            "mode": "rss",
            "execute": True,
            "filter": "all",
            "device": device,
            "sae_key": args.sae_key,
            "layer": args.layer,
        }
    )
    cfg.task.rss.source = str(ROOT / "scripts/RSS/rss.py")
    contexts = json.loads(args.contexts_json.read_text()) if args.contexts_json else CONTEXTS
    cfg.task.rss.contexts = contexts
    task = CaptureTask()
    cases = []
    prompts = (
        json.loads(args.prompts_json.read_text())
        if args.prompts_json
        else [" ".join(p) for p in PROMPTS]
    )
    order = (
        list(range(len(prompts))) if args.prompts_json else [3] + [i for i in range(24) if i != 3]
    )
    origin = Path.cwd()
    with loaded_pair_session(pair), annotation_session():
        for i in order[: args.limit]:
            cfg.task.prompt = prompts[i]
            directory = out / f"case_{i + 1:02d}"
            directory.mkdir()
            OmegaConf.save(cfg, directory / "config.yaml")
            try:
                os.chdir(directory)
                result = task.run(cfg)
                (directory / "task_result.json").write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
                )
            finally:
                os.chdir(origin)
            case = json.loads((directory / "rss_case.json").read_text())
            sizes = [len(g["features"]) for g in case["context_candidates"]]
            cases.append(
                {
                    "case_id": i + 1,
                    "prompt": cfg.task.prompt,
                    "rss_count": len(case["rss_features"]),
                    "candidate_counts": sizes,
                }
            )
            print(
                f"{i + 1:02d} RSS={len(case['rss_features'])}, candidates={sizes}: {cfg.task.prompt}",
                flush=True,
            )
    (out / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cases": cases,
                "contexts": contexts,
                "prompt_source": str(args.prompts_json.resolve())
                if args.prompts_json
                else "RSS demo 24 prompts",
                "identity": pair.identity.to_dict(),
                "model_loading": loading,
                "candidate_rule": "not RSS support; recurrence < 0.8; delta_v > 1e-5; delta_v >= 5 * max(abs(delta_others)); all features searched; sorted by variant amplitude",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    index = [
        "# 五个鲜明上下文：全部候选案例",
        "",
        "本目录保留本轮全部案例。表 B 是纯数值候选，需根据原始解释判断语义。完整展示结果以另存的案例报告为准。",
        "",
        "| 案例 | Clean | RSS 数 | V1–V5 数值候选数 |",
        "| --- | --- | --- | --- |",
    ]
    for c in cases:
        index.append(
            f"| [{c['case_id']:02d}](case_{c['case_id']:02d}/rss_case.md) | {c['prompt']} | {c['rss_count']} | {c['candidate_counts']} |"
        )
    (out / "index.md").write_text("\n".join(index) + "\n")
    print(f"完成：{out / 'index.md'}", flush=True)


if __name__ == "__main__":
    main()
