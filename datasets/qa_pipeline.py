"""Run verified SAE captures on the frozen 500 QA cohort.

This is an *explicit* real-checkpoint operation. It does not synthesize rows or
silently substitute unavailable weights/datasets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from saebase.datasets.qa_standard import (
    DATASETS,
    PROTOCOL_ID,
    digest,
    prepare_dataset,
    verify_dataset,
)
from saebase.datasets.qa_neuronpedia import load_mapping, load_annotations, annotate
from saebase.datasets.qa_inference import make_input


# Each pair is from the existing verified-site matrix, not inferred by layer fraction.
SITES: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("gemma270m", "gemma270m_scope_resid", (5, 12)),
    ("pythia410m", "pythia410m_mlp_topk", (12, 20)),
    ("gemma1b", "gemma1b_scope_resid", (7, 17)),
    ("qwen3_1_7b_base", "qwen3_1_7b_base_resid_w32k", (14, 23)),
    ("gemma2_2b_base", "gemma2_2b_pt_l25_canonical", (13, 25)),
    ("qwen35_2b", "qwen35_2b_resid_w32k", (6, 20)),
    ("gemma3_4b_pt", "gemma3_4b_pt_scope_resid", (9, 29)),
    ("gemma3_4b_it", "gemma3_4b_it_scope_resid", (9, 29)),
    ("qwen3_8b", "qwen3_8b_resid_w64k", (9, 31)),
    ("llama31_8b", "llama31_8b_lxr_8x", (8, 27)),
)


def site_list() -> list[dict[str, Any]]:
    from saebase.sae.registry import get_profile

    result = []
    for model_key, sae_key, layers in SITES:
        p = get_profile(sae_key)
        if p.model_key != model_key:
            raise ValueError(f"Model/SAE registry mismatch {model_key} {sae_key}")
        for layer in layers:
            if layer not in p.layers:
                raise ValueError(f"Unsupported checkpoint layer {sae_key} {layer}")
            result.append(
                {
                    "model_key": model_key,
                    "sae_key": sae_key,
                    "layer": layer,
                    "hook_name": p.hook_name(layer),
                    "sae_id": p.sae_id(layer),
                    "site_id": f"{model_key}:{sae_key}:L{layer}",
                }
            )
    assert len(result) == 20
    return result


def _append_once(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _read_verified(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = row["sample_id"]
        if key in data:
            raise RuntimeError(f"Duplicate committed result row {key}")
        if row["status"] != "executed" or row["protocol_id"] != PROTOCOL_ID:
            raise RuntimeError("Prior results contain non-executed or foreign records")
        data[key] = row
    return data


def _tokenize(
    pair: Any, record: dict[str, Any], model_key: str, limit: int
) -> tuple[Any, dict[str, Any]]:
    tokens, metadata = make_input(pair, record, model_key)
    length = int(tokens.shape[1])
    if length > limit:
        raise ValueError(f"token_length_exceeds_limit:{length}>{limit}")
    if length < 2:
        raise ValueError(f"token_length_too_short:{length}")
    return tokens, metadata


def _capture_one(
    pair: Any,
    tokens: Any,
    *,
    top_k: int,
    mapping: dict[str, Any],
    annotations: dict[str, Any],
    sae_sha256: str | None,
    site_id: str,
) -> dict[str, Any]:
    import torch
    from saebase.api import capture_features

    site = pair.site
    with torch.inference_mode():
        snap = capture_features(pair.model, pair.sae, tokens=tokens, site=site)
        codes = snap.codes.detach().float().cpu()
        active = torch.nonzero(codes != 0, as_tuple=False).flatten()
        top = sorted(
            (int(i) for i in active.tolist()),
            key=lambda i: (-abs(float(codes[i].item())), i),
        )[:top_k]
        h = snap.activation.detach().float().cpu()
        recon = snap.reconstruction.detach().float().cpu()
        mse = float(torch.mean((h - recon) ** 2).item())
        norm_h = float(torch.linalg.vector_norm(h).item())
        logit = pair.model.forward_logits(tokens=tokens)
        if not bool(torch.isfinite(logit).all()):
            raise RuntimeError("Model output contains non-finite logits")
    if not math.isfinite(mse) or len(codes) != pair.identity.d_sae:
        raise RuntimeError("Invalid SAE feature dimensions or reconstruction")
    if any(not math.isfinite(float(codes[i])) for i in active):
        raise RuntimeError("Non-finite active SAE feature")
    return {
        "token_count": int(tokens.shape[1]),
        "sae_d_in": pair.identity.d_in,
        "sae_d_sae": pair.identity.d_sae,
        "active_count": int(active.numel()),
        "l0": int(active.numel()),
        "reconstruction_mse": mse,
        "activation_norm_l2": norm_h,
        "baseline_argmax_id": int(torch.argmax(logit).item()),
        "baseline_logits_finite": True,
        "capture_position": -1,
        "top_features": [
            {
                **annotate(
                    i,
                    site_id=site_id,
                    sae_sha256=sae_sha256,
                    mapping=mapping,
                    annotations=annotations,
                ),
                "activation": float(codes[i].item()),
            }
            for i in top
        ],
    }


def run_capture(
    *,
    data_root: Path,
    output_root: Path,
    model_key: str,
    layer: int,
    dataset_id: str,
    device: str = "cuda",
    max_tokens: int = 2048,
    limit: int | None = None,
    top_k: int = 10,
    mapping_file: Path | None = None,
    annotation_file: Path | None = None,
    loaded_pair: Any | None = None,
) -> dict[str, Any]:
    from saebase.api import load_registered

    available = [s for s in site_list() if s["model_key"] == model_key and s["layer"] == layer]
    if len(available) != 1:
        raise ValueError("Select a site from the frozen 20-site matrix")
    target = available[0]
    manifest = verify_dataset(dataset_id, data_root)
    p = data_root / dataset_id / "questions.jsonl"
    records = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s]
    if limit is not None and not 1 <= limit <= len(records):
        raise ValueError("Pilot limit outside available question count")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be in [1, 100]")
    subset = records[:limit] if limit else records
    base = output_root / PROTOCOL_ID / model_key / target["sae_key"] / f"L{layer}" / dataset_id
    outpath = base / ("pilot.jsonl" if limit else "capture.jsonl")
    cfg = {
        "protocol_id": PROTOCOL_ID,
        "target": target,
        "dataset_manifest_sha256": digest(manifest),
        "device": device,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "sample_limit": limit,
    }
    base.mkdir(parents=True, exist_ok=True)
    config_path = base / ("pilot_config.json" if limit else "capture_config.json")
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("Existing output directory belongs to a different configuration")
    else:
        config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = _read_verified(outpath)
    expected_ids = {r["sample_id"] for r in subset}
    if not set(completed).issubset(expected_ids):
        raise RuntimeError("Existing results include samples outside this cohort")
    pair = loaded_pair or load_registered(
        model_key=model_key, sae_key=target["sae_key"], layer=layer, device=device
    )
    if (
        pair.identity.hook_name != target["hook_name"]
        or pair.identity.model_key != model_key
        or pair.identity.sae_key != target["sae_key"]
        or pair.identity.layer != layer
    ):
        raise RuntimeError("Loaded model/SAE/hook identity differs from frozen target")
    from saebase.store import sha256_state_dict

    sae_sha256 = sha256_state_dict(pair.sae)
    mappings = load_mapping(mapping_file)
    annotations = load_annotations(annotation_file)
    for item in subset:
        if item["sample_id"] in completed:
            old = completed[item["sample_id"]]
            if (
                old.get("prompt_sha256") != item["prompt_sha256"]
                or old.get("sae_sha256") != sae_sha256
            ):
                raise RuntimeError("Existing capture row uses a different prompt or SAE weight")
            continue
        try:
            tokens, input_meta = _tokenize(pair, item, model_key, max_tokens)
            measured = _capture_one(
                pair,
                tokens,
                top_k=top_k,
                mapping=mappings,
                annotations=annotations,
                sae_sha256=sae_sha256,
                site_id=target["site_id"],
            )
        except ValueError as exc:
            # Invalid prompts are explicit failures; never count as successful data.
            failure = base / ("pilot_failures.jsonl" if limit else "capture_failures.jsonl")
            _append_once(
                failure,
                {
                    "sample_id": item["sample_id"],
                    "status": "failed",
                    "reason": str(exc),
                    "protocol_id": PROTOCOL_ID,
                },
            )
            continue
        row = {
            "status": "executed",
            "protocol_id": PROTOCOL_ID,
            "sample_id": item["sample_id"],
            "dataset_id": dataset_id,
            "site_id": target["site_id"],
            "model_key": model_key,
            "sae_key": target["sae_key"],
            "layer": layer,
            "hook_name": target["hook_name"],
            "sae_sha256": sae_sha256,
            "source_revision": manifest["source_revision"],
            **input_meta,
            **measured,
        }
        _append_once(outpath, row)
        completed[item["sample_id"]] = row
    summary = {
        "protocol_id": PROTOCOL_ID,
        "site_id": target["site_id"],
        "dataset_id": dataset_id,
        "expected": len(subset),
        "executed": len(completed),
        "missing": len(subset) - len(completed),
        "status": "complete" if len(completed) == len(subset) else "partial",
        "sae_sha256": sae_sha256,
        "result_path": str(outpath),
    }
    (base / ("pilot_summary.json" if limit else "capture_summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SAEBase frozen QA500 capture pipeline")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("sites", help="List 20 registered and validated-by-metadata sites")
    prep = sub.add_parser("prepare", help="Freeze upstream HF samples with exact source commit")
    prep.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    prep.add_argument("--data-root", type=Path, default=Path("experiment/qa500/inputs"))
    check = sub.add_parser("verify", help="Check prepared cohort checksums and prompt identities")
    check.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    check.add_argument("--data-root", type=Path, default=Path("experiment/qa500/inputs"))
    cap = sub.add_parser(
        "capture", help="Run real model + SAE weights with explicit GPU authorization"
    )
    cap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    cap.add_argument("--model", required=True, choices=[s[0] for s in SITES])
    cap.add_argument("--layer", required=True, type=int)
    cap.add_argument("--data-root", type=Path, default=Path("experiment/qa500/inputs"))
    cap.add_argument("--output-root", type=Path, default=Path("experiment/qa500/results"))
    cap.add_argument("--device", default="cuda")
    cap.add_argument("--max-tokens", type=int, default=2048)
    cap.add_argument("--limit", type=int)
    cap.add_argument("--top-k", type=int, default=10)
    cap.add_argument("--neuronpedia-map", type=Path)
    cap.add_argument("--neuronpedia-records", type=Path)
    args = parser.parse_args(argv)
    if args.action == "sites":
        print(json.dumps(site_list(), indent=2))
    elif args.action == "prepare":
        keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
        for key in keys:
            print(json.dumps(prepare_dataset(key, args.data_root)))
    elif args.action == "verify":
        keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
        for key in keys:
            print(json.dumps(verify_dataset(key, args.data_root)))
    else:
        print(
            json.dumps(
                run_capture(
                    data_root=args.data_root,
                    output_root=args.output_root,
                    model_key=args.model,
                    layer=args.layer,
                    dataset_id=args.dataset,
                    device=args.device,
                    max_tokens=args.max_tokens,
                    limit=args.limit,
                    top_k=args.top_k,
                    mapping_file=args.neuronpedia_map,
                    annotation_file=args.neuronpedia_records,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
