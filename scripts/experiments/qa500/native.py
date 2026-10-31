"""Optional, deterministic native Circuit and SAE Train work.

These tasks are separate from the 20-site capture cross-product. They invoke
original upstream workers, never synthetic substitute results. Their execution
requires explicit native-work authorization.
"""

from __future__ import annotations

from scripts.experiments.qa500.context import task_dir, typed_file, study_dir

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.experiments.qa500.cohort import PROTOCOL_ID, render_prompt, verify_dataset

ROOT = study_dir(Path.cwd())
MODEL = "google/gemma-2-2b"
TRAIN_DATASET = "Salesforce/wikitext"
TRAIN_CONFIG = "wikitext-2-raw-v1"
TRAIN_MAX_RECORDS = 6400
TRAIN_SEED = 20261031


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_records(input_root: Path, dataset: str) -> list[dict[str, Any]]:
    verify_dataset(dataset, input_root)
    p = input_root / dataset / "questions.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def run_circuits(
    *,
    data_root: Path,
    result_root: Path,
    approved: bool,
) -> dict[str, Any]:
    """Fixed 10 short BoolQ prompts, original circuit-tracer Gemma shortcut."""
    if not approved:
        raise PermissionError("Original native circuit tracer requires explicit approval")
    from tasks.circuits.backend.circuits import run_native_circuit_tracer

    candidates = _load_records(data_root, "boolq")
    candidates = sorted(
        (row for row in candidates if len(render_prompt(row)) <= 1900),
        key=lambda row: hashlib.sha256(("circuit:" + row["sample_id"]).encode("utf-8")).hexdigest(),
    )[:10]
    if len(candidates) != 10:
        raise RuntimeError("Fewer than 10 short, frozen BoolQ inputs for circuit tracer")
    site_dir = task_dir(
        result_root,
        task="circuits",
        dataset="boolq",
        model="gemma2_2b_base",
        sae_key="native_gemma",
        initialize=True,
    )
    root = site_dir
    results: list[dict[str, Any]] = []
    for row in candidates:
        ident = row["sample_id"]
        out = root / "metrics" / ident
        receipt_path = typed_file(root, "metadata", f"{ident}.json")
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            if previous.get("status") == "native_raw_graph_generated_not_normalized":
                results.append(previous)
                continue
        try:
            result = run_native_circuit_tracer(
                prompt=render_prompt(row),
                transcoder_set="gemma",
                output=out,
                slug=f"qa-{ident[:16]}",
                approved=True,
                timeout_seconds=1800,
                offload="cpu",
                dtype="bfloat16",
                max_feature_nodes=64,
                batch_size=4,
            )
            receipt = {
                "protocol_id": PROTOCOL_ID,
                "sample_id": ident,
                "model_key": "gemma2_2b_base",
                "dataset_id": "boolq",
                "prompt_sha256": row["prompt_sha256"],
                "original_native_tracer": True,
                **result,
            }
        except Exception as exc:
            receipt = {
                "protocol_id": PROTOCOL_ID,
                "sample_id": ident,
                "model_key": "gemma2_2b_base",
                "dataset_id": "boolq",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:500],
            }
        _save(receipt_path, receipt)
        results.append(receipt)
    status = (
        "complete"
        if all(r["status"] == "native_raw_graph_generated_not_normalized" for r in results)
        else "partial"
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "task": "circuits",
        "status": status,
        "completed": sum(
            r["status"] == "native_raw_graph_generated_not_normalized" for r in results
        ),
        "expected": 10,
        "interpretation": "native transcoder graph, not SAE feature explanation",
    }
    _save(typed_file(root, "reports", "summary.json"), summary)
    return summary


def prepare_training(*, root: Path) -> dict[str, Any]:
    """Pin one non-QA training source and official base-model snapshot."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    dataset_rev = api.dataset_info(TRAIN_DATASET).sha
    model_rev = api.model_info(MODEL).sha
    if not dataset_rev or not model_rev:
        raise RuntimeError("Missing exact model or training corpus revision")
    raw = load_dataset(TRAIN_DATASET, name=TRAIN_CONFIG, split="train", revision=dataset_rev)
    strings = sorted(
        {str(row["text"]).strip() for row in raw if str(row.get("text", "")).strip()},
        key=lambda val: hashlib.sha256(f"{TRAIN_SEED}:{val}".encode("utf-8")).hexdigest(),
    )
    if len(strings) < TRAIN_MAX_RECORDS:
        raise RuntimeError("Training corpus has too few nonempty text records")
    subset = strings[:TRAIN_MAX_RECORDS]
    output = root / "native_train"
    output.mkdir(parents=True, exist_ok=True)
    data_file = output / "wikitext_train.jsonl"
    payload = "\n".join(json.dumps({"text": x}, ensure_ascii=False) for x in subset) + "\n"
    if data_file.exists() and data_file.read_text(encoding="utf-8") != payload:
        raise RuntimeError("Frozen training corpus changed after preparation")
    data_file.write_text(payload, encoding="utf-8")
    model_path = snapshot_download(repo_id=MODEL, revision=model_rev)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "model_repo": MODEL,
        "model_revision": model_rev,
        "model_snapshot": model_path,
        "dataset_repo": TRAIN_DATASET,
        "dataset_revision": dataset_rev,
        "n_text_records": len(subset),
        "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "dataset_path": str(data_file),
        "qa_evaluation_inputs_used": False,
    }
    _save(output / "data_and_model.json", receipt)
    return receipt


def run_train(
    *,
    root: Path,
    approved: bool,
) -> dict[str, Any]:
    """Bounded 200-step official SAE-Lens smoke on separate Wikitext."""
    if not approved:
        raise PermissionError("Native SAE training requires explicit approval")
    from tasks.train.backend.training import TrainPlan, run_training

    prepared = root / "native_train" / "data_and_model.json"
    if not prepared.is_file():
        raise FileNotFoundError(
            "Run qa_native prepare-train first; local source snapshots are required"
        )
    evidence = json.loads(prepared.read_text(encoding="utf-8"))
    model_path = Path(evidence["model_snapshot"])
    data_path = Path(evidence["dataset_path"])
    if not model_path.is_dir() or not data_path.is_file():
        raise FileNotFoundError("Local model/training source missing")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != evidence["dataset_sha256"]:
        raise RuntimeError("Training source checksum mismatch")
    plan = TrainPlan(
        backend="saelens",
        model_path=str(model_path),
        dataset_path=str(data_path),
        model_revision=evidence["model_revision"],
        dataset_revision=evidence["dataset_revision"],
        hookpoint="blocks.13.hook_resid_post",
        d_in=2304,
        d_sae=4096,
        k=32,
        architecture="topk",
        max_examples=TRAIN_MAX_RECORDS,
        steps=200,
        batch_size=32,
        ctx_len=128,
        run_name="qa500_train_gemma2b",
        device="cuda",
        model_dtype="float16",
        sae_dtype="float32",
    )
    plan.validate(allow_large=False)
    run_dir = task_dir(
        Path.cwd(),
        task="train",
        dataset="wikitext",
        model="gemma2_2b_base",
        sae_key="native_topk",
        layer=13,
        initialize=True,
    )
    destination = run_dir / "checkpoints" / "native_training"
    result = run_training(
        plan,
        output=destination,
        approved=True,
        allow_large=False,
        timeout_seconds=1800,
    )
    answer = {
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "task": "train",
        "model_key": "gemma2_2b_base",
        "training_corpus": TRAIN_DATASET,
        "training_corpus_revision": evidence["dataset_revision"],
        "qa_evaluation_inputs_used": False,
        "train_plan": asdict(plan),
        "native": result,
    }
    _save(root / "native_train" / "summary.json", answer)
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA500 native task controls")
    parser.add_argument("task", choices=["circuits", "prepare-train", "train"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    if args.task == "circuits":
        result = run_circuits(
            data_root=args.root / "inputs",
            result_root=args.root / "results",
            approved=args.approved,
        )
    elif args.task == "prepare-train":
        result = prepare_training(root=args.root)
    else:
        result = run_train(root=args.root, approved=args.approved)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(result.get("status") == "partial")


if __name__ == "__main__":
    raise SystemExit(main())
