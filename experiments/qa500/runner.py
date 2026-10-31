"""Single-command, fixed-policy QA500 agent execution.

An agent chooses only the phase and hardware allocation supplied by its owner.
All datasets, model/SAE sites, budgets, sample selection and result names are
frozen in code. No HF publication, training or paid inference is implicit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from experiments.qa500.cohort import DATASETS, PROTOCOL_ID, prepare_dataset, verify_dataset
from experiments.qa500.capture import SITES, site_list, run_capture
from experiments.qa500.baseline import run_baseline
from experiments.qa500.effects import run_effects
from experiments.qa500.explain import explain_from_captures
from experiments.qa500.benchmark import benchmark_from_capture


DATA_ROOT = Path("experiment/qa500/inputs")
RESULT_ROOT = Path("experiment/qa500/results")
TASK_STATUS = {
    "capture": "executable",
    "benchmark": "frozen_500_question_core_summary_with_bootstrap",
    "intervene": "executable_after_full_capture",
    "generate": "executable_after_full_capture",
    "explain": "external_annotations_only_not_independent_validation",
    "circuits": "native_tracer_requires_separate_approval",
    "train": "native_training_requires_separate_corpus_and_approval",
}


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment() -> dict[str, Any]:
    import torch
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("torch", "transformer-lens", "sae-lens", "datasets", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "source_commit": _git_revision(),
        "packages": packages,
        "cuda": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu_name"] = gpu.name
        result["gpu_total_bytes"] = int(gpu.total_memory)
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def make_plan() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "n_datasets": len(DATASETS),
        "dataset_ids": list(DATASETS),
        "n_questions_per_dataset": 500,
        "n_models": len(SITES),
        "n_sites": len(site_list()),
        "expected_model_qa_baselines": len(SITES) * 500 * len(DATASETS),
        "expected_capture_rows": 500 * len(DATASETS) * len(site_list()),
        "expected_benchmark_summaries": len(DATASETS) * len(site_list()),
        "expected_intervention_rows": 20 * len(DATASETS) * len(site_list()),
        "expected_generation_rows": 5 * len(DATASETS) * len(site_list()),
        "task_status": TASK_STATUS,
        "sites": site_list(),
        "selection": "fixed sources, SHA256-stable sample IDs, sorted per-task hashes",
        "publication": "disabled_until_independent_acceptance",
    }


def prepare_all(data_root: Path) -> dict[str, Any]:
    report = {}
    for key in DATASETS:
        try:
            report[key] = {"status": "verified", **prepare_dataset(key, data_root)}
        except Exception as exc:
            report[key] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
    _atomic_json(data_root / "preparation_status.json", report)
    return report


def run_model(
    model_key: str,
    *,
    phase: str,
    device: str,
    data_root: Path,
    result_root: Path,
    max_tokens: int = 2048,
    neuronpedia_map: Path | None = None,
    neuronpedia_records: Path | None = None,
) -> dict[str, Any]:
    if model_key not in {entry[0] for entry in SITES}:
        raise KeyError("Model is not in frozen QA500 protocol")
    if phase not in {"pilot", "core", "effects", "complete"}:
        raise ValueError("Unknown execution phase")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; refusing a fake GPU execution")
    job_root = result_root / PROTOCOL_ID / "_receipts" / model_key
    job_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "model_key": model_key,
        "phase": phase,
        "device": device,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "steps": [],
    }
    _atomic_json(job_root / f"{phase}.json", receipt)
    pairs = next((e for e in SITES if e[0] == model_key), None)
    assert pairs is not None
    for layer in pairs[2]:
        from saebase.api import load_registered

        pair = None
        try:
            pair = load_registered(
                sae_key=pairs[1], layer=layer, model_key=model_key, device=device
            )
            from saebase.store import sha256_state_dict

            # The actual loaded *model* bytes, not just an HF display name,
            # must be the same across both SAE layers and every QA dataset.
            current_model_sha = sha256_state_dict(pair.model.model)
            if len(current_model_sha) != 64:
                raise RuntimeError("Loaded model state digest invalid")
            by_layer = receipt.setdefault("model_state_sha256_by_layer", {})
            if by_layer and current_model_sha not in set(by_layer.values()):
                raise RuntimeError("Two SAE layers loaded different base-model weights")
            by_layer[str(layer)] = current_model_sha
            _atomic_json(job_root / f"{phase}.json", receipt)
            for dataset in DATASETS:
                step: dict[str, Any] = {
                    "site": f"{model_key}:{pairs[1]}:L{layer}",
                    "dataset_id": dataset,
                    "status": "running",
                    "start_utc": datetime.now(timezone.utc).isoformat(),
                }
                receipt["steps"].append(step)
                _atomic_json(job_root / f"{phase}.json", receipt)
                try:
                    verify_dataset(dataset, data_root)
                    limit = 10 if phase == "pilot" else None
                    if phase in {"pilot", "core", "complete"}:
                        # The first SAE site per model computes QA once, never twice.
                        if layer == pairs[2][0]:
                            step["qa_baseline"] = run_baseline(
                                pair,
                                model_key=model_key,
                                dataset_id=dataset,
                                data_root=data_root,
                                output_root=result_root,
                                max_tokens=max_tokens,
                                pilot=limit,
                            )
                        step["capture"] = run_capture(
                            model_key=model_key,
                            layer=layer,
                            dataset_id=dataset,
                            data_root=data_root,
                            output_root=result_root,
                            device=device,
                            max_tokens=max_tokens,
                            limit=limit,
                            top_k=10,
                            loaded_pair=pair,
                            mapping_file=neuronpedia_map,
                            annotation_file=neuronpedia_records,
                        )
                    if phase in {"core", "complete"}:
                        if step.get("capture", {}).get("status") != "complete":
                            raise RuntimeError("Benchmark requires 500 complete Capture results")
                        step["benchmark"] = benchmark_from_capture(
                            data_root=data_root,
                            result_root=result_root,
                            model_key=model_key,
                            sae_key=pairs[1],
                            layer=layer,
                            dataset_id=dataset,
                        )
                    if phase in {"effects", "complete"}:
                        if (
                            phase == "complete"
                            and step.get("capture", {}).get("status") != "complete"
                        ):
                            raise RuntimeError("Effects require complete Capture coverage")
                        step["intervene"] = run_effects(
                            pair,
                            task="intervene",
                            data_root=data_root,
                            results_root=result_root,
                            dataset_id=dataset,
                            model_key=model_key,
                            sae_key=pairs[1],
                            layer=layer,
                            max_tokens=max_tokens,
                        )
                        step["generate"] = run_effects(
                            pair,
                            task="generate",
                            data_root=data_root,
                            results_root=result_root,
                            dataset_id=dataset,
                            model_key=model_key,
                            sae_key=pairs[1],
                            layer=layer,
                            max_tokens=max_tokens,
                        )
                        step["explain"] = {
                            "status": "source_annotations_only",
                            "details": explain_from_captures(
                                data_root=data_root,
                                result_root=result_root,
                                model_key=model_key,
                                sae_key=pairs[1],
                                layer=layer,
                                dataset_id=dataset,
                                mapping_file=neuronpedia_map,
                                annotation_file=neuronpedia_records,
                            )["candidates"],
                        }
                    statuses = [
                        result["status"]
                        for key, result in step.items()
                        if key in {"qa_baseline", "capture", "benchmark", "intervene", "generate"}
                    ]
                    step["status"] = (
                        "complete"
                        if statuses and all(x == "complete" for x in statuses)
                        else "partial"
                    )
                except (KeyboardInterrupt, SystemExit):
                    step["status"] = "interrupted"
                    raise
                except Exception as exc:
                    step["status"] = "blocked"
                    step["error_type"] = type(exc).__name__
                    step["reason"] = str(exc)[:1000]
                    step["traceback"] = traceback.format_exc()[-3000:]
                step["end_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(job_root / f"{phase}.json", receipt)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            receipt["steps"].append(
                {
                    "site": f"{model_key}:{pairs[1]}:L{layer}",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:1000],
                    "traceback": traceback.format_exc()[-3000:],
                }
            )
            _atomic_json(job_root / f"{phase}.json", receipt)
        finally:
            del pair
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
    receipt["end_utc"] = datetime.now(timezone.utc).isoformat()
    statuses = [step["status"] for step in receipt["steps"]]
    receipt["status"] = (
        "complete"
        if len(statuses) == len(DATASETS) * 2 and all(x == "complete" for x in statuses)
        else "partial"
    )
    _atomic_json(job_root / f"{phase}.json", receipt)
    return {
        "model_key": model_key,
        "phase": phase,
        "status": receipt["status"],
        "complete_steps": statuses.count("complete"),
        "total_steps": len(statuses),
        "receipt": str(job_root / f"{phase}.json"),
    }


def run_matrix(
    *,
    phase: str,
    device: str,
    data_root: Path,
    result_root: Path,
    model_keys: list[str],
    max_tokens: int,
    neuronpedia_map: Path | None,
    neuronpedia_records: Path | None,
) -> dict[str, Any]:
    output = {}
    for name in model_keys:
        try:
            output[name] = run_model(
                name,
                phase=phase,
                device=device,
                data_root=data_root,
                result_root=result_root,
                max_tokens=max_tokens,
                neuronpedia_map=neuronpedia_map,
                neuronpedia_records=neuronpedia_records,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            output[name] = {
                "status": "blocked",
                "reason": str(exc)[:1000],
                "error_type": type(exc).__name__,
            }
        _atomic_json(result_root / PROTOCOL_ID / "_receipts" / f"{phase}_matrix.json", output)
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frozen QA500 model+SAE execution manager")
    ap.add_argument(
        "action", choices=["plan", "prepare", "verify", "pilot", "core", "effects", "complete"]
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    ap.add_argument("--model", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--neuronpedia-map", type=Path)
    ap.add_argument("--neuronpedia-records", type=Path)
    args = ap.parse_args(argv)
    if args.action == "plan":
        result: dict[str, Any] = make_plan()
    elif args.action == "prepare":
        result = prepare_all(args.data_root)
    elif args.action == "verify":
        result = {}
        for key in DATASETS:
            try:
                result[key] = {"status": "verified", **verify_dataset(key, args.data_root)}
            except Exception as exc:
                result[key] = {"status": "blocked", "reason": str(exc)[:500]}
    else:
        if args.model != "all" and args.model not in {site[0] for site in SITES}:
            ap.error("Unknown model key")
        keys = [x[0] for x in SITES] if args.model == "all" else [args.model]
        result = run_matrix(
            phase=args.action,
            device=args.device,
            data_root=args.data_root,
            result_root=args.result_root,
            model_keys=keys,
            max_tokens=args.max_tokens,
            neuronpedia_map=args.neuronpedia_map,
            neuronpedia_records=args.neuronpedia_records,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return (
        1
        if any(
            row.get("status") in {"blocked", "partial"}
            for row in result.values()
            if isinstance(row, dict)
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
