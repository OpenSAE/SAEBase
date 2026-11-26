"""Isolated, budgeted Precheck worker. Only this module runs real test operations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tasks.precheck.report import error_result


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_is_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(v) for v in value)
    return True


def _validate_task(task: str, data: Any) -> None:
    """The Task must have run, and its native minimum evidence must exist."""
    if not isinstance(data, dict) or data.get("status") != "executed":
        raise RuntimeError(f"{task} did not return an executed result")
    if data.get("operation") != task or not _is_finite(data):
        raise RuntimeError(f"{task} operation identity or numeric result is invalid")
    if task == "capture":
        if not isinstance(data.get("feature_count"), int) or data["feature_count"] < 1:
            raise RuntimeError("Capture did not return any SAE features")
        if data.get("capture_format_version") == 3:
            path = Path(data.get("samples_path", ""))
            if (
                not data.get("selection_exhausted")
                or data.get("sample_count", 0) < 1
                or not path.is_file()
            ):
                raise RuntimeError("Dataset capture did not finish the selected samples")
            count = sum(1 for line in path.open() if line.strip())
            if count != data.get("ledger_rows"):
                raise RuntimeError("Dataset capture ledger count differs from its summary")
        elif not isinstance(data.get("top_features"), list):
            raise RuntimeError("Capture did not return feature observations")
    elif task == "intervene":
        if "resolved_plan" in data:
            expected = data["resolved_plan"]["measurements"]
            if expected < 1 or data.get("completion", {}).get("completed") != expected:
                raise RuntimeError("Intervene did not commit every requested measurement")
        elif not isinstance(data.get("logit_delta_l2"), (float, int)):
            raise RuntimeError("Intervene did not return measured logit differences")
    elif task == "generate":
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Generate did not return a real generation trace")
        traces = (
            [
                {
                    "baseline_generated_token_ids": row.get("baseline_generated_token_ids"),
                    "steered_generated_token_ids": row.get("edited_generated_token_ids"),
                    "steered_steps": row.get("steps"),
                }
                for row in result["runs"]
            ]
            if "runs" in result
            else [result]
        )
        if not traces:
            raise RuntimeError("Generate did not return any condition traces")
        for trace in traces:
            for key in ("baseline_generated_token_ids", "steered_generated_token_ids"):
                if not isinstance(trace.get(key), list) or not trace[key]:
                    raise RuntimeError(f"Generate missing real generated tokens: {key}")
            if not isinstance(trace.get("steered_steps"), list) or not trace["steered_steps"]:
                raise RuntimeError("Generate did not record real intervention steps")
    elif task == "circuits":
        native = data.get("native")
        if not isinstance(native, dict) or not list(Path("figures").rglob("*.*")):
            raise RuntimeError("Circuit tracer returned no real graph artifacts")
        if native.get("native_json_requires_source_specific_converter"):
            raise RuntimeError(
                "BLOCKED_VALIDATION: Native circuit graph requires validated "
                "source-specific conversion and node/edge checks before PASS"
            )
    elif task == "explain":
        hypothesis = data.get("hypothesis")
        if not isinstance(hypothesis, dict) or hypothesis.get("status") != "heldout_tested":
            raise ValueError("Explain PASS requires a real heldout predictor and tested result")
        if not isinstance(hypothesis.get("heldout_score"), (int, float)):
            raise RuntimeError("Explain heldout metric missing")
    elif task == "benchmark":
        metrics = data.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise RuntimeError("Benchmark produced no real metric records")
        if any(
            not isinstance(m, dict)
            or m.get("status") != "completed"
            or int(m.get("n_valid", 0)) < 1
            for m in metrics
        ):
            raise ValueError(
                "Benchmark PASS requires every selected metric completed with real valid samples"
            )
    elif task == "train":
        native = data.get("native")
        if not isinstance(native, dict) or not any(
            p.is_file() for p in Path("checkpoints").rglob("*")
        ):
            raise RuntimeError("Train produced no native checkpoint")
        if native.get("load_and_encode_verified") is not True:
            raise RuntimeError(
                "BLOCKED_VALIDATION: Native SAE checkpoint requires verified "
                "reload and real encode/decode before PASS"
            )


def check(request: dict[str, Any]) -> dict[str, Any]:
    kind = str(request["kind"])
    try:
        if kind == "prepare_assets":
            from saebase.setup.assets import prepare_environment

            prepared = prepare_environment(
                **request["parameters"], log_dir=Path.cwd() / "logs" / "setup"
            )
            return {
                **prepared,
                "status": "PASS" if prepared["status"] == "READY" else prepared["status"],
                "preparation": prepared,
                "scientific_test": "NOT_RUN",
            }
        if kind == "prepare_dataset":
            from saebase.datasets.prompts import prepare_selected_dataset

            from tasks.precheck.report import blocked

            try:
                prepared = prepare_selected_dataset(
                    OmegaConf.create(request["config"]),
                    n_generate=request["n_generate"],
                    min_kept=request["min_kept"],
                )
            except (OSError, ValueError) as exc:
                return blocked(
                    stage="dataset_prepare",
                    code="BLOCKED_DATA_PREPARATION",
                    reason=str(exc),
                    attempted=["Dataset-owned preparation"],
                    repair="Inspect the selected input and Dataset preparation log; existing artifacts were preserved",
                )
            except RuntimeError as exc:
                if "kept" not in str(exc) and "active writer" not in str(exc):
                    raise
                return blocked(
                    stage="dataset_prepare",
                    code="BLOCKED_DATA_PREPARATION",
                    reason=str(exc),
                    attempted=["Dataset generation, tokenization and original filtering"],
                    repair="Review the recorded preparation sample budget or competing writer without changing the Dataset filter",
                )
            return {
                "status": "PASS",
                "preparation": prepared,
                "scope": "Dataset preparation; not a model or Task check",
            }
        from saebase.extensions.registry import ensure_extensions

        ensure_extensions(request.get("extensions", []))
        if kind == "model_sae_pair":
            from tasks.precheck.sae_checks import verify_pair, verify_target

            if request.get("sae_key"):
                return verify_pair(request["sae_key"], device=request.get("device", "cpu"))
            return verify_target(request["model"], device=request.get("device", "cuda"))
        if kind == "sae_checkpoint":
            from tasks.precheck.checkpoint import probe_checkpoint

            return probe_checkpoint(request["sae_key"], request["layer"], device=request["device"])
        if kind == "sae":
            from tasks.precheck.sae_checks import check_real_sae

            probes = request.get("prompts", [str(request["prompt"])])
            checks = []
            from contextlib import nullcontext

            from saebase.runtime.sae_operations import (
                _ACTIVE_PAIR,
                load_selection,
                loaded_pair_session,
                resolve_sae,
            )

            session = nullcontext()
            if "prompts" in request and _ACTIVE_PAIR.get() is None:
                selected = resolve_sae(
                    OmegaConf.create({"name": request["model_key"]}),
                    request["sae_key"],
                    request["layer"],
                    operation=request.get("operation", "precheck"),
                )
                pair = load_selection(selected, request["device"])
                session = loaded_pair_session(pair)
            with session:
                for index, prompt in enumerate(probes):
                    try:
                        checked = check_real_sae(
                            operation=str(request.get("operation", "precheck")),
                            capture_mode=str(request.get("capture_mode", "non_whiten")),
                            whiten_chunk_size=int(request.get("whiten_chunk_size", 256)),
                            annotation_options=request.get("annotation_options")
                            if index == 0
                            else None,
                            model_key=str(request["model_key"]),
                            sae_key=str(request["sae_key"]),
                            layer=int(request["layer"]),
                            prompt=prompt,
                            device=str(request["device"]),
                        )
                    except Exception as exc:
                        checked = error_result(
                            exc,
                            stage=f"sae_sample:{index}",
                            attempted=["real capture/encode/decode"],
                        )
                    checks.append(checked)
            if "prompts" not in request:
                result = checks[0]
            else:
                result = {
                    "status": "FAIL"
                    if any(row["status"] == "FAIL" for row in checks)
                    else "BLOCKED"
                    if any(row["status"] == "BLOCKED" for row in checks)
                    else "PASS",
                    "observed": len(checks),
                    "sample_ids": request.get("sample_ids", []),
                    "samples": checks,
                    "scope": "sampled real capture/encode/decode; not full-cohort scientific acceptance",
                }
        elif kind == "dataset":
            from tasks.precheck.dataset_checks import check_dataset_artifact

            result = check_dataset_artifact(
                dataset_key=str(request["dataset_key"]),
                model_key=str(request["model_key"]),
                data_root=Path(str(request["data_root"])),
                max_records=int(request.get("max_records", 500)),
                sampling_seed=int(request.get("sampling_seed", 42)),
                dataset_cfg=OmegaConf.create(request["dataset_config"])
                if request.get("dataset_config")
                else None,
            )
        elif kind == "model":
            from tasks.precheck.model_checks import check_real_model

            result = check_real_model(
                model_key=str(request["model_key"]),
                model_name=str(request["model_name"]),
                prompt=str(request["prompt"]),
                device=str(request["device"]),
            )
        elif kind == "compatibility":
            import hashlib

            from saebase.datasets.prompts import read_prompts

            from tasks.precheck.model_checks import check_real_model

            data_path = Path(str(request["data_root"])) / str(request["dataset_key"])
            cfg = OmegaConf.create(
                {
                    "name": str(request["dataset_key"]),
                    "data_path": str(data_path),
                    "prompt_file": "",
                }
            )
            rows = read_prompts(cfg, model_key=str(request["model_key"]))
            budget = int(request.get("max_samples", 2))
            if not 1 <= budget <= 2:
                raise ValueError("Compatibility Precheck requires 1–2 real samples")
            prompts = []
            for item in rows:
                value = item.get("prompt") if isinstance(item, dict) else None
                if isinstance(value, str) and value.strip():
                    prompts.append(value)
                if len(prompts) >= budget:
                    break
            if not prompts:
                raise ValueError("No real compatible prompt records")
            observed = []
            for prompt in prompts:
                if request.get("site_id"):
                    from tasks.precheck.sae_checks import check_real_sae

                    check = check_real_sae(
                        model_key=str(request["model_key"]),
                        sae_key=str(request["sae_key"]),
                        layer=int(request["layer"]),
                        prompt=prompt,
                        device=str(request["device"]),
                    )
                else:
                    check = check_real_model(
                        model_key=str(request["model_key"]),
                        model_name=str(request["model_name"]),
                        prompt=prompt,
                        device=str(request["device"]),
                    )
                observed.append(check)
            states = {row.get("status") for row in observed}
            result = {
                "status": (
                    "FAIL"
                    if "FAIL" in states
                    else "BLOCKED"
                    if "BLOCKED" in states
                    else "PASS"
                    if states == {"PASS"}
                    else "FAIL"
                ),
                "scope": "bounded real pair native smoke; no scientific acceptance",
                "target_id": str(request["target_id"]),
                "dataset_key": str(request["dataset_key"]),
                "model_key": str(request["model_key"]),
                "site_id": request.get("site_id"),
                "sample_count": len(prompts),
                "prompt_sha256": [
                    hashlib.sha256(item.encode("utf-8")).hexdigest() for item in prompts
                ],
                "results": observed,
                "scientific_acceptance": "NOT_COVERED",
            }
        elif kind == "task_preparation":
            from saebase.registry import get_task_plugin

            cfg = OmegaConf.create(request["config"])
            task = get_task_plugin(str(request["operation"])).task_class()
            task.validate_cfg(cfg)
            result = task.prepare_execution(cfg)
            result = {**result, "task": request["operation"], "scientific_computation": "NOT_RUN"}
        elif kind == "task":
            from saebase.registry import get_task_plugin

            name = str(request["operation"])
            cfg = OmegaConf.create(request["config"])
            task = get_task_plugin(name).task_class()
            task.validate_cfg(cfg)
            outcome = task.run_validated(cfg)
            if outcome.task != name:
                raise RuntimeError(f"{name} returned a result for {outcome.task}")
            result = dict(outcome.summary)
            result.setdefault("operation", outcome.task)
            result.setdefault("status", outcome.status)
            if outcome.completion:
                result.setdefault("completion", outcome.completion)
            _validate_task(name, result)
            completion = result.get("completion", {})
            if completion and completion.get("status") != "complete":
                return {
                    "status": "FAIL" if completion["status"] == "failed" else "BLOCKED",
                    "task": name,
                    "code": "TASK_INCOMPLETE",
                    "reason": "Native Task has incomplete required work",
                    "native_result": result,
                }
            result = {
                "status": "PASS",
                "task": name,
                "scope": "one bounded native Task smoke, not scientific performance acceptance",
                "native_result": result,
            }
        else:
            raise TypeError(f"Invalid worker kind: {kind}")
        if isinstance(result, dict) and "run_id" in request:
            result["run_id"] = str(request["run_id"])
            result["source_sha"] = str(request.get("source_sha", ""))
        return result
    except Exception as exc:
        return error_result(
            exc,
            stage=f"{kind}:{request.get('operation', request.get('sae_key'))}",
            attempted=["isolated real worker invoked"],
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    req = json.loads(args.input.read_text(encoding="utf-8"))
    result = check(req)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    # Parent orchestrator reads the receipt even on a BLOCKED/FAIL.
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
