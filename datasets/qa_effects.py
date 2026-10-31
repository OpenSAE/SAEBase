"""Frozen, small-budget QA500 intervention and generation experiments.

Feature selection is per-prompt and therefore post-selected. Report effect
magnitudes descriptively; these are not held-out causal feature-meaning tests.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from saebase.datasets.qa_standard import PROTOCOL_ID, verify_dataset
from saebase.datasets.qa_inference import make_input


TASK_BUDGET = {"intervene": 20, "generate": 5}
FEATURE_SELECTION_RULE = "max_abs_nonzero_activation_on_same_input_v1"


def _candidate_rows(
    *, data_root: Path, results_root: Path, dataset_id: str,
    model_key: str, sae_key: str, layer: int, task: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    verify_dataset(dataset_id, data_root)
    root = (
        results_root / PROTOCOL_ID / model_key / sae_key / f"L{layer}" / dataset_id
    )
    capture_file = root / "capture.jsonl"
    source_file = data_root / dataset_id / "questions.jsonl"
    sources = {
        record["sample_id"]: record
        for record in (json.loads(line) for line in source_file.read_text().splitlines())
    }
    if not capture_file.exists():
        raise FileNotFoundError("Full capture results required before task effects")
    captures = [json.loads(line) for line in capture_file.read_text().splitlines() if line]
    by_id = {row["sample_id"]: row for row in captures}
    if len(by_id) != 500 or len(captures) != 500:
        raise RuntimeError("Effects require all 500 unique captured questions")
    if set(by_id) != set(sources):
        raise RuntimeError("Capture IDs and frozen question IDs differ")
    if any(row.get("status") != "executed" for row in captures):
        raise RuntimeError("Cannot run effects on unexecuted capture data")
    ordered = sorted(
        sources, key=lambda k: (hashlib.sha256(f"{task}:{k}".encode()).hexdigest(), k)
    )
    return [(sources[k], by_id[k]) for k in ordered[:TASK_BUDGET[task]]]


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["sample_id"] in done:
                    raise RuntimeError("Duplicate task observation in saved results")
                done[row["sample_id"]] = row
    return done


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


@torch.no_grad()
def run_effects(
    pair: Any, *, task: str, data_root: Path, results_root: Path,
    dataset_id: str, model_key: str, sae_key: str, layer: int,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    if task not in TASK_BUDGET:
        raise ValueError("Only intervention and generation have a bounded QA effect protocol")
    choices = _candidate_rows(
        data_root=data_root, results_root=results_root, dataset_id=dataset_id,
        model_key=model_key, sae_key=sae_key, layer=layer, task=task,
    )
    root = results_root / PROTOCOL_ID / model_key / sae_key / f"L{layer}" / dataset_id
    output = root / f"{task}.jsonl"
    existing = _existing(output)
    expected = {item["sample_id"] for item, _ in choices}
    if not set(existing).issubset(expected):
        raise RuntimeError("Existing effects contain a foreign sample ID")
    from saebase.store import sha256_state_dict

    actual_sha = sha256_state_dict(pair.sae)
    for item, captured in choices:
        sid = item["sample_id"]
        if captured["sae_sha256"] != actual_sha or captured["site_id"] != (
            f"{model_key}:{sae_key}:L{layer}"
        ):
            raise RuntimeError("SAE identity mismatch between capture and effects")
        if captured["prompt_sha256"] != item["prompt_sha256"]:
            raise RuntimeError("Prompt changed between capture and effects")
        if sid in existing:
            continue
        features = captured["top_features"]
        if not features or float(features[0]["activation"]) == 0:
            _append(root / f"{task}_failures.jsonl", {
                "sample_id": sid, "status": "blocked",
                "reason": "no_nonzero_feature_at_capture_position",
            })
            continue
        fid = int(features[0]["feature_id"])
        if not (0 <= fid < int(pair.identity.d_sae)):
            raise RuntimeError("Captured feature ID outside the actual SAE width")
        tokens, input_meta = make_input(pair, item, model_key)
        if tokens.size(1) > max_tokens:
            raise RuntimeError("Previously captured QA input unexpectedly exceeds token budget")
        if captured.get("token_ids_sha256") != input_meta["token_ids_sha256"]:
            raise RuntimeError("Model tokenization changed after capture")
        if task == "intervene":
            from saebase.api import evaluate_edit

            result = evaluate_edit(
                pair.model, pair.sae, site=pair.site, tokens=tokens,
                replacements={fid: 0.0}, verify_identity=True,
            )
            observed = float(result.snapshot.codes[fid].float().item())
            prior = float(features[0]["activation"])
            if not math.isclose(observed, prior, rel_tol=2e-3, abs_tol=2e-4):
                raise RuntimeError("Real feature activation differs from frozen capture")
            before = result.baseline_logits.float()
            after = result.edited_logits.float()
            p = torch.softmax(before, dim=-1)
            kl = float(torch.sum(p * (
                torch.log_softmax(before, -1) - torch.log_softmax(after, -1)
            )).item())
            delta = after - before
            measurement = {
                "feature_activation_replayed": observed,
                "logit_delta_l2": float(torch.linalg.vector_norm(delta).item()),
                "max_abs_logit_delta": float(delta.abs().max().item()),
                "kl_clean_to_ablated": max(0.0, kl),
                "clean_argmax_token_id": int(before.argmax().item()),
                "ablated_argmax_token_id": int(after.argmax().item()),
                "identity_replay_passed": True,
            }
        else:
            from saebase.api import capture_features
            from saebase.store import SampleRecord, TokenSelection
            from tasks.intervene.backend.specs import Condition, MeasurementSample
            from tasks.intervene.backend.steering import (
                GenerationSpec, compare_steered_generation,
            )

            snapshot = capture_features(pair.model, pair.sae, site=pair.site, tokens=tokens)
            observed = float(snapshot.codes[fid].float().item())
            if not math.isclose(
                observed, float(features[0]["activation"]), rel_tol=2e-3, abs_tol=2e-4
            ):
                raise RuntimeError("Generation feature activation does not match capture")
            record = SampleRecord(
                sample_id=sid,
                token_ids=tuple(int(k) for k in tokens[0].detach().cpu().tolist()),
                selection=TokenSelection(mode="last_valid"),
                prompt_group_id=item["prompt_group_id"],
            )
            measurement = compare_steered_generation(
                pair.model, pair.sae, site=pair.site,
                sample=MeasurementSample(record, readout_position=-1),
                condition=Condition(
                    name="qa_selected_feature_ablation", operation="ablate",
                    feature_ids=(fid,), value=0.0,
                ),
                spec=GenerationSpec(
                    max_new_tokens=16, seed=20261031, temperature=0.0,
                ),
            )
        row = {
            "status": "executed", "protocol_id": PROTOCOL_ID, "task": task,
            "sample_id": sid, "dataset_id": dataset_id, "model_key": model_key,
            "sae_key": sae_key, "layer": layer, "sae_sha256": actual_sha,
            "feature_id": fid, "feature_selection": FEATURE_SELECTION_RULE,
            "intervention": "ablate", "replacement_value": 0.0,
            "input": input_meta, "measurement": measurement,
        }
        _append(output, row)
        existing[sid] = row
    summary = {
        "protocol_id": PROTOCOL_ID, "task": task, "dataset_id": dataset_id,
        "model_key": model_key, "sae_key": sae_key, "layer": layer,
        "expected": len(choices), "executed": len(existing),
        "status": "complete" if len(existing) == len(choices) else "partial",
        "selected_on_evaluation_input": True,
    }
    (root / f"{task}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
