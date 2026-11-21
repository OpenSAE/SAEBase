"""Stream a Dataset-selected corpus through the existing real capture operation.

Owner: capture Task. Dataset owns population, split and limits. Per-record
receipts survive partial failures; summaries never equate a first row with EOF.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

import hashlib
import json
import os
import time
from pathlib import Path

from tasks.capture.configuration import capture_section


def _prepare_ioi(cfg, pair):
    """Release SAE memory only when synthetic inputs actually need generation."""
    from saebase.datasets.selection import select_dataset

    original_device = str(next(pair.sae.parameters()).device)
    pair.sae.to("cpu")
    try:
        return select_dataset(
            cfg.dataset,
            model_key=pair.identity.model_key,
            model=pair.model.model,
            loaded_model_name=pair.identity.loaded_model_name,
            seed=int(cfg.seed),
        )
    finally:
        pair.sae.to(original_device)


def capture_dataset(cfg, *, sar_context=None):
    from saebase.runtime.performance import current, stage
    from tasks.capture.operation import _capture, _pair, capture_mode
    from saebase.runtime.sae_operations import loaded_pair_session, pair_state_fingerprints
    from saebase.datasets.selection import select_dataset
    from saebase.runtime.experiment_results import write_json

    task = cfg.get("task", {})
    mode = capture_mode(cfg)
    if mode == "sar" and sar_context is None:
        from tasks.capture.sar_mode import prepare_sar

        sar_context = prepare_sar(cfg)
    if sar_context is None:
        from tasks.capture.selection import prepare_filter

        sar_context = prepare_filter(cfg)
    pair = _pair(cfg)
    # IOI preparation evaluates logits; temporarily release the unused SAE while
    # keeping model generation/filtering on GPU, then restore before observation.
    if "sample_count" not in cfg.dataset:
        from saebase.datasets.prompts import read_prompts
        from saebase.datasets.selection import SampleSelection

        rows = read_prompts(cfg.dataset, model_key=pair.identity.model_key)
        selection = SampleSelection(
            rows, {"source": "prepared_dataset", "selected_count": len(rows)}
        )
    elif str(cfg.dataset.name) == "IOI":
        from saebase.datasets.selection import selection_path

        if (selection_path(cfg.dataset, pair.identity.model_key) / "manifest.json").is_file():
            selection = select_dataset(cfg.dataset, model_key=pair.identity.model_key)
        else:
            selection = _prepare_ioi(cfg, pair)
    else:
        selection = select_dataset(
            cfg.dataset,
            model_key=pair.identity.model_key,
            model=pair.model.model,
            loaded_model_name=pair.identity.loaded_model_name,
            seed=int(cfg.seed),
        )
    task = cfg.get("task", {})
    mode = capture_mode(cfg)
    normalization = None
    snapshots = None
    from tasks.capture.selection import needs_normalization, undefined_reason

    needs_whiten = needs_normalization(cfg, sar_context)
    cohort_rows = list(selection) if needs_whiten else None
    if needs_whiten:
        from tasks.capture.selection import reference_rows

        cohort_rows = reference_rows(cfg, pair, cohort_rows, sar_context)
    from tasks.capture.selection import observed_selection

    selection = observed_selection(selection, task_value(task, "sample_ids"))
    query = str(cfg.get("task", {}).get("prompt", "") or "").strip()
    if needs_whiten and query:
        from saebase.datasets.selection import SampleSelection

        selection = SampleSelection(
            [
                {
                    "sample_id": "explicit:" + hashlib.sha256(query.encode()).hexdigest(),
                    "prompt": query,
                }
            ],
            {
                "source": "explicit_prompt",
                "selected_count": 1,
                "normalization_cohort": selection.metadata,
            },
        )
    directory = Path.cwd() / "reports"
    directory.mkdir(exist_ok=True)
    ledger = directory / "capture_samples.jsonl"
    progress = Path.cwd() / "metadata/capture_progress.json"
    counts = {k: 0 for k in ("completed", "undefined", "blocked", "unsupported", "failed")}
    planned = int(selection.metadata["selected_count"])
    if planned < 1:
        raise ValueError("Capture requires a nonempty selected cohort")
    raw_seen = excluded = attempted = ledger_rows = 0
    reasons = []
    digest = hashlib.sha256()
    selection_exhausted = False
    stream_error = None
    sources = {}
    sar_artifacts = []
    model_answers = {}
    annotation_items = []

    def report():
        return {
            "sample_count": sum(counts.values()),
            "attempted_count": attempted,
            "planned_count": planned,
            "captured_count": counts["completed"] + counts["blocked"] + counts["undefined"],
            "remaining_count": planned - sum(counts.values()),
            "raw_records_seen": raw_seen,
            "excluded_records": excluded,
            "counts": dict(counts),
            "selection_exhausted": selection_exhausted,
            "selection": selection.metadata,
            "samples_path": str(ledger),
        }

    with ledger.open("x") as output, loaded_pair_session(pair):
        with stage("checkpoint_identity"):
            model_sha, sae_sha = pair_state_fingerprints(pair)
        try:
            if needs_whiten:
                from tasks.capture.selection import fit_reference

                with stage("reference_normalization_and_frequency"):
                    normalization, snapshots = fit_reference(
                        cfg, pair, cohort_rows, sar_context, Path.cwd() / "metadata"
                    )
            from tasks.capture.model_answer import answer_rows

            for row, answer in answer_rows(cfg, pair, selection):
                model_answers[row["sample_id"]] = answer
                sample_started = time.perf_counter()
                raw_seen += 1
                prompt = str(row.get("prompt", ""))
                digest.update(
                    json.dumps(
                        {
                            "id": row["sample_id"],
                            "prompt": prompt,
                            "split": row.get("split"),
                            **({"sar_sample": row} if mode in {"sar", "g_s", "I"} else {}),
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode()
                )
                if not prompt.strip():
                    excluded += 1
                    status = "unsupported"
                    entry = {
                        "sample_id": row["sample_id"],
                        "status": status,
                        "reason": row.get("exclusion_reason", "empty_source_text"),
                    }
                else:
                    attempted += 1
                    try:
                        inputs = {
                            "pair": pair,
                            "prompt": prompt,
                            "sample": row,
                            "model_answer": model_answers[row["sample_id"]],
                        }
                        if needs_whiten:
                            inputs.update(
                                snapshot=snapshots.get((row["sample_id"], prompt)),
                                normalization=normalization,
                            )
                        if sar_context is not None:
                            inputs["sar_context"] = sar_context
                        with stage("capture_sample"):
                            result = _capture(cfg, **inputs)
                        model_answers[row["sample_id"]] = result.get("model_answer", answer)
                        sar_artifacts.extend(result.get("required_artifacts", []))
                        undefined = undefined_reason(result)
                        status = (
                            "blocked"
                            if result["completion"]["status"] != "complete"
                            else "undefined"
                            if undefined
                            else "completed"
                        )
                        annotation = result["neuronpedia"]
                        sha = annotation.get("export_sha256")
                        if sha:
                            if sha not in sources:
                                sources[sha] = {
                                    k: annotation[k]
                                    for k in ("provenance", "source_metadata", "source_validation")
                                    if k in annotation
                                }
                                write_json(
                                    Path.cwd() / "metadata/capture_annotation_sources.json", sources
                                )
                            for key in ("provenance", "source_metadata"):
                                annotation.pop(key, None)
                            annotation["source_ref"] = (
                                "metadata/capture_annotation_sources.json#" + sha
                            )
                        entry = {
                            "sample_id": row["sample_id"],
                            "split": row.get("split"),
                            "source_index": row.get("source_index"),
                            "status": status,
                            "result": result,
                        }
                        if status == "undefined":
                            entry["reason"] = undefined
                    except Exception as exc:
                        from saebase.datasets.answers import ReferenceAnswerUnavailable
                        from saebase.api import UnsupportedCapability

                        status = (
                            "unsupported"
                            if isinstance(exc, (ReferenceAnswerUnavailable, UnsupportedCapability))
                            else "failed"
                        )
                        entry = {
                            "sample_id": row["sample_id"],
                            "status": status,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    if status == "unsupported" and "error" in entry:
                        entry["reason"] = entry["error"]["message"]
                    if status != "completed" and len(reasons) < 100:
                        reasons.append(
                            {
                                "status": status,
                                "reason": f"{row['sample_id']}: "
                                + (
                                    entry.get("reason")
                                    or entry.get("error", {}).get("message")
                                    or result.get("model_answer", {})
                                    .get("error", {})
                                    .get("message")
                                    or "Published feature annotations incomplete"
                                ),
                            }
                        )
                with stage("sample_serialization_and_write"):
                    output.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                if "result" in entry:
                    feature_completion = (
                        entry["result"]
                        .get("stages", {})
                        .get("features_and_annotations", entry["result"]["completion"])
                    )
                    annotation_items.append(
                        {
                            "status": "completed"
                            if feature_completion["status"] == "complete"
                            else "blocked"
                        }
                    )
                counts[status] += 1
                ledger_rows += 1
                if current() is not None:
                    current().sample(
                        row["sample_id"],
                        time.perf_counter() - sample_started,
                        status=status,
                        timing_scope="question processing and ledger serialization; shared normalization timed separately",
                    )
                if current() is not None or raw_seen % 100 == 0:
                    output.flush()
                    write_json(progress, report())
            selection_exhausted = True
        except Exception as exc:
            stream_error = {"type": type(exc).__name__, "message": str(exc)}
            reasons.append(
                {"status": "failed", "reason": "Dataset iteration did not finish: " + str(exc)}
            )
        finally:
            output.flush()
            os.fsync(output.fileno())
            write_json(progress, report())
    from saebase.results import completion_summary

    def summarize(values):
        return completion_summary(
            [
                {
                    "status": status,
                    **(
                        {"reason": "Capture measurement is undefined; see sample records"}
                        if status == "undefined"
                        else {}
                    ),
                }
                for status, count in values.items()
                for _ in range(count)
            ],
            expected=planned,
            unit="sample",
        )

    completion = summarize(counts)
    completion["reasons"].extend(reasons)
    capture_counts = {**counts, "completed": counts["completed"] + counts["blocked"], "blocked": 0}
    captured = capture_counts["completed"] + capture_counts["undefined"]
    capture_stage = summarize(capture_counts)
    annotation_stage = completion_summary(annotation_items, expected=captured, unit="sample")
    from models.numerics import identity as numerical_identity
    from tasks.capture.model_answer import summary as answer_summary

    return {
        **report(),
        "mode": mode,
        "normalization": normalization.metadata if normalization is not None else None,
        "capture_format_version": 3,
        "operation": "capture",
        "status": "failed" if counts["failed"] or stream_error else "executed",
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "sae_identity": pair.identity.to_dict() if hasattr(pair.identity, "to_dict") else None,
        "layer": pair.identity.layer,
        "site": pair.site.hook_name,
        "feature_count": pair.identity.d_sae,
        "ledger_rows": ledger_rows,
        "source_exhausted": selection_exhausted,
        "completion": completion,
        "source_error": stream_error,
        "error": stream_error["message"] if stream_error else None,
        "stages": {
            "capture": capture_stage,
            "annotation": annotation_stage,
            "model_answer": answer_summary(list(model_answers.values())),
        },
        "required_artifacts": sar_artifacts
        + [{"path": "reports/capture_samples.jsonl", "bytes": ledger.stat().st_size}]
        + (
            [
                {
                    "path": "metadata/capture_annotation_sources.json",
                    "bytes": (Path.cwd() / "metadata/capture_annotation_sources.json")
                    .stat()
                    .st_size,
                }
            ]
            if sources
            else []
        )
        + (
            [
                {
                    "path": "metadata/" + name,
                    "bytes": (Path.cwd() / "metadata" / name).stat().st_size,
                }
                for name in ("capture_normalization.npz", "capture_normalization.json")
            ]
            if normalization is not None
            else []
        ),
        "input_identity": {
            "numerics": numerical_identity(),
            "schema": "saebase.task_input.v1",
            "task": "capture",
            "protocol_id": "saebase.capture.dataset.I.v1"
            if mode == "I"
            else "saebase.capture.dataset.sar.v1"
            if mode == "sar"
            else "saebase.capture.dataset.whiten.v1"
            if mode == "whiten"
            else "saebase.capture.dataset.v1",
            "execution": {
                "output_scope": task_value(cfg.get("task", {}), "output_scope", "last_token"),
                "capture_mode": mode,
                "top_feature_num": task_value(cfg.get("task", {}), "top_feature_num", 10),
                "sar_algorithm": "exhaustive"
                if mode == "I"
                else "surrogate"
                if mode == "sar"
                else None,
                "filter": capture_section(task, "selection").get("candidate_count", "all"),
                "capture_count": capture_section(task, "output").get("feature_count"),
            },
            "normalization": normalization.identity if normalization is not None else None,
            **(
                {"sar": {**sar_context["settings"], "source_sha256": sar_context["source_sha256"]}}
                if mode == "sar" and sar_context is not None and "source_sha256" in sar_context
                else {}
            ),
            "model_sha256": model_sha,
            "sae_sha256": sae_sha,
            "data_sha256": digest.hexdigest(),
            "site": {
                "layer": pair.identity.layer,
                "hook_name": pair.site.hook_name,
                **(
                    {"selection": "all_prompt_tokens"}
                    if task_value(cfg.get("task", {}), "output_scope") == "all_tokens"
                    else {"position": -1}
                ),
            },
            "selection": selection.metadata,
        },
    }
