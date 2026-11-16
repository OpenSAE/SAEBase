"""Concrete capture task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from typing import Any
import os
from pathlib import Path

from tasks.sae_runtime import _pair, _prompt, _fingerprint_prompt


def offload_capture_tail(pair):
    """Explicit driver context: free unused block weights during prefix capture.

    Owner: capture. Reuses the loaded pair, keeps every weight's dtype/value,
    and restores device placement after precheck and capture have finished.
    """
    from contextlib import contextmanager
    import inspect

    @contextmanager
    def session():
        raw = pair.model.model
        if (
            pair.model.backend != "transformer_lens"
            or "stop_at_layer" not in inspect.signature(raw.forward).parameters
        ):
            raise ValueError("Tail offload requires a backend with explicit prefix stopping")
        if not pair.site.hook_name.startswith(f"blocks.{pair.site.layer}."):
            raise ValueError("Tail offload requires a block activation site")
        cutoff = pair.site.layer + 1
        tail = list(raw.blocks[cutoff:])
        prefix_ids = {
            id(parameter) for block in raw.blocks[:cutoff] for parameter in block.parameters()
        }
        if any(id(parameter) in prefix_ids for block in tail for parameter in block.parameters()):
            raise ValueError("Tail weights are shared with a required prefix block")
        moved = []
        try:
            for block in tail:
                device = next(block.parameters()).device
                if device.type != "cpu":
                    moved.append((block, device))
                    block.to("cpu")
            import torch

            if moved:
                torch.cuda.empty_cache()
            yield
        finally:
            if moved:
                torch.cuda.empty_cache()
            for block, device in moved:
                block.to(device)

    return session()


def neuronpedia_settings(cfg) -> dict:
    raw = cfg.task.get("neuronpedia", {})
    if not hasattr(raw, "get"):
        raise ValueError("task.neuronpedia must contain annotation settings")
    values = {
        "enabled": raw.get("enabled", True),
        "allow_network": raw.get("allow_network", True),
        "timeout_seconds": raw.get("timeout_seconds", 15),
        "total_timeout_seconds": raw.get("total_timeout_seconds", 120),
        "max_download_bytes": raw.get("max_download_bytes", 64 * 1024**2),
        "max_uncompressed_bytes": raw.get("max_uncompressed_bytes", 256 * 1024**2),
        "cache_dir": raw.get("cache_dir"),
    }
    if values["cache_dir"] is None:
        values["cache_dir"] = str(
            Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "saebase/neuronpedia"
        )
    for key in ("enabled", "allow_network"):
        if type(values[key]) is not bool:
            raise ValueError(f"task.neuronpedia.{key} must be a boolean")
    for key in (
        "timeout_seconds",
        "total_timeout_seconds",
        "max_download_bytes",
        "max_uncompressed_bytes",
    ):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"task.neuronpedia.{key} must be a positive integer")
    if not isinstance(values["cache_dir"], str) or not values["cache_dir"].strip():
        raise ValueError("task.neuronpedia.cache_dir must be a path")
    if any(
        os.environ.get(key, "").lower() in {"1", "true", "yes"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        values["allow_network"] = False
    return values


def capture_mode(cfg):
    mode = cfg.get("task", {}).get("mode", "non_whiten")
    if not isinstance(mode, str) or mode not in {"non_whiten", "whiten", "sar", "g_s", "I"}:
        raise ValueError("task.mode must be non_whiten, whiten, g_s, sar or I")
    size = cfg.get("task", {}).get("whiten_chunk_size", 256)
    if type(size) is not int or size < 1:
        raise ValueError("task.whiten_chunk_size must be a positive integer")
    raw = cfg.get("task", {}).get("sar", {})
    if mode == "sar" and raw.get("algorithm", "surrogate") != "surrogate":
        raise ValueError(
            "mode=sar is the regression method; use task.mode=I for exhaustive interaction"
        )
    if mode == "I":
        maximum = raw.get("max_exhaustive_features", 16)
        if type(maximum) is not int or maximum < 2:
            raise ValueError("task.sar.max_exhaustive_features must be an integer >= 2")
        limit = cfg.task.get("filter")
        if type(limit) is int and not 2 <= limit <= maximum:
            raise ValueError("mode=I requires 2 <= task.filter <= max_exhaustive_features")
    if mode in {"g_s", "I"}:
        if raw.get("readout", "answer_mean_logprob") != "answer_mean_logprob":
            raise ValueError("g_s and I require answer_mean_logprob")
        if raw.get("answer_backend", "full_sequence") not in {
            "full_sequence",
            "full_context_tail",
            "auto",
        }:
            raise ValueError("Invalid canonical answer_backend")
    return mode


def capture_top_feature_num(cfg: Any) -> int | None:
    """Validate the requested feature count before loading model resources."""
    if "top_k" in cfg.task:
        raise ValueError("task.top_k was renamed to task.top_feature_num")
    if (
        "filter" in cfg.task
        or "capture_count" in cfg.task
        or cfg.task.get("mode") == "g_s"
        or cfg.task.get("mode") == "I"
    ):
        from tasks.capture.selection import settings

        value = settings(cfg)["capture_count"]
        return None if value == "all" else value
    value = cfg.task.get("top_feature_num", 10)
    if type(value) is not int or value < 1:
        raise ValueError("task.top_feature_num must be a positive integer")
    return value


def input_identity(pair, snapshot, prompt: str, cfg: Any) -> dict[str, Any] | None:
    """Describe measured inputs; unsupported custom state remains explicitly unknown."""
    from saebase.store import fingerprint
    from tasks.sae_runtime import pair_state_fingerprints

    try:
        tokens = snapshot.tokens.detach().cpu().tolist()
        model_sha, sae_sha = pair_state_fingerprints(pair)
        return {
            "schema": "saebase.task_input.v1",
            "task": "capture",
            "protocol_id": "saebase.capture.v1",
            "model_sha256": model_sha,
            "sae_sha256": sae_sha,
            "data_sha256": fingerprint(
                {"tokens": tokens, "prompt_sha256": _fingerprint_prompt(prompt)}
            ),
            "site": {
                "layer": snapshot.site.layer,
                "hook_name": snapshot.site.hook_name,
                "position": snapshot.site.position,
            },
            "execution": {
                "backend": pair.model.backend,
                "output_scope": "last_token",
                "dtype": str(snapshot.activation.dtype),
                "seed": cfg.get("seed"),
                "top_feature_num": capture_top_feature_num(cfg),
            },
        }
    except (TypeError, AttributeError, ValueError):
        return None


def _capture(
    cfg: Any,
    *,
    pair=None,
    prompt=None,
    snapshot=None,
    normalization=None,
    sample=None,
    sar_context=None,
) -> dict[str, Any]:
    top_feature_num = capture_top_feature_num(cfg)
    mode = capture_mode(cfg)
    if mode in {"sar", "g_s", "I"}:
        from tasks.capture.sar_mode import behavior

        behavior(cfg, sample)
    if mode == "whiten" and normalization is None:
        raise ValueError("Whiten capture requires Dataset-owned all-token normalization")
    from saebase.api.neuronpedia import annotate_features
    from saebase.results import completion_summary

    options = neuronpedia_settings(cfg)
    from saebase.api import capture_features
    import torch

    pair = _pair(cfg) if pair is None else pair
    prompt = _prompt(cfg) if prompt is None else prompt
    if cfg.task.get("output_scope", "last_token") == "all_tokens":
        from tasks.capture.explore import capture_prompt_result

        return capture_prompt_result(cfg, pair, prompt, sample)
    snapshot = (
        snapshot
        if snapshot is not None
        else capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    )
    codes = snapshot.codes.detach().float().cpu()
    top_ids = torch.topk(
        codes.abs(), k=min(top_feature_num or codes.numel(), codes.numel())
    ).indices.tolist()
    active_ids = torch.nonzero(codes, as_tuple=False).flatten().tolist()
    active_ids.sort(key=lambda index: abs(float(codes[index])), reverse=True)
    whitened = None
    if normalization is not None:
        whitened, white_ids = normalization.rank(codes, top_feature_num or codes.numel())
        if mode != "non_whiten":
            top_ids = white_ids
            active_ids.sort(key=lambda index: (-abs(float(whitened[index])), index))
    from tasks.capture.selection import candidate_ids, output_count, prepare_filter, settings

    selection = settings(cfg)
    exhaustive = mode == "I"
    modern = selection["explicit"] or mode == "g_s" or exhaustive
    candidates = top_ids
    if modern:
        sar_context = prepare_filter(cfg) if sar_context is None else sar_context
        candidates = candidate_ids(cfg, pair, codes, normalization, sar_context, options)
        top_ids = candidates[: output_count(cfg, len(candidates))]
    sar = None
    single = None
    if exhaustive or mode == "g_s":
        from tasks.capture.measure_behavior import measure_behavior

        measured = measure_behavior(
            cfg, pair, snapshot, prompt, sample=sample, feature_ids=candidates, context=sar_context
        )
        if mode == "g_s":
            single = measured
            top_ids = [item["feature_id"] for item in single["top_features"]]
        else:
            sar = measured
            top_ids = list(
                dict.fromkeys(
                    index for item in sar["pairwise"]["top_pairs"] for index in item["feature_ids"]
                )
            )
    elif mode == "sar":
        from tasks.capture.sar_mode import measure_sar, prepare_sar
        from tasks.capture.sar_filter import select_features

        sar_context = prepare_sar(cfg) if sar_context is None else sar_context
        top_ids = (
            candidates
            if modern
            else select_features(pair, codes, normalization, top_feature_num, sar_context, options)
        )

        sar = measure_sar(
            cfg,
            pair,
            snapshot,
            prompt,
            sample=sample,
            context=sar_context,
            feature_ids=top_ids,
            normalization=normalization,
        )
        top_ids = sar["feature_ids"]
        candidates = top_ids
        if modern:
            top_ids = (
                list(
                    dict.fromkeys(
                        index
                        for item in sar.get("pairwise", {}).get("top_pairs", [])
                        for index in item["feature_ids"]
                    )
                )
                or candidates[: output_count(cfg, len(candidates))]
            )
    annotation, descriptions = annotate_features(pair, active_ids + top_ids, options)

    def feature(index):
        feature = {
            "feature_id": index,
            "activation": float(codes[index]),
            "neuronpedia": descriptions[index],
        }
        if whitened is not None:
            feature.update(
                zscore=float(whitened[index]),
                mean=float(normalization.mean[index]),
                std=float(normalization.std[index]),
                zero_native_activation=bool(codes[index] == 0),
                whiten_eligible=bool(normalization.std[index] > 0),
            )
        if single is not None and index in candidates:
            score = next(item for item in single["candidates"] if item["feature_id"] == index)
            feature.update(g=score["g"], delta_g=score["delta_g"])
        return feature

    if sar is not None and "pairwise" in sar:
        for item in sar["pairwise"]["top_pairs"]:
            item["features"] = [feature(i) for i in item["feature_ids"]]
            item["interpretation"] = (
                "固定其他候选背景下的实测四格交互g11−g10−g01+g00；每对取背景中的最大值。"
                if exhaustive
                else "二阶拟合中，两特征同时保留产生的额外贡献；正值提高目标行为评分，负值降低评分。"
                "这是组合干预的拟合说明，特征本身的公开解释来自 Neuronpedia。"
            )

    work = [{"status": "completed"}]
    if annotation["status"] != "disabled":
        work.extend(
            {
                "status": "completed" if value["status"] == "available" else "blocked",
                **(
                    {}
                    if value["status"] == "available"
                    else {
                        "reason": f"Neuronpedia feature {index}: {value.get('reason', 'No published explanation in the selected exports')}"
                    }
                ),
            }
            for index in top_ids
            for value in [descriptions[index]]
        )
    identity = input_identity(pair, snapshot, prompt, cfg)
    if identity is not None:
        identity["execution"]["capture_mode"] = mode
        if normalization is not None:
            identity["normalization"] = normalization.identity
        if sar is not None:
            identity["sar"] = sar["identity"]
        if single is not None:
            identity["g_s"] = single["identity"]
        if modern:
            identity["selection"] = selection
    if mode == "whiten" and not top_ids:
        work.append({"status": "undefined", "reason": "All features have zero population variance"})
    if sar is not None and sar["status"] == "undefined":
        work.append({"status": "undefined", "reason": "SAR coalition behavior has zero variance"})
    filter_artifacts = [
        sar_context[key]["artifact"]
        for key in ("feature_filter_identity", "cohort_filter_identity")
        if sar_context and sar_context.get(key, {}).get("enabled")
    ]
    return {
        **({"g_s": single} if single is not None else {}),
        **(
            {"required_artifacts": filter_artifacts + ([single["artifact"]] if single else [])}
            if modern and sar is None
            else {}
        ),
        **(
            {
                "sar": sar,
                "required_artifacts": [sar["artifact"]]
                + (
                    [sar["identity"]["feature_filter"]["artifact"]]
                    if sar["identity"]["feature_filter"].get("enabled")
                    else []
                )
                + (
                    [sar["identity"]["cohort_filter"]["artifact"]]
                    if sar["identity"]["cohort_filter"].get("enabled")
                    else []
                ),
            }
            if sar is not None
            else {}
        ),
        "mode": mode,
        "ranking": "signed_exhaustive_interaction_descending"
        if exhaustive
        else "descending_deleted_answer_mean_logprob"
        if single is not None
        else "absolute_zscore"
        if mode != "non_whiten" and whitened is not None
        else "absolute_native_activation",
        "selection": selection,
        "candidate_feature_ids": candidates,
        "feature_filter": sar_context.get("feature_filter_identity") if sar_context else None,
        "cohort_filter": sar_context.get("cohort_filter_identity") if sar_context else None,
        "ranking_status": "undefined" if mode == "whiten" and not top_ids else "defined",
        "normalization": normalization.identity if normalization is not None else None,
        "capture_format_version": 2,
        "status": "executed",
        "operation": "capture",
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "layer": getattr(pair.identity, "layer", None),
        "site": pair.site.hook_name,
        "prompt_sha256": _fingerprint_prompt(prompt),
        "input_identity": identity,
        "feature_count": int(codes.numel()),
        "active_count": int(torch.count_nonzero(codes).item()),
        "top_features": [feature(index) for index in top_ids],
        "active_features": [feature(index) for index in active_ids],
        "neuronpedia": annotation,
        "completion": completion_summary(work, unit="work_item"),
        "reconstruction_l2": float(
            torch.linalg.vector_norm(snapshot.reconstruction.float()).item()
        ),
    }


def execute(cfg: Any) -> dict[str, Any]:
    mode = capture_mode(cfg)
    scope = cfg.task.get("output_scope", "last_token")
    if scope not in {"last_token", "all_tokens"}:
        raise ValueError("output_scope must be last_token or all_tokens")
    if scope == "all_tokens" and mode != "non_whiten":
        raise ValueError("all_tokens currently records native activations; choose mode=non_whiten")
    if scope == "all_tokens" and cfg.task.get("filter", "all") != "all":
        raise ValueError("all_tokens records native sparse codes; use task.filter=all")
    if scope == "all_tokens":
        from tasks.sae_runtime import validate_fields

        validate_fields(cfg, integers=(("max_tokens", 1), ("max_active_per_token", 1)))
    capture_top_feature_num(cfg)
    neuronpedia_settings(cfg)
    context = None
    if mode == "sar":
        from tasks.capture.sar_mode import prepare_sar, behavior

        if str(cfg.task.get("prompt", "") or "").strip():
            behavior(cfg)
        context = prepare_sar(cfg)
    else:
        from tasks.capture.selection import prepare_filter

        context = prepare_filter(cfg)
    from tasks.capture.selection import needs_normalization

    needs_whiten = needs_normalization(cfg, context)
    if (
        mode in {"non_whiten", "sar", "g_s", "I"}
        and not needs_whiten
        and str(cfg.task.get("prompt", "") or "").strip()
    ):
        return _capture(cfg, **({"sar_context": context} if context is not None else {}))
    from tasks.capture.batch import capture_dataset

    return capture_dataset(cfg, **({"sar_context": context} if context is not None else {}))


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from tasks.sae_runtime import validate_sae_selection

    mode = capture_mode(cfg)
    capture_top_feature_num(cfg)
    neuronpedia_settings(cfg)
    if mode == "sar":
        from tasks.capture.sar_mode import sar_settings, behavior

        sar_settings(cfg)
        if str(cfg.task.get("prompt", "") or "").strip():
            behavior(cfg)
    if mode in {"g_s", "I"}:
        from tasks.capture.sar_mode import behavior

        if cfg.task.get("sar", {}).get("readout", "answer_mean_logprob") != "answer_mean_logprob":
            raise ValueError("g_s and I require answer_mean_logprob")
        if str(cfg.task.get("prompt", "") or "").strip():
            behavior(cfg)
    from tasks.capture.selection import prepare_filter, needs_normalization

    if (
        mode == "whiten"
        or needs_normalization(cfg, prepare_filter(cfg))
        or (mode == "sar" and cfg.task.sar.get("feature_selection") == "whiten")
        or not str(cfg.task.get("prompt", "") or "").strip()
    ):
        from saebase.datasets.selection import validate_selection

        validate_selection(cfg.dataset)
    validate_sae_selection(cfg)
