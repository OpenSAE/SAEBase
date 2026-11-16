"""Capture-owned SAR configuration, labeled behavior and auditable measurement.

Reuses capture.sar and the shared explicit paper loader; no estimator is copied.
Dataset owns prompts/labels, capture owns coalition measurements and artifacts.
"""

from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path

import numpy as np

from saebase.api.paper_sources import load_paper_module
from tasks.sae_runtime import input_path

PROTOCOL = "saebase.capture.sar.affine_residual_preserving.v1"


def sar_settings(cfg):
    """Check the full protocol without importing source or loading weights."""
    from tasks.capture.operation import capture_top_feature_num

    raw = cfg.task.get("sar", {})
    if not hasattr(raw, "get"):
        raise ValueError("task.sar must contain SAR settings")
    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("task.sar.source must name the trusted original sar.py")
    source = input_path(source)
    if source.is_dir():
        source = source / "sar.py"
    if source.name != "sar.py" or not source.is_file():
        raise ValueError(f"task.sar.source is not an existing sar.py: {source}")
    from tasks.capture.selection import settings

    filtered = settings(cfg)["filter"]
    K = filtered if type(filtered) is int else capture_top_feature_num(cfg)
    if "filter" in cfg.task and filtered == "all":
        K = None
    selection = raw.get("feature_selection", "native_activation")
    if selection not in {"native_activation", "whiten"}:
        raise ValueError("task.sar.feature_selection must be native_activation or whiten")
    top_pairs = raw.get("top_pairs", 0)
    if type(top_pairs) is not int or top_pairs < 0:
        raise ValueError("task.sar.top_pairs must be a nonnegative integer")
    pair_ranking = raw.get("pair_ranking", "absolute")
    if pair_ranking not in {"absolute", "signed"}:
        raise ValueError("task.sar.pair_ranking must be absolute or signed")
    answer_backend = raw.get("answer_backend", "full_sequence")
    if answer_backend not in {"full_sequence", "auto", "full_context_tail"}:
        raise ValueError("task.sar.answer_backend must be full_sequence, full_context_tail or auto")
    if K is not None and K < 2:
        raise ValueError("SAR requires task.top_feature_num >= 2")
    M, n_fit = raw.get("M", 384), raw.get("n_fit", 256)
    if type(M) is not int or type(n_fit) is not int or not 2 <= n_fit <= M - 2:
        raise ValueError("task.sar requires integer M/n_fit and at least two fit/held rows")
    parameters = 1 + K + K * (K - 1) // 2 if K is not None else None
    if parameters is not None and n_fit < parameters:
        raise ValueError(f"task.sar.n_fit must cover {parameters} pairwise coefficients")
    p = raw.get("p", 0.5)
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("task.sar.p must be finite and between zero and one")
    if raw.get("readout", "logit_difference") not in {"logit_difference", "answer_mean_logprob"}:
        raise ValueError("task.sar.readout must be logit_difference or answer_mean_logprob")
    position = raw.get("readout_position", -1)
    if type(position) is not int:
        raise ValueError("task.sar.readout_position must be an integer")
    affine = raw.get("require_affine_decode", True)
    if type(affine) is not bool:
        raise ValueError("task.sar.require_affine_decode must be boolean")
    seed = cfg.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("SAR requires a nonnegative integer seed")
    from tasks.capture.sar_filter import filter_settings, cohort_filter_settings

    feature_filter = filter_settings(raw.get("feature_filter", {}))
    cohort_filter = cohort_filter_settings(raw.get("cohort_filter", {}))
    if cohort_filter["enabled"] and selection != "whiten":
        raise ValueError("SAR cohort filtering requires task.sar.feature_selection=whiten")
    for key in ("target_token_id", "reference_token_id"):
        value = raw.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"task.sar.{key} must be a nonnegative integer")
    return {
        "source": str(source),
        "K": K,
        "M": M,
        "n_fit": n_fit,
        "p": float(p),
        "seed": seed,
        "readout_position": position,
        "require_affine_decode": affine,
        "feature_selection": selection,
        "top_pairs": top_pairs,
        "answer_backend": answer_backend,
        "feature_filter": feature_filter,
        "cohort_filter": cohort_filter,
        "pair_ranking": pair_ranking,
    }


def behavior(cfg, sample=None):
    """Use explicit per-record token IDs; never infer a target from model output."""
    row = sample or {}
    raw = cfg.task.get("sar", {})
    default_readout = (
        "answer_mean_logprob" if cfg.task.get("mode") in {"g_s", "I"} else "logit_difference"
    )
    if raw.get("readout", default_readout) == "answer_mean_logprob":
        from saebase.datasets.answers import reference_answer

        if any(raw.get(key) is not None for key in ("target_token_id", "reference_token_id")):
            raise ValueError(
                "answer_mean_logprob uses a full reference answer; omit target/reference token IDs"
            )
        if raw.get("readout_position", -1) != -1:
            raise ValueError(
                "answer_mean_logprob scores the complete answer, not a readout_position"
            )
        if raw.get("answer") is not None:
            row = {**row, "reference_answer": raw["answer"]}
        reference = reference_answer(row, str(cfg.get("dataset", {}).get("name", "explicit")))
        if raw.get("answer") is not None:
            reference["source_field"] = "task.sar.answer"
        return {
            "name": "answer_mean_logprob",
            "reference": reference,
            "readout_position": -1,
            "feature_selection_uses_answer": False,
            "aggregation": "mean teacher-forced answer-token log probability; no EOS",
        }
    tokenization = row.get("tokenization", {})
    if not hasattr(tokenization, "get"):
        raise ValueError("SAR sample tokenization must contain explicit token IDs")
    values = {}
    for key, token_key, legacy in (
        ("target_token_id", "correct_ids", "tok_io"),
        ("reference_token_id", "distractor_ids", "tok_s"),
    ):
        candidates = [row.get(key), row.get(legacy), raw.get(key)]
        ids = tokenization.get(token_key)
        if ids is not None:
            if not isinstance(ids, (list, tuple)) or len(ids) != 1:
                raise ValueError(f"SAR requires one token in sample.tokenization.{token_key}")
            candidates.append(ids[0])
        candidates = [v for v in candidates if v is not None]
        if not candidates or any(type(v) is not int or v < 0 for v in candidates):
            raise ValueError(f"SAR requires explicit nonnegative {key} for every sample")
        if len(set(candidates)) != 1:
            raise ValueError(f"Conflicting SAR {key} declarations")
        values[key] = candidates[0]
    if values["target_token_id"] == values["reference_token_id"]:
        raise ValueError("SAR target and reference token IDs must differ")
    values["readout_position"] = row.get("readout_position", raw.get("readout_position", -1))
    if type(values["readout_position"]) is not int:
        raise ValueError("SAR sample readout_position must be an integer")
    return {"name": "logit_difference", **values}


def prepare_sar(cfg):
    """Load the explicitly selected original estimator once per capture run."""
    settings = sar_settings(cfg)
    if cfg.task.get("filter") == "all":
        settings["feature_filter"]["enabled"] = False
        settings["cohort_filter"]["enabled"] = False
    source = Path(settings["source"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    paper = load_paper_module(source, paper="sar")
    if settings["top_pairs"] and not callable(getattr(paper, "design", None)):
        raise ValueError("Pair reporting requires the original SAR design() function")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("SAR estimator source changed during import")
    return {"settings": settings, "paper": paper, "source_sha256": digest}


def measure_sar(
    cfg, pair, snapshot, prompt, *, sample=None, context=None, feature_ids=None, normalization=None
):
    """Measure one captured sample and save raw arrays under its run directory."""
    from tasks.capture.sar import evaluate_sar

    readout = behavior(cfg, sample)
    context = prepare_sar(cfg) if context is None else context
    settings = context["settings"]
    k = len(feature_ids) if feature_ids is not None else settings["K"] or snapshot.codes.numel()
    required = 1 + k + k * (k - 1) // 2
    if settings["n_fit"] < required:
        raise ValueError(f"Surrogate SAR needs at least {required} fitting rows for {k} candidates")
    from tasks.capture.selection import output_count

    top_pairs = (
        output_count(cfg, k * (k - 1) // 2) if "filter" in cfg.task else settings["top_pairs"]
    )
    source = Path(settings["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != context["source_sha256"]:
        raise ValueError("SAR estimator source changed after run preparation")

    answer_tokens = None
    readout_activation = None
    activation_readout = None
    if readout["name"] == "answer_mean_logprob":
        from tasks.capture.answer import prepare_answer_readout

        activation_readout, answer_ids, readout_activation = prepare_answer_readout(
            pair,
            snapshot,
            prompt,
            readout,
            feature_ids=feature_ids
            or snapshot.codes.abs().argsort(descending=True)[: settings["K"]].tolist(),
            answer_backend=settings["answer_backend"],
        )
        import torch

        answer_tokens = torch.tensor([answer_ids], device=snapshot.tokens.device)

    def score(logits):
        target, reference = readout["target_token_id"], readout["reference_token_id"]
        if max(target, reference) >= logits.numel():
            raise ValueError("SAR behavior token IDs exceed the model vocabulary")
        return logits[target] - logits[reference]

    result = evaluate_sar(
        pair.model,
        pair.sae,
        site=pair.site,
        paper=context["paper"],
        readout=score,
        tokens=snapshot.tokens,
        snapshot=snapshot,
        identity=pair.identity,
        K=k,
        M=settings["M"],
        n_fit=settings["n_fit"],
        seed=settings["seed"],
        p=settings["p"],
        readout_position=readout["readout_position"],
        require_affine_decode=settings["require_affine_decode"],
        activation_readout=activation_readout,
        feature_ids=feature_ids,
        top_pairs=top_pairs,
        pair_ranking=settings["pair_ranking"],
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != context["source_sha256"]:
        raise ValueError("SAR estimator source changed during measurement")
    protocol = (
        PROTOCOL
        if settings["require_affine_decode"]
        else ("saebase.capture.sar.decoder_mediated_residual_preserving.v1")
    )
    identity = {
        "protocol_id": protocol,
        "source": settings["source"],
        "source_sha256": context["source_sha256"],
        "selection": "absolute_zscore; stable feature ID ties"
        if settings["feature_selection"] == "whiten"
        else "absolute_native_activation; stable feature ID ties",
        **({"normalization": normalization.identity} if normalization is not None else {}),
        "affine_diagnostic": "decoder additivity before model FP16/BF16 patch rounding",
        **{k: settings[k] for k in ("K", "M", "n_fit", "p", "seed", "require_affine_decode")},
        "K": k,
        "behavior": readout,
        "feature_ids": result.feature_ids.tolist(),
        "feature_filter": context.get("feature_filter_identity", {"enabled": False}),
        "cohort_filter": context.get("cohort_filter_identity", {"enabled": False}),
        "pair_ranking": settings["pair_ranking"],
    }
    key = hashlib.sha256(
        json.dumps(
            {
                "sample_id": (sample or {}).get("sample_id"),
                "prompt": prompt,
                "tokens": result.source_tokens.tolist(),
                "identity": identity,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    directory = Path.cwd() / "reports/sar"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".npz")
    if path.exists():
        raise FileExistsError(f"SAR measurement already exists: {path}")
    temporary = path.with_suffix(".tmp.npz")

    def array(tensor):
        value = tensor.detach().cpu()
        # NumPy cannot store BF16; widening to FP32 is exact for these values.
        return value.float().numpy() if str(value.dtype) == "torch.bfloat16" else value.numpy()

    np.savez_compressed(
        temporary,
        feature_ids=result.feature_ids,
        masks=result.masks,
        fit_indices=result.fit_indices,
        held_indices=result.held_indices,
        responses=result.responses,
        full_response=np.array(result.full_response),
        source_tokens=result.source_tokens,
        activation=array(snapshot.activation),
        **(
            {"readout_activation": array(readout_activation)}
            if readout_activation is not None
            else {}
        ),
        selected_codes=array(snapshot.codes)[result.feature_ids],
        **(
            {
                "pair_coefficients": result.pair_coefficients,
                "pair_feature_ids": result.pair_feature_ids,
            }
            if result.pair_coefficients is not None
            else {}
        ),
        **(
            {"answer_token_ids": np.asarray(answer_ids, dtype=np.int64)}
            if answer_tokens is not None
            else {}
        ),
    )
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    undefined = [k for k, v in result.metrics.items() if not np.isfinite(v)]
    return {
        **result.summary(),
        "sae_identity": pair.identity.to_dict(),
        "identity": identity,
        "status": "undefined" if result.metrics.get("var") == 0 else "completed",
        "undefined_metrics": undefined,
        "artifact": {
            "path": str(path.relative_to(Path.cwd())),
            "bytes": path.stat().st_size,
            "sha256": digest,
        },
    }
