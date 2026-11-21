"""Measure canonical-answer single ablations or exhaustive conditional pairs."""

import hashlib
import json
from saebase.runtime.configuration.task_options import task_value
from tasks.capture.configuration import capture_section
from pathlib import Path

import numpy as np

from saebase.api import edit_features, UnsupportedCapability
from tasks.capture.answer import prepare_answer_readout
from tasks.capture.behavior import choose_pair, rank_singles
from tasks.capture.selection import active_whiten, g_s_candidate_count, output_count, settings


def score_masks(
    pair, snapshot, ids, masks, score, answer_ids, readout_activation, full, performance
):
    """Use the same native edit and cached/batched scoring for screening and I."""
    from saebase.runtime.performance import stage

    optimization = None
    if performance["enabled"]:
        from tasks.capture.performance import evaluate_edits, prepare_activations

        with stage("feature_edits"):
            activations, inverse, edit_info = prepare_activations(pair.sae, snapshot, ids, masks)
        try:
            values, optimization = evaluate_edits(
                pair, snapshot, activations, answer_ids, readout_activation, score, performance
            )
        except UnsupportedCapability as exc:
            values = np.asarray([float(score(value)) for value in activations])
            optimization = {"selected": False, "execution": "original_scalar", "reason": str(exc)}
        responses = values[inverse]
        optimization.update(edit_info)
    else:
        responses = np.empty(len(masks), dtype=np.float64)
        for index, mask in enumerate(masks):
            remove = {int(fid): 0.0 for fid, retained in zip(ids, mask) if not retained}
            responses[index] = (
                float(score(edit_features(pair.sae, snapshot, replacements=remove).activation))
                if remove
                else full
            )
    if not np.isfinite(responses).all() or np.any(responses > 0):
        raise ValueError(
            "Canonical answer scores must be finite, nonpositive mean log probabilities"
        )
    return responses, optimization


def measure_behavior(cfg, pair, snapshot, prompt, *, sample, feature_ids, context):
    from tasks.capture.sar_mode import behavior

    mode = task_value(cfg.task, "mode")
    from saebase.runtime.performance import settings as performance_settings, stage

    performance = performance_settings(cfg)
    ids = list(feature_ids)
    readout = behavior(cfg, sample)
    if readout["name"] != "answer_mean_logprob":
        raise ValueError("g_s and I require the complete-answer log probability readout")
    score, answer_ids, readout_activation = prepare_answer_readout(
        pair,
        snapshot,
        prompt,
        readout,
        feature_ids=ids,
        answer_backend=capture_section(cfg.task, "answer").get("backend", "full_sequence"),
    )
    full = float(score(None))
    screening = None
    screen_arrays = {}
    from tasks.capture.selection import i_feature_selection, i_g_s_shortfall

    if i_feature_selection(cfg) == "g_s":
        pool = list(ids)
        single_masks = np.ones((len(pool) + 1, len(pool)), dtype=bool)
        for index in range(len(pool)):
            single_masks[index + 1, index] = False
        with stage("i_g_s_screening"):
            single_scores, screen_performance = score_masks(
                pair,
                snapshot,
                pool,
                single_masks,
                score,
                answer_ids,
                readout_activation,
                full,
                performance,
            )
        ranking = rank_singles(pool, full, single_scores[1:])
        requested = settings(cfg)["filter"]
        ids = [row["feature_id"] for row in ranking[:requested]]
        readout["feature_selection_uses_answer"] = True
        screening = {
            "method": "g_s",
            "active_only": True,
            "shortfall_policy": i_g_s_shortfall(cfg),
            "pool_filtering": dict(context.get("i_g_s_pool", {})) if context else {},
            "uses_reference_answer": True,
            "background": "all other native features unchanged; only one feature zeroed",
            "requested_count": requested,
            "eligible_pool_count": len(pool),
            "selected_count": len(ids),
            "selected_feature_ids": ids,
            "full_response": full,
            "ranking": ranking,
            "tie_break": "original pool order: absolute native activation then feature ID",
            "performance": screen_performance,
        }
        screen_arrays = {
            "screen_feature_ids": np.asarray(pool, dtype=np.int64),
            "screen_deleted_responses": single_scores[1:],
        }
    if mode == "I":
        maximum = capture_section(cfg.task, "i").get("max_exhaustive_features", 16)
        if len(ids) > maximum:
            raise ValueError(
                f"mode=I has {len(ids)} candidates; maximum is {maximum}. "
                "Lower task.capture.selection.candidate_count or increase task.capture.i.max_exhaustive_features"
            )
        if len(ids) < 2:
            raise ValueError("mode=I requires at least two candidate features")
    if mode == "g_s":
        masks = np.ones((len(ids) + 1, len(ids)), dtype=bool)
        for index in range(len(ids)):
            masks[index + 1, index] = False
    else:
        states = np.arange(1 << len(ids), dtype=np.int64)
        masks = ((states[:, None] >> np.arange(len(ids))) & 1).astype(bool)
    responses, optimization = score_masks(
        pair, snapshot, ids, masks, score, answer_ids, readout_activation, full, performance
    )
    identity = {
        "protocol_id": f"saebase.capture.{mode}.answer_behavior.v1",
        "algorithm": "single_ablation" if mode == "g_s" else "exhaustive",
        "behavior": readout,
        "feature_ids": ids,
        "selection": settings(cfg),
        "feature_filter": context.get("feature_filter_identity", {"enabled": False})
        if context
        else {"enabled": False},
        "cohort_filter": context.get("cohort_filter_identity", {"enabled": False})
        if context
        else {"enabled": False},
        "activation_dtype": str(snapshot.activation.dtype),
    }
    from models.numerics import identity as numerical_identity

    identity["numerics"] = numerical_identity()
    if mode == "g_s" and g_s_candidate_count(cfg) is not None:
        identity["preselection"] = {
            "method": "absolute_zscore",
            "requested_count": g_s_candidate_count(cfg),
            "selected_count": len(ids),
            "zero_activation_policy": "retain eligible zero activations, as in Whiten",
        }
    if mode == "I":
        identity["i_feature_selection"] = (
            "whiten"
            if active_whiten(cfg.task)
            else i_feature_selection(cfg)
            or capture_section(cfg.task, "selection").get("feature_selection", "whiten")
        )
    if screening is not None:
        identity["preselection"] = {
            k: v
            for k, v in screening.items()
            if k not in {"ranking", "performance", "full_response"}
        }
    key = hashlib.sha256(
        json.dumps({"prompt": prompt, "identity": identity}, sort_keys=True).encode()
    ).hexdigest()
    directory = Path.cwd() / "reports" / mode
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    if path.exists():
        raise FileExistsError(f"Capture measurement already exists: {path}")
    np.savez_compressed(
        path,
        feature_ids=np.array(ids),
        masks=masks,
        responses=responses,
        full_response=np.array(full),
        answer_token_ids=np.array(answer_ids),
        source_tokens=snapshot.tokens.detach().cpu().numpy(),
        activation=snapshot.activation.detach().float().cpu().numpy(),
        readout_activation=readout_activation.detach().float().cpu().numpy(),
        selected_codes=snapshot.codes.detach().float().cpu().numpy()[ids],
        **screen_arrays,
    )
    result = {
        "status": "completed",
        "identity": identity,
        "feature_ids": ids,
        "full_response": full,
        "measured_states": len(masks),
        **({"performance": optimization} if optimization is not None else {}),
        "artifact": {
            "path": str(path.relative_to(Path.cwd())),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    if screening is not None:
        result["preselection"] = screening
    if mode == "g_s":
        ranking = rank_singles(ids, full, responses[1:])
        result.update(
            ranking="descending_deleted_answer_mean_logprob",
            top_features=ranking[: output_count(cfg, len(ranking))],
            candidates=ranking,
            **({"preselection": identity["preselection"]} if "preselection" in identity else {}),
        )
    else:
        with stage("interaction_ranking"):
            ranked = choose_pair(ids, responses, len(answer_ids), return_ranking=True)
        pairs = [
            {**row, "feature_ids": row["pair_ids"], "coefficient": row["interaction"]}
            for row in ranked
        ]
        result["pairwise"] = {
            "ranking": "signed_exhaustive_interaction_descending",
            "basis": "largest g11-g10-g01+g00 over every fixed background per pair",
            "backgrounds_per_pair": 1 << (len(ids) - 2),
            "complete_four_state_backgrounds": len(pairs) * (1 << (len(ids) - 2)),
            "top_pairs": pairs[: output_count(cfg, len(pairs))],
        }
    return result
