"""A real prompt-to-sparse-token-feature capture for the local SAEBase worker.

This opt-in Python API records the complete nonzero codes at each token.
It is distinct from model generation, feature steering and a public inference
service. The returned record is private until reviewed for redistribution.
"""

from __future__ import annotations

from saebase.runtime.configuration.task_options import task_value

import hashlib
from typing import Any

import torch

from saebase.api import load_registered
from saebase.store import fingerprint


@torch.no_grad()
def capture_prompt_trace(
    *,
    sae_key: str,
    layer: int,
    prompt: str,
    device: str = "cpu",
    sample_id: str = "local-prompt",
    prompt_group_id: str = "local-group",
    split: str = "discovery",
    max_tokens: int = 128,
    max_active_per_token: int = 256,
    pair: Any | None = None,
) -> dict[str, Any]:
    """Capture one real prompt from a registered TransformerLens/SAE pair.

    Returns complete sparse token rows within the configured budgets. Does not publish or invoke
    an external LLM API. For private texts, keep output off public GitHub/HF.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be nonempty text")
    if not isinstance(split, str) or not split or not sample_id or not prompt_group_id:
        raise ValueError("Explicit sample/group/split identity required")
    if (
        type(max_tokens) is not int
        or max_tokens < 1
        or type(max_active_per_token) is not int
        or max_active_per_token < 1
    ):
        raise ValueError("Token and sparse-row budgets must be positive integers")
    if pair is None:
        pair = load_registered(sae_key=sae_key, layer=layer, device=device)
    elif pair.identity.sae_key != sae_key or pair.identity.layer != layer:
        raise ValueError("Selected loaded model/SAE does not match requested prompt site")
    model, sae, site = pair.model, pair.sae, pair.site
    token_ids = model.tokens(prompt=prompt)
    n_tokens = int(token_ids.shape[1])
    if not 1 <= n_tokens <= max_tokens:
        raise ValueError("Prompt token length exceeds bounded capture budget")
    raw_model = model.model
    activation = model.capture_sequence(tokens=token_ids, site=site)
    if activation.shape != (n_tokens, pair.identity.d_in):
        raise ValueError("Exact token/SAE input dimensions mismatch")
    codes = sae.encode(activation)
    if (
        not isinstance(codes, torch.Tensor)
        or codes.ndim != 2
        or tuple(codes.shape) != (n_tokens, pair.identity.d_sae)
        or not bool(torch.isfinite(codes).all())
    ):
        raise ValueError("Native SAE returned invalid per-token feature codes")
    tokenizer = getattr(raw_model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "get_vocab", None)):
        raise ValueError("A revision-verifiable tokenizer is required")
    token_state = {
        "vocab": tokenizer.get_vocab(),
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    tokenizer_sha = fingerprint(token_state)
    ids = token_ids[0].detach().cpu().tolist()
    if callable(getattr(tokenizer, "convert_ids_to_tokens", None)):
        texts = tokenizer.convert_ids_to_tokens(ids)
    else:
        texts = [tokenizer.decode([token], skip_special_tokens=False) for token in ids]
    if not isinstance(texts, list) or len(texts) != n_tokens:
        raise ValueError("Tokenizer failed to return one string per token ID")
    tokens: list[dict[str, Any]] = []
    for position, (token_id, token_text) in enumerate(zip(ids, texts)):
        active = torch.nonzero(codes[position], as_tuple=True)[0]
        if active.numel() > max_active_per_token:
            raise ValueError("Feature activity exceeds complete sparse row budget")
        features = [
            {"feature_id": int(fid), "activation": float(codes[position, fid].float().item())}
            for fid in active.tolist()
        ]
        tokens.append(
            {
                "position": position,
                "token_id": int(token_id),
                "text": str(token_text),
                "features": sorted(
                    features, key=lambda v: (-abs(v["activation"]), v["feature_id"])
                ),
            }
        )
    from saebase.runtime.sae_operations import pair_state_fingerprints

    model_sha, sae_sha = pair_state_fingerprints(pair)
    reconstruction = sae.decode(codes)[-1]
    return {
        "reconstruction_l2": float(torch.linalg.vector_norm(reconstruction.float()).item()),
        "site_id": f"{pair.identity.model_key}:{pair.identity.sae_key}:L{layer}",
        "model_sha256": model_sha,
        "sae_sha256": sae_sha,
        "source_type": "synthetic_test"
        if model.backend == "synthetic"
        or str(getattr(pair.identity, "checkpoint", "")).startswith("synthetic://")
        else "real_checkpoint",
        "sample_id": sample_id,
        "prompt_group_id": prompt_group_id,
        "split": split,
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tokenizer_sha256": tokenizer_sha,
        "model_processing": str(getattr(raw_model, "_saebase_weight_processing", "unspecified")),
        "complete_sparse_codes": True,
        "tokens": tokens,
        "public_text_reviewed": False,
        "rights_reviewed": False,
    }


def capture_prompt_result(cfg, pair, prompt, sample=None):
    """Capture all tokens through the ordinary capture Task result protocol."""
    from tasks.capture.operation import capture_top_feature_num, neuronpedia_settings
    from saebase.api.neuronpedia import annotate_features
    from saebase.results import completion_summary
    import torch

    trace = capture_prompt_trace(
        sae_key=pair.identity.sae_key,
        layer=pair.identity.layer,
        prompt=prompt,
        pair=pair,
        sample_id=(sample or {}).get(
            "sample_id", "prompt-" + hashlib.sha256(prompt.encode()).hexdigest()[:16]
        ),
        split=(sample or {}).get("split") or "discovery",
        prompt_group_id=(sample or {}).get("prompt_group_id")
        or (sample or {}).get("sample_id", "local-group"),
        max_tokens=int(task_value(cfg.task, "max_tokens", 128)),
        max_active_per_token=int(task_value(cfg.task, "max_active_per_token", 256)),
    )
    ids = {feature["feature_id"] for token in trace["tokens"] for feature in token["features"]}
    codes = torch.zeros(pair.identity.d_sae, dtype=torch.float32)
    for feature in trace["tokens"][-1]["features"]:
        codes[feature["feature_id"]] = feature["activation"]
    top_ids = torch.topk(
        codes.abs(), min(capture_top_feature_num(cfg) or codes.numel(), codes.numel())
    ).indices.tolist()
    info, descriptions = annotate_features(
        pair, sorted(ids | set(top_ids)), neuronpedia_settings(cfg)
    )
    for token in trace["tokens"]:
        for feature in token["features"]:
            feature["neuronpedia"] = descriptions[feature["feature_id"]]
    top = [
        {"feature_id": i, "activation": float(codes[i]), "neuronpedia": descriptions[i]}
        for i in top_ids
    ]
    work = [{"status": "completed"}]
    if info["status"] != "disabled":
        work.extend(
            {
                "status": "completed" if row["status"] == "available" else "blocked",
                **(
                    {"reason": row.get("reason", "No published explanation for this feature")}
                    if row["status"] != "available"
                    else {}
                ),
            }
            for row in descriptions.values()
        )
    input_identity = {
        "schema": "saebase.task_input.v1",
        "task": "capture",
        "protocol_id": "saebase.capture.all_tokens.v1",
        "model_sha256": trace["model_sha256"],
        "sae_sha256": trace["sae_sha256"],
        "data_sha256": fingerprint(
            {
                "tokens": [[row["token_id"] for row in trace["tokens"]]],
                "prompt_sha256": trace["prompt_sha256"],
            }
        ),
        "site": {
            "layer": pair.identity.layer,
            "hook_name": pair.site.hook_name,
            "selection": "all_prompt_tokens",
        },
        "execution": {
            "capture_mode": "non_whiten",
            "output_scope": "all_tokens",
            "top_feature_num": capture_top_feature_num(cfg),
        },
    }
    return {
        "input_identity": input_identity,
        **trace,
        "status": "executed",
        "operation": "capture",
        "capture_format_version": 2,
        "mode": "non_whiten",
        "output_scope": "all_tokens",
        "ranking": "absolute_native_activation",
        "model_key": pair.identity.model_key,
        "sae_key": pair.identity.sae_key,
        "layer": pair.identity.layer,
        "site": pair.site.hook_name,
        "feature_count": pair.identity.d_sae,
        "active_count": len(trace["tokens"][-1]["features"]),
        "active_features": trace["tokens"][-1]["features"],
        "top_features": top,
        "neuronpedia": info,
        "completion": completion_summary(work, unit="work_item"),
    }
