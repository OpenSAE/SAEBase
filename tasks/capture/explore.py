"""A real prompt-to-sparse-token-feature capture for the local SAEBase worker.

This opt-in Python API records the complete nonzero codes at each token.
It is distinct from model generation, feature steering and a public inference
service. The returned record is private until reviewed for redistribution.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from saebase.api import load_registered
from saebase.store import fingerprint, sha256_state_dict


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

    Returns at most 128 complete sparse token rows. Does not publish or invoke
    an external LLM API. For private texts, keep output off public GitHub/HF.
    """
    if not isinstance(prompt, str) or not prompt or len(prompt) > 16000:
        raise ValueError("Prompt must be 1–16000 characters")
    if split not in {"discovery", "evaluation"} or not sample_id or not prompt_group_id:
        raise ValueError("Explicit sample/group/split identity required")
    if not 1 <= max_tokens <= 128 or not 1 <= max_active_per_token <= 256:
        raise ValueError("P2 small-capture budget exceeded")
    if pair is None:
        pair = load_registered(sae_key=sae_key, layer=layer, device=device)
    elif pair.identity.sae_key != sae_key or pair.identity.layer != layer:
        raise ValueError("Selected loaded model/SAE does not match requested prompt site")
    model, sae, site = pair.model, pair.sae, pair.site
    if model.backend != "transformer_lens":
        raise RuntimeError("Full token trace requires a supported cached TL hook backend")
    token_ids = model.tokens(prompt=prompt)
    n_tokens = int(token_ids.shape[1])
    if not 1 <= n_tokens <= max_tokens:
        raise ValueError("Prompt token length exceeds bounded capture budget")
    raw_model = model.model
    _, cache = raw_model.run_with_cache(token_ids, names_filter=site.hook_name)
    if site.hook_name not in cache:
        raise RuntimeError("Selected SAE hook did not fire")
    activation = cache[site.hook_name]
    if (
        not isinstance(activation, torch.Tensor)
        or activation.ndim != 3
        or activation.shape[0] != 1
        or activation.shape[1] != n_tokens
        or activation.shape[2] != pair.identity.d_in
    ):
        raise ValueError("Exact token/SAE input dimensions mismatch")
    activation = activation[0]
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
    return {
        "site_id": f"{pair.identity.model_key}:{pair.identity.sae_key}:L{layer}",
        "model_sha256": sha256_state_dict(raw_model),
        "sae_sha256": sha256_state_dict(sae),
        "source_type": "real_checkpoint",
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
