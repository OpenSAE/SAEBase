"""Clean-prefix SAE edits that do not reuse edited history.

The shipped reference recomputes the whole clean prefix at every steered step.
Suffix replay can replace the *edited* forward with the later blocks only.
A persistent edited KV cache is a different scientific protocol and is not
implemented. Incremental clean KV for Qwen is BLOCKED: one DynamicCache would
have to keep clean keys for the early blocks and a separate edited suffix,
and that split is not verified across Transformers versions.
"""

from __future__ import annotations

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features
from .operators import apply_condition
from .specs import Condition

CLEAN_KV_STATUS = "BLOCKED"
CLEAN_KV_REASON = (
    "Incremental clean KV is blocked. The edit must see clean earlier positions "
    "and must not store the edited residual in the cache used for the next token. "
    "The current Qwen cache object does not expose a verified way to keep those "
    "two histories apart, so each step still recomputes the clean prefix."
)


@torch.no_grad()
def edited_logits_clean_prefix(
    model: ModelAdapter,
    sae: object,
    *,
    site: ActivationSite,
    tokens: torch.Tensor,
    condition: Condition,
) -> torch.Tensor:
    """Edited next-token logits for one clean prefix.

    The clean site activation comes from the original full forward. Only the
    edit runs from the cached block output when this model supports suffix replay.
    """
    if site.position != -1 and site.position != int(tokens.shape[1]) - 1:
        raise ValueError("clean-prefix edit is defined at the final token of the current prefix")
    current = ActivationSite(site.layer, site.hook_name, int(tokens.shape[1]) - 1)
    snapshot = capture_features(model, sae, site=current, tokens=tokens)
    patch = apply_condition(sae, snapshot, condition)
    backend = getattr(model.model, "logits_from_site_hidden", None)
    supported = getattr(model.model, "supports_suffix_replay", None)
    if model.backend == "hf_hooks" and callable(backend) and callable(supported) and supported():
        hidden = model.model.cache_site_hidden(tokens, site.layer)
        edited = hidden.clone()
        edited[0, -1] = patch.activation.to(device=edited.device, dtype=edited.dtype)
        return backend(tokens, site.layer, edited, readout_offset=0).detach().clone()
    return (
        model.patch_logits(
            tokens=tokens, site=current, activation=patch.activation, readout_position=-1
        )
        .detach()
        .clone()
    )
