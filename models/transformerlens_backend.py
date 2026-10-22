"""Choose the actual TransformerLens model system without hiding weight processing.

Legacy HookedTransformer 3.x remains an explicit supported route. TL 4.x
replaces that class with TransformerBridge. Both expose the hook/cache API,
but their model coordinate systems must be recorded separately.
"""

from __future__ import annotations

from typing import Any

import torch


def boot_transformerlens(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype | None,
    no_processing: bool,
    hf_model: Any | None = None,
    tokenizer: Any | None = None,
) -> tuple[Any, str]:
    """Load a local model; never fall back to another weight-processing mode.

    A bridge without compatibility mode uses raw HF weights and different hook
    names. Enabling compatibility with no_processing=True registers historical
    hook aliases but deliberately keeps raw HF parameters unchanged.
    """
    from models.hf_compat import import_transformer_lens

    tl = import_transformer_lens()
    legacy = getattr(tl, "HookedTransformer", None)
    if legacy is not None:
        loader = legacy.from_pretrained_no_processing if no_processing else legacy.from_pretrained
        kwargs: dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if hf_model is not None:
            kwargs["hf_model"] = hf_model
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        model = loader(model_name, **kwargs)
        engine = "hooked_transformer_legacy"
    else:
        from transformer_lens.model_bridge import TransformerBridge

        bridge_kwargs: dict[str, Any] = {
            "device": device,
            "dtype": dtype if dtype is not None else torch.float32,
        }
        if hf_model is not None:
            bridge_kwargs["hf_model"] = hf_model
        if tokenizer is not None:
            bridge_kwargs["tokenizer"] = tokenizer
        model = TransformerBridge.boot_transformers(model_name, **bridge_kwargs)
        compatibility = getattr(model, "enable_compatibility_mode", None)
        if not callable(compatibility):
            raise RuntimeError("TransformerBridge lacks legacy hook compatibility mode")
        # This registers old hook aliases in BOTH modes, but only folds/centers
        # weights if the original SAEBase profile called from_pretrained.
        compatibility(no_processing=no_processing)
        engine = "transformer_bridge_local"

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._saebase_transformerlens_engine = engine
    model._saebase_weight_processing = "raw_hf" if no_processing else "legacy_compatible"
    return model, engine
