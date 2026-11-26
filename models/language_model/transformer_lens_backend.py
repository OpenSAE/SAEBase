"""Choose the actual TransformerLens model system without hiding weight processing.

Legacy HookedTransformer 3.x remains an explicit supported route. TL 4.x
replaces that class with TransformerBridge. Both expose the hook/cache API,
but their model coordinate systems must be recorded separately.
"""

from __future__ import annotations

from typing import Any

import torch


def _context_window(model_name: str, hf_model: Any | None) -> int:
    from models.loading_policy import execution_context_window

    if hf_model is not None:
        config = hf_model.config
    else:
        from transformers import AutoConfig

        try:
            from transformer_lens.supported_models import get_official_model_name
        except ImportError:
            from transformer_lens.loading_from_pretrained import get_official_model_name

        config = AutoConfig.from_pretrained(get_official_model_name(model_name))
    window = execution_context_window(config)
    if window is None:
        raise ValueError("Model configuration does not declare its native context window")
    return window


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

    TL4 raw mode uses boot_transformers alone, while the historical
    processed-weight route explicitly enables its compatibility mode.
    Legacy hook aliases are available in both modes; processing is not
    silently changed to make an incompatible SAE appear to work.
    """
    from models.hf_compat import import_transformer_lens

    tl = import_transformer_lens()
    window = _context_window(model_name, hf_model)
    # TransformerLens 4 exports a removal notice through module __getattr__.
    legacy = vars(tl).get("HookedTransformer")
    if legacy is not None:
        loader = legacy.from_pretrained_no_processing if no_processing else legacy.from_pretrained
        kwargs: dict[str, Any] = {"device": device, "n_ctx": window}
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
        if not no_processing:
            compatibility = getattr(model, "enable_compatibility_mode", None)
            if not callable(compatibility):
                raise RuntimeError("TransformerBridge lacks weight compatibility mode")
            # Full processed weights must be explicitly requested; historical
            # hook aliases are exposed even by a raw TransformerBridge.
            compatibility(no_processing=False)
        engine = "transformer_bridge_local"

    # The bridge exposes the same context setting used by tokenization and the
    # public answer-scoring API. Legacy attention buffers were sized above.
    model.cfg.n_ctx = window
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._saebase_transformerlens_engine = engine
    model._saebase_weight_processing = "raw_hf" if no_processing else "legacy_compatible"
    return model, engine
