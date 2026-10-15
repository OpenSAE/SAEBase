"""HF CausalLM residual-stream capture/patch via forward hooks (TL fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.hf_logits_backend import HFLogitsModel, _reject_quantized
from models.hooked_lm_common import HookedLMHandle, require_cuda


def _offset_to_index(seq_len: int, offset: int) -> int:
    """Convert end-relative offset to 0-based index (offset 0 = final token)."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    pos = int(seq_len) - 1 - int(offset)
    if not (0 <= pos < seq_len):
        raise IndexError(f"offset={offset} out of range for seq_len={seq_len}")
    return pos


@dataclass
class HFHooksModelCfg:
    device: str
    n_layers: int
    model_name: str


class HFHooksModel(HFLogitsModel):
    """HF wrapper exposing residual readout and position-wise patching."""

    _saebase_backend = "hf_hooks"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str,
        device: str,
    ) -> None:
        super().__init__(model, tokenizer, model_name=model_name, device=device)
        self.cfg = HFHooksModelCfg(
            device=str(device),
            n_layers=int(self.cfg.n_layers),
            model_name=str(model_name),
        )

    @property
    def n_layers(self) -> int:
        return int(self.cfg.n_layers)

    def _layer_module(self, layer: int) -> Any:
        n = int(layer)
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            layers = self._model.model.layers
            if 0 <= n < len(layers):
                return layers[n]
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            layers = self._model.transformer.h
            if 0 <= n < len(layers):
                return layers[n]
        raise ValueError(f"Cannot resolve HF layer module for layer={layer}")

    @torch.no_grad()
    def get_resid(
        self,
        input_ids: torch.Tensor,
        layer: int,
        offsets: list[int],
    ) -> torch.Tensor:
        """Return residual activations at end-relative offsets, shape [n_offsets, d_model]."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        dev = self.cfg.device
        if input_ids.device.type != dev.split(":")[0]:
            input_ids = input_ids.to(dev)
        seq_len = int(input_ids.shape[1])
        captured: dict[str, torch.Tensor] = {}

        def hook_fn(_module: Any, _inp: Any, out: Any) -> Any:
            hs = out[0] if isinstance(out, tuple) else out
            captured["resid"] = hs.detach()
            return out

        handle = self._layer_module(layer).register_forward_hook(hook_fn)
        try:
            self._model(input_ids=input_ids)
        finally:
            handle.remove()
        if "resid" not in captured:
            raise RuntimeError(f"hook failed to capture layer={layer}")
        resid = captured["resid"][0]
        rows = []
        for off in offsets:
            pos = _offset_to_index(seq_len, int(off))
            rows.append(resid[pos].clone())
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def run_with_patch(
        self,
        input_ids: torch.Tensor,
        layer: int,
        patches: dict[int, torch.Tensor],
        *,
        readout_offset: int = 0,
    ) -> torch.Tensor:
        """Forward with residual patches; return logits at readout_offset [vocab]."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        dev = self.cfg.device
        if input_ids.device.type != dev.split(":")[0]:
            input_ids = input_ids.to(dev)
        seq_len = int(input_ids.shape[1])
        index_patches: dict[int, torch.Tensor] = {}
        for off, vec in patches.items():
            pos = _offset_to_index(seq_len, int(off))
            index_patches[pos] = vec.to(dev)

        def hook_fn(_module: Any, _inp: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                hs = out[0].clone()
                rest = out[1:]
            else:
                hs = out.clone()
                rest = ()
            for pos, vec in index_patches.items():
                hs[0, pos] = vec
            return (hs, *rest) if rest else hs

        handle = self._layer_module(layer).register_forward_hook(hook_fn)
        try:
            out = self._model(input_ids=input_ids)
        finally:
            handle.remove()
        logits = out.logits[0, _offset_to_index(seq_len, int(readout_offset))]
        return logits


def load_hf_hooks_model(
    model_name: str,
    *,
    key: str,
    family: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HookedLMHandle:
    """Load HF model wrapped for residual hook capture/patch."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Qwen3.5's top-level config is conditional-generation multimodal,
    # even though this Stage A path supplies text-only input_ids.
    if model_name.startswith("Qwen/Qwen3.5-"):
        from transformers import Qwen3_5ForConditionalGeneration
        model_class = Qwen3_5ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM

    dev = device
    if dev == "cuda":
        dev = require_cuda()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _reject_quantized(model, model_name)
    model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = HFHooksModel(model, tok, model_name=model_name, device=dev)
    try:
        wrapped._saebase_weight_source = model_name  # type: ignore[attr-defined]
        wrapped._saebase_weight_source_deviation = False  # type: ignore[attr-defined]
    except Exception:
        pass
    return HookedLMHandle(
        model=wrapped,
        loaded_name=model_name,
        key=key,
        family=family,
    )


def is_hf_hooks_model(model: Any) -> bool:
    return getattr(model, "_saebase_backend", None) == "hf_hooks"


def is_transformer_lens_model(model: Any) -> bool:
    return hasattr(model, "run_with_cache") and hasattr(model, "run_with_hooks")
