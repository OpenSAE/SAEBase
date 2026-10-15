"""HuggingFace CausalLM logits backend (no TransformerLens hooks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.hooked_lm_common import HookedLMHandle, require_cuda


@dataclass
class HFLogitsModelCfg:
    device: str
    n_layers: int
    model_name: str


class HFLogitsModel:
    """Thin wrapper around AutoModelForCausalLM + AutoTokenizer for logit readout."""

    _saebase_backend = "hf"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str,
        device: str,
    ) -> None:
        self._model = model
        self.tokenizer = tokenizer
        text_config = getattr(model.config, "text_config", None)
        n_layers = int(getattr(model.config, "num_hidden_layers", None)
                       or getattr(text_config, "num_hidden_layers", -1))
        self.cfg = HFLogitsModelCfg(
            device=str(device),
            n_layers=n_layers,
            model_name=str(model_name),
        )

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError(f"to_tokens expects str, got {type(text).__name__}")
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if prepend_bos:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                encoded = [int(bos_id)] + list(encoded)
        ids = torch.tensor([encoded], dtype=torch.long, device=self.cfg.device)
        return ids

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be Tensor, got {type(input_ids).__name__}")
        dev = self.cfg.device
        if input_ids.device.type != dev.split(":")[0]:
            input_ids = input_ids.to(dev)
        with torch.no_grad():
            out = self._model(input_ids=input_ids)
        return out.logits


def _reject_quantized(model: Any, repo_id: str) -> None:
    """Quantized checkpoints are allowed; keep a no-op for call-site compatibility."""
    del model, repo_id


def load_hf_logits_model(
    model_name: str,
    *,
    key: str,
    family: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HookedLMHandle:
    """Load HF CausalLM for logit-only evaluations (quantized checkpoints allowed)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device
    if dev == "cuda":
        dev = require_cuda()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _reject_quantized(model, model_name)

    model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = HFLogitsModel(model, tok, model_name=model_name, device=dev)
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
