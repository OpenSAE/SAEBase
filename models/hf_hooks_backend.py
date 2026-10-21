"""HF CausalLM residual-stream capture/patch via forward hooks (TL fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

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
    def capture_with_logits(
        self,
        input_ids: torch.Tensor,
        layer: int,
        offsets: list[int],
        *,
        readout_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One unpadded forward: site rows [n_offsets, d_model] and readout logits [vocab]."""
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
            output = self._model(input_ids=input_ids)
        finally:
            handle.remove()
        if "resid" not in captured:
            raise RuntimeError(f"hook failed to capture layer={layer}")
        resid = captured["resid"][0]
        rows = [resid[_offset_to_index(seq_len, int(off))].clone() for off in offsets]
        logits = output.logits[0, _offset_to_index(seq_len, int(readout_offset))].clone()
        return torch.stack(rows, dim=0), logits

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

    def supports_suffix_replay(self) -> bool:
        """True when early blocks can be skipped without changing later-block inputs.

        Only ``model.model.layers`` stacks whose parent forward calls each block
        in order are supported. Other layouts are refused instead of approximated.
        """
        layers = getattr(getattr(self._model, "model", None), "layers", None)
        return layers is not None and len(layers) == self.n_layers and self.n_layers > 0

    def _decoder_layers(self) -> Any:
        if not self.supports_suffix_replay():
            raise ValueError(
                "suffix replay is unsupported for this architecture; use the original hook forward"
            )
        return self._model.model.layers

    def _inject_site_hidden(self, layer: int, hidden: torch.Tensor) -> Iterator[None]:
        """Yield while layers ``0..layer`` return ``hidden`` instead of recomputing."""
        layers = self._decoder_layers()
        n = int(layer)
        if not 0 <= n < len(layers):
            raise ValueError(f"layer {layer} outside suffix-replay stack")
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("site hidden must have shape [1, seq, d_model]")
        saved: list[tuple[Any, Any]] = []

        def emit(_hidden_states: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            del args, kwargs
            return hidden.clone()

        def passthrough(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return hidden_states

        try:
            for index, module in enumerate(layers):
                if index > n:
                    break
                saved.append((module, module.forward))
                module.forward = emit if index == 0 else passthrough  # type: ignore[method-assign]
            yield
        finally:
            for module, original in saved:
                module.forward = original  # type: ignore[method-assign]

    @torch.no_grad()
    def cache_site_with_logits(
        self, input_ids: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One full forward: site hidden ``[1, seq, d]`` and logits ``[1, seq, vocab]``."""
        self._decoder_layers()
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("suffix replay expects one unpadded prompt with shape [1, seq]")
        dev = self.cfg.device
        if input_ids.device.type != dev.split(":")[0]:
            input_ids = input_ids.to(dev)
        captured: dict[str, torch.Tensor] = {}

        def hook_fn(_module: Any, _inp: Any, out: Any) -> Any:
            hs = out[0] if isinstance(out, tuple) else out
            captured["resid"] = hs.detach()
            return out

        handle = self._layer_module(layer).register_forward_hook(hook_fn)
        try:
            output = self._model(input_ids=input_ids)
        finally:
            handle.remove()
        if "resid" not in captured or captured["resid"].shape[0] != 1:
            raise RuntimeError(f"suffix replay failed to cache layer={layer}")
        if output.logits.ndim != 3 or output.logits.shape[1] != input_ids.shape[1]:
            raise RuntimeError("suffix replay logits do not match the prompt length")
        return captured["resid"].clone(), output.logits.detach().clone()

    @torch.no_grad()
    def cache_site_hidden(self, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        """Full-sequence block output at ``layer``, shape ``[1, seq, d_model]``."""
        hidden, _logits = self.cache_site_with_logits(input_ids, layer)
        return hidden

    @torch.no_grad()
    def logits_from_site_hidden(
        self,
        input_ids: torch.Tensor,
        layer: int,
        hidden: torch.Tensor,
        *,
        readout_offset: int = 0,
    ) -> torch.Tensor:
        """Logits after replacing the cached block output and running only later blocks."""
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("suffix replay expects one unpadded prompt with shape [1, seq]")
        dev = self.cfg.device
        if input_ids.device.type != dev.split(":")[0]:
            input_ids = input_ids.to(dev)
        site = hidden.to(device=input_ids.device)
        if site.shape[1] != input_ids.shape[1]:
            raise ValueError("cached site sequence length does not match the prompt")
        if not bool(torch.isfinite(site).all()):
            raise ValueError("cached site hidden contains non-finite values")
        seq_len = int(input_ids.shape[1])
        try:
            injector = self._inject_site_hidden(layer, site)
            next(injector)
            try:
                out = self._model(input_ids=input_ids)
            finally:
                injector.close()
        except StopIteration:
            raise RuntimeError("suffix replay injector ended before the forward") from None
        return out.logits[0, _offset_to_index(seq_len, int(readout_offset))]

    @torch.no_grad()
    def run_batch(
        self,
        input_ids: torch.Tensor,
        *,
        layer: int,
        positions: list[int],
        readout_positions: list[int],
        attention_mask: torch.Tensor,
        patches: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-wise HF forward matching get_resid and run_with_patch.

        Positions are absolute, zero-based token positions for each sample.
        Returns original site activations [batch, d_model] and readout logits
        [batch, vocab]. When patches are supplied, each row is edited with
        run_with_patch. A fused padded GEMM is not used: on Qwen3 bf16 the
        reduction depends on the row count and changes TopK feature ids.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        batch, length = input_ids.shape
        if batch == 0 or length == 0:
            raise ValueError("empty token batch")
        if (
            len(positions) != batch
            or len(readout_positions) != batch
            or any(
                type(p) is not int or p < 0 or p >= length for p in positions + readout_positions
            )
        ):
            raise ValueError("one valid absolute patch/readout position per sample required")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        mask = attention_mask.to(dtype=torch.long, device=self.cfg.device)
        if (
            not bool(((mask == 0) | (mask == 1)).all())
            or not bool((mask[:, 1:] <= mask[:, :-1]).all())
            or any(
                int(mask[i, p]) != 1 or int(mask[i, r]) != 1
                for i, (p, r) in enumerate(zip(positions, readout_positions))
            )
        ):
            raise ValueError("right padding with valid patch/readout positions required")
        ids = input_ids.to(self.cfg.device)
        if patches is not None:
            if (
                not isinstance(patches, torch.Tensor)
                or patches.ndim != 2
                or patches.shape[0] != batch
                or not bool(torch.isfinite(patches).all())
            ):
                raise ValueError("patches must be finite [batch, d_model]")
        # A fused bf16 GEMM uses a different K-reduction tile once the row count
        # changes. On Qwen3-1.7B that 1-ulp gap in block 0 grows into TopK
        # feature substitutions by the readout layer. Each row therefore uses
        # the same unpadded forward as get_resid / run_with_patch.
        activations: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for i in range(batch):
            length = int(mask[i].sum().item())
            row = ids[i : i + 1, :length]
            pos = positions[i]
            read = readout_positions[i]
            if pos >= length or read >= length:
                raise ValueError("patch/readout position is outside the unpadded row")
            offset = length - 1 - pos
            read_offset = length - 1 - read
            activations.append(self.get_resid(row, layer, [offset])[0])
            if patches is None:
                row_logits = self(row)
                logits.append(row_logits[0, read].detach().clone())
            else:
                logits.append(
                    self.run_with_patch(
                        row,
                        layer,
                        {offset: patches[i]},
                        readout_offset=read_offset,
                    )
                    .detach()
                    .clone()
                )
        return torch.stack(activations, dim=0), torch.stack(logits, dim=0)


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
