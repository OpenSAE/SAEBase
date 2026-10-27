"""Real SAE feature editing on the upstream interp-engine vLLM hook backend.

Uses interp-engine's own per-request read/write taps; SAE codes and the
residual-preserving delta are computed with SAEBase's original operator.
This is an opt-in backend-native experiment, never a silent replacement for
HF/SAEBase. vLLM numerical equivalence must be measured separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from saebase.api import ActivationSite
from saebase.api.features import snapshot_from_activation
from tasks.LLM.intervene.backend.operators import apply_condition
from tasks.LLM.intervene.backend.specs import Condition


@dataclass(frozen=True)
class VLLMSAEOutcome:
    backend: str
    input_length: int
    layer: int
    patch_position: int
    feature_ids_before: tuple[int, ...]
    activation_error: float
    clean_residual: torch.Tensor
    edited_residual: torch.Tensor
    expected_residual: torch.Tensor
    clean_next_text: str | None
    edited_next_text: str | None
    clean_logits: torch.Tensor
    edited_logits: torch.Tensor
    protocol: str = "single_prompt_position_edit_v1"
    scientific_parity: str = "UNVERIFIED_VS_HF"


class InterpVLLMSAE:
    """Per-request SAE interventions using interp-engine's real vLLM tap API.

    The generated token is a *single edited-prompt response*, not a proof of
    reference clean-prefix cached steering. No hidden use of generation-only
    vLLM is allowed. Each model must be used inside a dedicated worker process
    because vLLM-static graph settings are process-global.
    """

    def __init__(self, model: Any, sae: Any, *, layer: int, backend: str) -> None:
        if backend not in ("vllm", "vllm-static", "eager"):
            raise ValueError("vllm-generate has no SAE activation hooks")
        if backend == "vllm-static":
            from interp_engine import Address

            point = Address("resid_post", layer)
            reads = getattr(model, "static_points", ())
            writes = getattr(model, "static_writes", ())
            if point not in reads or point not in writes:
                raise RuntimeError("static backend did not declare both read and write taps")
        elif not bool(getattr(model, "hooks_available", False)):
            raise RuntimeError("selected interp-engine backend has no capture/steering hooks")
        if layer < 0 or layer >= int(model.n_layers):
            raise ValueError("requested SAE site not present")
        self.model = model
        self.sae = sae
        self.layer = layer
        self.backend = backend

    @staticmethod
    def load(
        model_name: str,
        sae: Any,
        *,
        layer: int,
        backend: str = "vllm",
        gpu_memory_utilization: float = 0.55,
        max_model_len: int = 1024,
    ) -> "InterpVLLMSAE":
        if backend not in ("vllm", "vllm-static"):
            raise ValueError("load() requires a capture-capable vLLM backend")
        if not 0.1 <= gpu_memory_utilization <= 0.85:
            raise ValueError("GPU memory budget outside conservative bounds")
        # Intentional lazy import. SAEBase base install must not require vLLM.
        from interp_engine import load_model

        kwargs: dict[str, Any] = {
            "backend": backend,
            "dtype": "bfloat16",
            "num_gpus": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }
        if backend == "vllm-static":
            from transformers import AutoConfig

            n_layers = int(AutoConfig.from_pretrained(model_name).num_hidden_layers)
            point = f"resid_post.{layer}"
            final = f"resid_post.{n_layers - 1}"
            kwargs["static_points"] = list(dict.fromkeys((point, final)))
            kwargs["static_writes"] = [point]
        model = load_model(model_name, **kwargs)
        return InterpVLLMSAE(model, sae, layer=layer, backend=backend)

    def close(self) -> None:
        from interp_engine.sync import sync_model

        sync_model(self.model).shutdown()

    @torch.no_grad()
    def measure(
        self,
        token_ids: Sequence[int],
        condition: Condition,
        *,
        patch_position: int = -1,
        next_token_probe: bool = True,
        atol: float = 0.01,
        rtol: float = 0.01,
    ) -> VLLMSAEOutcome:
        from interp_engine import Address
        from interp_engine.steer import steer
        from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec
        from interp_engine.sync import sync_model

        ids = tuple(int(t) for t in token_ids)
        if not ids or any(t < 0 for t in ids):
            raise ValueError("input token IDs must be nonempty nonnegative integers")
        idx = (len(ids) + patch_position) if patch_position < 0 else patch_position
        if idx < 0 or idx >= len(ids):
            raise ValueError("invalid token position")
        point = Address("resid_post", self.layer)
        sync = sync_model(self.model)
        final_point = Address("resid_post", int(self.model.n_layers) - 1)
        points = list(dict.fromkeys((point, final_point)))
        clean_cache = sync.capture(ids, points)
        clean_tensor = clean_cache[point]
        if (
            not isinstance(clean_tensor, torch.Tensor)
            or clean_tensor.ndim != 2
            or clean_tensor.shape[0] != len(ids)
        ):
            raise RuntimeError("interp-engine capture must return [tokens, hidden_dim]")
        original = clean_tensor[idx].detach().clone()
        site = ActivationSite(self.layer, f"blocks.{self.layer}.hook_resid_post", idx)
        tokens = torch.tensor([ids], device=original.device, dtype=torch.long)
        snapshot = snapshot_from_activation(self.sae, tokens=tokens, site=site, activation=original)
        patch = apply_condition(self.sae, snapshot, condition)
        delta = (patch.activation.to(original.device, original.dtype) - original).detach()
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError("nonfinite SAE edit delta")
        ops = SteeringSpec(
            layers={
                self.layer: LayerSteeringSpec(
                    operations=[AddSpec(vector=delta.float().cpu(), scale=1.0)]
                )
            },
            point="resid_post",
        )
        # Position mask lists EXCLUDED token IDs, not the edited one.
        masked = [i for i in range(len(ids)) if i != idx]
        with steer(self.model, ops, prompt_token_ids=ids, position_mask=masked, generated=False):
            edited_cache = sync.capture(ids, points)
        edited_tensor = edited_cache[point]
        if edited_tensor.shape != clean_tensor.shape:
            raise RuntimeError("edited vLLM capture returned mismatched token geometry")
        actual = edited_tensor[idx].detach().clone()
        expected = original + delta.to(device=original.device, dtype=original.dtype)
        err = float((actual.float() - expected.float()).abs().max().item())
        if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
            raise RuntimeError(
                f"vLLM SAE edit was not applied at the requested tap (max abs {err})"
            )
        # Capture is [tokens, hidden]. Compare the excluded token rows, not
        # a slice of the first token's hidden units.
        if masked:
            kept = torch.tensor(masked, dtype=torch.long)
            if not torch.allclose(
                edited_tensor[kept].float(),
                clean_tensor[kept].float(),
                atol=atol,
                rtol=rtol,
            ):
                raise RuntimeError("steer changed excluded prompt positions")

        # Read the final residual from the SAME worker forward, then apply
        # the backend's genuine final norm and unembedding. The resulting
        # vectors are checked against independent HF reference logits below.
        def final_logits(cache: dict) -> torch.Tensor:
            resid = cache[final_point]
            if not isinstance(resid, torch.Tensor) or resid.ndim != 2:
                raise RuntimeError("missing [tokens, width] final residual for logits")
            vector = sync.decode_residuals(resid[-1:].contiguous())
            if not isinstance(vector, torch.Tensor):
                raise RuntimeError("interp-engine residual unembedding failed")
            return vector.detach().float().reshape(-1).cpu()

        clean_logits = final_logits(clean_cache)
        edited_logits = final_logits(edited_cache)
        if clean_logits.shape != edited_logits.shape or clean_logits.numel() < 2:
            raise RuntimeError("vLLM readout logits mismatched")
        clean_text = edited_text = None
        if next_token_probe:
            clean_text = sync.generate_text(ids, max_tokens=1, temperature=0.0, seed=0)
            with steer(
                self.model, ops, prompt_token_ids=ids, position_mask=masked, generated=False
            ):
                edited_text = sync.generate_text(ids, max_tokens=1, temperature=0.0, seed=0)
        support = tuple(
            torch.nonzero(snapshot.codes != 0, as_tuple=False).reshape(-1).cpu().tolist()
        )
        return VLLMSAEOutcome(
            backend=self.backend,
            input_length=len(ids),
            layer=self.layer,
            patch_position=idx,
            feature_ids_before=support,
            activation_error=err,
            clean_residual=original.cpu(),
            edited_residual=actual.cpu(),
            expected_residual=expected.cpu(),
            clean_next_text=clean_text,
            edited_next_text=edited_text,
            clean_logits=clean_logits,
            edited_logits=edited_logits,
        )
