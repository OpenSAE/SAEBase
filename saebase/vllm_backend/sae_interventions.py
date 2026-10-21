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
from saebase.interventions.operators import apply_condition
from saebase.interventions.specs import Condition


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
        if not bool(getattr(model, "hooks_available", False)):
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
            point = f"resid_post.{layer}"
            kwargs["static_points"] = [point]
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
        clean_cache = sync.capture(ids, [point])
        clean_tensor = clean_cache[point]
        if (
            not isinstance(clean_tensor, torch.Tensor)
            or clean_tensor.ndim != 3
            or clean_tensor.shape[0] != 1
            or clean_tensor.shape[1] != len(ids)
        ):
            raise RuntimeError("vLLM capture failed to return the exact requested token rows")
        original = clean_tensor[0, idx].detach().clone()
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
            edited_cache = sync.capture(ids, [point])
        edited_tensor = edited_cache[point]
        if edited_tensor.shape != clean_tensor.shape:
            raise RuntimeError("edited vLLM capture returned mismatched token geometry")
        actual = edited_tensor[0, idx].detach().clone()
        expected = original + delta.to(device=original.device, dtype=original.dtype)
        err = float((actual.float() - expected.float()).abs().max().item())
        if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
            raise RuntimeError(
                f"vLLM SAE edit was not applied at the requested tap (max abs {err})"
            )
        # Verify that OTHER prompt positions were not steered.
        mask = torch.tensor(masked, dtype=torch.long, device=edited_tensor.device)
        if len(masked) and not torch.allclose(
            edited_tensor[0, mask].float(),
            clean_tensor[0, mask].float(),
            atol=atol,
            rtol=rtol,
        ):
            raise RuntimeError("steer changed excluded prompt positions")
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
        )
