"""Typed, runnable SAE edits and native PyTorch interventions for WP12 P3.

The two modes are deliberately distinct: SAE code changes preserve the
reconstruction residual; PyTorch hooks operate on native module tensors.
A native pyvene execution can be supplied via its original implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Literal, Sequence

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features
from saebase.api.features import snapshot_from_activation
from saebase.interventions.specs import Condition
from saebase.interventions.operators import apply_condition


@dataclass(frozen=True)
class Edit:
    operation: Literal["identity", "ablate", "scale", "clamp", "add", "coalition", "reconstruction"]
    feature_ids: tuple[int, ...] = ()
    value: float = 0.0
    retained_ids: tuple[int, ...] = ()

    def condition(self) -> Condition:
        return Condition(name=self.operation, operation=self.operation,
                         feature_ids=self.feature_ids, value=self.value,
                         retained_ids=self.retained_ids)


@dataclass(frozen=True)
class EditPlan:
    sae_key: str
    layer: int
    token_position: int
    steps: tuple[Edit, ...]
    compose: Literal["parallel", "sequential"] = "parallel"
    schema: str = "opensae.wp12.p3.edit.v1"

    def __post_init__(self) -> None:
        if self.schema != "opensae.wp12.p3.edit.v1":
            raise ValueError("Unknown edit schema")
        if not self.sae_key or type(self.layer) is not int or self.layer < 0:
            raise ValueError("Explicit SAE identity and layer required")
        if type(self.token_position) is not int:
            raise ValueError("Token position must be integer")
        if not 1 <= len(self.steps) <= 8 or self.compose not in {"parallel", "sequential"}:
            raise ValueError("1-8 edits and a valid composition are required")
        touched: set[int] = set()
        for step in self.steps:
            if not isinstance(step, Edit):
                raise TypeError("Only typed SAE Edit records accepted")
            step.condition()
            if self.compose == "parallel":
                if step.operation in {"identity", "reconstruction"} and len(self.steps) > 1:
                    raise ValueError("Identity/reconstruction cannot be combined in parallel")
                overlap = touched & set(step.feature_ids)
                if overlap:
                    raise ValueError("Parallel edits must target disjoint SAE features")
                touched.update(step.feature_ids)

    def json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False, separators=(",", ":"))

    @classmethod
    def parse(cls, raw: str) -> "EditPlan":
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {
            "schema", "sae_key", "layer", "token_position", "steps", "compose"
        }:
            raise ValueError("Edit config has unknown or missing fields")
        if not isinstance(value["steps"], list):
            raise ValueError("Steps must be a list")
        steps = []
        for item in value["steps"]:
            if not isinstance(item, dict) or set(item) != {
                "operation", "feature_ids", "value", "retained_ids"
            }:
                raise ValueError("Unexpected step fields")
            if not isinstance(item["feature_ids"], list) or not isinstance(item["retained_ids"], list):
                raise ValueError("Feature IDs must be lists")
            steps.append(Edit(item["operation"], tuple(item["feature_ids"]),
                              item["value"], tuple(item["retained_ids"])))
        return cls(value["sae_key"], value["layer"], value["token_position"],
                   tuple(steps), value["compose"], value["schema"])


@torch.no_grad()
def compare_sae_edits(bundle: Any, tokens: torch.Tensor, plan: EditPlan,
                      *, readout_position: int = -1) -> dict[str, Any]:
    """Real baseline vs edited teacher-forced logits, no invented generation.

    The supplied bundle is an already loaded SAEBase LoadedResearchPair, so
    callers control weights and device. Sequential edits re-encode the current
    activation; parallel edits combine disjoint feature changes in one code
    vector. An unsupported SAE normalization is rejected by existing API.
    """
    if bundle.identity.sae_key != plan.sae_key or bundle.identity.layer != plan.layer:
        raise ValueError("Plan selects a different pretrained SAE/layer")
    model, sae = bundle.model, bundle.sae
    site = ActivationSite(plan.layer, bundle.identity.hook_name, plan.token_position)
    ids = model.tokens(tokens=tokens)
    snapshot = capture_features(model, sae, tokens=ids, site=site)
    model.assert_identity_patch(tokens=ids, site=site)
    baseline = model.forward_logits(tokens=ids, readout_position=readout_position)
    if plan.compose == "parallel":
        replacements: dict[int, float] = {}
        if len(plan.steps) == 1 and plan.steps[0].operation in {"identity", "reconstruction"}:
            patched = apply_condition(sae, snapshot, plan.steps[0].condition())
        else:
            for step in plan.steps:
                codes = snapshot.codes
                for idx in step.feature_ids:
                    if idx >= codes.numel():
                        raise IndexError("Feature outside actual SAE width")
                    original = float(codes[idx].item())
                    replacement = {
                        "ablate": 0.0,
                        "scale": original * step.value,
                        "clamp": float(step.value),
                        "add": original + step.value,
                        "coalition": original if idx in step.retained_ids else 0.0,
                    }[step.operation]
                    replacements[idx] = replacement
            patched = edit_features(sae, snapshot, replacements=replacements)
    else:
        current = snapshot
        patched = None
        for step in plan.steps:
            patched = apply_condition(sae, current, step.condition())
            current = snapshot_from_activation(
                sae, tokens=ids, site=site, activation=patched.activation
            )
        assert patched is not None
    changed = model.patch_logits(tokens=ids, site=site, activation=patched.activation,
                                 readout_position=readout_position)
    if changed.shape != baseline.shape or not bool(torch.isfinite(changed).all()):
        raise RuntimeError("Invalid edited logits")
    return {
        "schema": "opensae.wp12.p3.edit_result.v1",
        "backend": model.backend,
        "model_key": bundle.identity.model_key,
        "sae_key": plan.sae_key,
        "site": site.hook_name,
        "token_position": site.position,
        "readout_position": readout_position,
        "compose": plan.compose,
        "residual_semantics": "preserving" if all(x.operation != "reconstruction" for x in plan.steps)
                              else "full_reconstruction",
        "baseline_logits": baseline.detach().cpu(),
        "edited_logits": changed.detach().cpu(),
        "logit_delta_l2": float(torch.linalg.vector_norm(changed.float() - baseline.float()).item()),
        "plan": json.loads(plan.json()),
        "model_engine": getattr(model.model, "_saebase_transformerlens_engine", None),
    }


@dataclass(frozen=True)
class ModuleEdit:
    """A native PyTorch module's forward output, not SAE decoder features."""

    module_name: str
    operation: Literal["zero", "scale", "replace"]
    positions: tuple[int, ...] = ()
    value: float = 0.0

    def __post_init__(self) -> None:
        if not self.module_name or any(part.startswith("_") or not part for part in self.module_name.split(".")):
            raise ValueError("Choose a named public model module")
        if self.operation not in {"zero", "scale", "replace"}:
            raise ValueError("Unknown module edit")
        if any(type(x) is not int or x < 0 for x in self.positions) or len(set(self.positions)) != len(self.positions):
            raise ValueError("Positions must be unique nonnegative indices")
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Invalid module edit value")


@torch.no_grad()
def compare_module_edits(model: torch.nn.Module, inputs: Any,
                         edits: Sequence[ModuleEdit]) -> dict[str, Any]:
    """Genuine forward hooks on arbitrary PyTorch modules including CNNs.

    Caller retains responsibility for user-selected output semantics. The
    same module cannot be targeted twice; unsupported tuple outputs fail.
    """
    if not 1 <= len(edits) <= 8:
        raise ValueError("Require one to eight native edits")
    modules = dict(model.named_modules())
    if len({e.module_name for e in edits}) != len(edits):
        raise ValueError("Duplicate module targets; merge effects explicitly")
    for edit in edits:
        if edit.module_name not in modules or not isinstance(edit, ModuleEdit):
            raise ValueError("Unknown or invalid module path")
    model.eval()
    baseline = model(inputs)
    if not isinstance(baseline, torch.Tensor):
        raise TypeError("This local adapter requires Tensor output")
    counts = {edit.module_name: 0 for edit in edits}
    handles = []
    def make_hook(edit: ModuleEdit):
        def hook(module: torch.nn.Module, args: Any, output: Any) -> Any:
            if not isinstance(output, torch.Tensor):
                raise TypeError("Unsupported non-Tensor module output")
            counts[edit.module_name] += 1
            data = output.clone()
            positions = edit.positions
            if positions and (data.ndim < 2 or any(p >= data.shape[-1] for p in positions)):
                raise ValueError("Neuron index outside output last dimension")
            target = data[..., list(positions)] if positions else data
            value = (target * 0 if edit.operation == "zero" else
                     target * edit.value if edit.operation == "scale" else
                     torch.full_like(target, edit.value))
            if positions:
                data[..., list(positions)] = value
            else:
                data = value
            return data
        return hook
    try:
        for edit in edits:
            handles.append(modules[edit.module_name].register_forward_hook(make_hook(edit)))
        edited = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if any(v != 1 for v in counts.values()):
        raise RuntimeError("A selected module did not fire exactly once")
    if not isinstance(edited, torch.Tensor) or edited.shape != baseline.shape:
        raise RuntimeError("Native edited output shape changed")
    return {
        "baseline": baseline.detach().clone(), "edited": edited.detach().clone(),
        "delta_l2": float((edited.float()-baseline.float()).norm().item()),
        "fired": counts, "backend": "pytorch_forward_hooks",
        "semantics": "module_tensor_not_sae_feature",
    }


def run_native_pyvene(intervenable: Any, *, base: Any, sources: Any,
                      unit_locations: dict[str, Any], **upstream_kwargs: Any) -> Any:
    """Invoke an already constructed ORIGINAL pyvene IntervenableModel.

    The caller supplies its official IntervenableConfig, registrations and
    source, so OpenSAE never invents a pyvene location map or changes the
    computation to SAE steering. Use pyvene's own output and provenance.
    """
    if "pyvene" not in type(intervenable).__module__.lower():
        raise TypeError("Expected an actual native pyvene intervenable")
    if not callable(intervenable) or not isinstance(unit_locations, dict):
        raise TypeError("Native pyvene callable and unit locations required")
    return intervenable(base, sources=sources, unit_locations=unit_locations, **upstream_kwargs)
