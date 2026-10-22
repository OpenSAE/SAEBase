"""Native pyvene composition and serialization, not a PyTorch-hook substitute.

Construction, execution and source intervention hooks are handled by the
upstream pyvene IntervenableModel. No SAEs are silently passed off as pyvene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class PyveneRepresentation:
    layer: int
    component: str
    unit: Literal["pos", "h", "h.pos"] = "pos"
    maximum_units: int = 1
    intervention: Literal["vanilla", "zero"] = "vanilla"

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("pyvene layer must be nonnegative")
        if self.component not in {
            "block_output",
            "block_input",
            "mlp_output",
            "attention_output",
        }:
            raise ValueError("Choose a declared supported pyvene component")
        if self.unit not in {"pos", "h", "h.pos"}:
            raise ValueError("Unsupported original pyvene unit")
        if type(self.maximum_units) is not int or not 1 <= self.maximum_units <= 64:
            raise ValueError("Maximum intervened units out of bounds")
        if self.intervention not in {"vanilla", "zero"}:
            raise ValueError("Unsupported native pyvene intervention type")


@dataclass(frozen=True)
class PyvenePlan:
    representations: tuple[PyveneRepresentation, ...]
    mode: Literal["parallel", "serial"] = "parallel"
    schema: str = "opensae.wp12.native_pyvene.v1"

    def __post_init__(self) -> None:
        if self.schema != "opensae.wp12.native_pyvene.v1":
            raise ValueError("Wrong native pyvene config schema")
        if not 1 <= len(self.representations) <= 8:
            raise ValueError("Native pyvene must select 1–8 representations")
        if self.mode not in {"parallel", "serial"}:
            raise ValueError("Original pyvene accepts parallel or serial mode")
        for item in self.representations:
            if not isinstance(item, PyveneRepresentation):
                raise TypeError("Typed native pyvene site required")
        if len(set((r.layer, r.component, r.unit) for r in self.representations)) != len(
            self.representations
        ):
            raise ValueError("Duplicate native pyvene layer, component or unit")

    def json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def parse(cls, source: str) -> "PyvenePlan":
        record = json.loads(source)
        if not isinstance(record, dict) or set(record) != {"representations", "mode", "schema"}:
            raise ValueError("Native pyvene JSON has unexpected fields")
        values = record["representations"]
        if not isinstance(values, list):
            raise ValueError("pyvene representations must be a list")
        allowed = {"layer", "component", "unit", "maximum_units", "intervention"}
        if any(not isinstance(v, dict) or set(v) != allowed for v in values):
            raise ValueError("pyvene representation has unknown fields")
        return cls(
            tuple(PyveneRepresentation(**v) for v in values),
            mode=record["mode"],
            schema=record["schema"],
        )


def build_native_pyvene(model: Any, plan: PyvenePlan) -> Any:
    """Create the original pyvene backend (lazy-imported in its own environment)."""
    if not isinstance(plan, PyvenePlan):
        raise TypeError("Typed native pyvene plan required")
    import pyvene as pv

    types = {"vanilla": pv.VanillaIntervention, "zero": pv.ZeroIntervention}
    representations = [
        pv.RepresentationConfig(
            layer=site.layer,
            component=site.component,
            unit=site.unit,
            max_number_of_units=site.maximum_units,
            intervention_type=types[site.intervention],
        )
        for site in plan.representations
    ]
    upstream_config = pv.IntervenableConfig(
        representations=representations,
        mode=plan.mode,
    )
    return pv.IntervenableModel(upstream_config, model)


def execute_native_pyvene(
    intervenable: Any,
    *,
    base: Any,
    sources: Any,
    unit_locations: dict[str, Any],
) -> tuple[Any, Any]:
    """Return upstream baseline/intervened outputs, no post-hoc simulated logits."""
    if not type(intervenable).__module__.startswith("pyvene."):
        raise TypeError("Require original native pyvene IntervenableModel")
    if not isinstance(unit_locations, dict) or not unit_locations:
        raise ValueError("Actual native pyvene intervention positions required")
    original, counterfactual = intervenable(base, sources=sources, unit_locations=unit_locations)
    if counterfactual is None:
        raise RuntimeError("Original pyvene returned no intervention output")
    return original, counterfactual


def save_native_pyvene(intervenable: Any, output: Path) -> None:
    """Serialize via original library; never include the underlying base model."""
    if not type(intervenable).__module__.startswith("pyvene."):
        raise TypeError("Native pyvene object required")
    if output.exists():
        raise FileExistsError("Native intervention artifact must be immutable")
    intervenable.save(str(output), include_model=False, save_to_hf_hub=False)


def load_native_pyvene(*, model: Any, directory: Path, trusted_local_source: bool = False) -> Any:
    """pyvene's original loader uses pickle; never auto-load untrusted artifacts."""
    if not trusted_local_source:
        raise PermissionError("pyvene loads pickle-backed intervention states; trust is required")
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError("Trusted native pyvene intervention folder unavailable")
    import pyvene as pv

    return pv.IntervenableModel.load(
        str(directory), model=model, from_huggingface_hub=False, include_model=False
    )
