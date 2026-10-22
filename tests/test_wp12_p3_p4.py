"""Fast CPU contracts for the P3 intervention and P4 native benchmark bridge."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from saebase.wp12.interventions import (
    Edit,
    EditPlan,
    ModuleEdit,
    compare_module_edits,
)
from saebase.wp12.benchmarks import ComparisonPoint, compare_sae_points


def test_edit_roundtrip_and_parallel_conflict():
    plan = EditPlan(
        "pythia70m_resid_sm",
        3,
        -1,
        (Edit("ablate", (7,)), Edit("scale", (11,), value=2.0)),
    )
    assert EditPlan.parse(plan.json()) == plan
    with pytest.raises(ValueError, match="disjoint"):
        EditPlan("s", 0, -1, (Edit("ablate", (1,)), Edit("clamp", (1,), value=1)))
    with pytest.raises(ValueError, match="unknown or missing"):
        EditPlan.parse(
            plan.json().replace(
                '"sae_key":"pythia70m_resid_sm"', '"sae_key":"pythia70m_resid_sm","unsafe":true'
            )
        )


def test_native_pytorch_intervention_is_real_and_reset():
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
    data = torch.randn(2, 4)
    untouched = model(data).clone()
    changed = compare_module_edits(model, data, (ModuleEdit("0", "zero", (0, 1)),))
    assert changed["fired"] == {"0": 1}
    assert changed["delta_l2"] >= 0
    assert torch.equal(model(data), untouched)
    with pytest.raises(ValueError):
        compare_module_edits(model, data, (ModuleEdit("missing", "zero"),))
    with pytest.raises(ValueError, match="Duplicate"):
        compare_module_edits(
            model, data, (ModuleEdit("0", "zero"), ModuleEdit("0", "scale", value=2))
        )


def _point(sae, metric=0.75, model="a"):
    return ComparisonPoint(
        model_sha256=model * 64,
        sae_sha256=sae * 64,
        hook="blocks.3.hook_resid_post",
        dataset_sha256="d" * 64,
        native_commit="8" * 40,
        metric_id="saebench.core.mse",
        metric_value=metric,
        l0=32 if sae == "1" else 64,
        sae_name="SAE " + sae,
    )


def test_compare_requires_two_matched_native_checkpoints():
    result = compare_sae_points([_point("1"), _point("2", 0.62)])
    assert result["comparable"] is True
    assert [p["l0"] for p in result["points"]] == [32, 64]
    assert result["uncertainty"].startswith("not_computed")
    mismatch = compare_sae_points([_point("1"), _point("2", model="b")])
    assert mismatch["comparable"] is False
    assert "model_sha256" in mismatch["mismatches"]
    with pytest.raises(ValueError):
        compare_sae_points([_point("1"), _point("1")])
