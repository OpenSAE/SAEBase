"""Offline Localize integration contracts; original estimator is optional."""
from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pytest

from saebase.api import ActivationSite, ModelAdapter
from saebase.integrations import (
    LocalizationGrid, collect_localization_grid, evaluate_localization,
)
from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE


class OriginalStub:
    __version__ = "reference-stub"

    def __init__(self):
        self.received = None

    def exact_scope_curve(self, z, n_factors):
        self.received = np.array(z, copy=True)
        assert n_factors == 3
        class Result:
            total_variance = 1.0
            curve = np.linspace(1, 0, 22)
            def budget(self, tolerance):
                assert tolerance == pytest.approx(.1)
                return 19
            def assignment(self, budget):
                assert budget == 19
                return np.array([1, 2, 3, 4, 0, 1, 2])
        return Result()

    def estimated_scope_curve(self, z, pairs_per_scope, seed):
        assert pairs_per_scope == 32 and seed == 4
        return np.zeros(22)


def _collect():
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    return collect_localization_grid(
        model, sae, site=ActivationSite(0, "blocks.0.hook_resid_post"),
        prompts_by_state={
            i: f"{1 + (i & 1)} {3 + ((i >> 1) & 1)} {5 + ((i >> 2) & 1)}"
            for i in reversed(range(8))
        }, n_factors=3
    )


def test_capture_order_and_delegate():
    grid = _collect()
    assert grid.codes.shape == (8, 7)
    assert grid.state_sample_ids == tuple(f"state-{i}" for i in range(8))
    assert not np.array_equal(grid.codes[0], grid.codes[-1])
    stub = OriginalStub()
    result = evaluate_localization(grid, paper=stub, pairs_per_scope=32, seed=4)
    np.testing.assert_array_equal(stub.received, grid.codes)
    assert result.budget == 19 and result.estimated_curve.shape == (22,)
    assert json.loads(json.dumps(result.summary(), allow_nan=False))["paired_estimate"]


def test_incomplete_states_rejected():
    model = ModelAdapter(ResearchToyModel().eval())
    with pytest.raises(ValueError, match="state keys"):
        collect_localization_grid(
            model, ResearchToySAE().eval(),
            site=ActivationSite(0, "blocks.0.hook_resid_post"),
            prompts_by_state={i: str(i + 1) for i in range(7)}, n_factors=3
        )


def test_invalid_data_and_options_rejected():
    with pytest.raises(ValueError, match="finite"):
        LocalizationGrid(np.full((8, 2), np.nan), 3, tuple(str(i) for i in range(8)), "bad")
    grid = _collect()
    with pytest.raises(ValueError, match="relative_omission"):
        evaluate_localization(grid, paper=OriginalStub(), relative_omission=1.1)
    with pytest.raises(ValueError, match="pairs_per_scope"):
        evaluate_localization(grid, paper=OriginalStub(), pairs_per_scope=0)


def test_original_package_when_installed():
    paper = pytest.importorskip("sae_localize")
    grid = _collect()
    result = evaluate_localization(grid, paper=paper)
    native = paper.exact_scope_curve(grid.codes, n_factors=3)
    np.testing.assert_allclose(result.curve, native.curve)
    assert result.budget == native.budget(.1 * native.total_variance)
