"""Original Localize numerical acceptance for the Research Lab six-factor controls.

This test executes the unchanged upstream estimator when its pinned optional
package is present. In generic CPU CI the package is optional and this test
is skipped, while the localization-native lane installs the pinned source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sae_localize = pytest.importorskip("sae_localize")
from sae_localize import exact_scope_curve, omission_errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "saebase" / "research_lab"))

from backend.controlled_designs import (  # noqa: E402
    make_candidate,
    synthetic_reference_codes,
    to_saebase_design,
)


def test_original_localize_recovers_known_six_factor_dependence():
    payload = make_candidate("synthetic_known_factor_control")
    design = to_saebase_design(payload)
    assert design.n_states == 64 and design.n_factors == 6
    data = synthetic_reference_codes(payload)
    assert data["source_type"] == "synthetic_logic_not_real_sae"
    grid = np.asarray(data["feature_values"], dtype=np.float64)
    assert grid.shape == (64, 6)

    omitted, coefficients = omission_errors(grid, n_factors=6)
    assert omitted.shape == (64, 6)
    assert coefficients.shape == (64, 6)

    # Feature zero depends only on factor zero.
    assert omitted[1 << 0, 0] == pytest.approx(0.0, abs=1e-10)
    assert omitted[0, 0] == pytest.approx(0.25, abs=1e-10)
    # The d*e feature needs both factor three and four.
    assert omitted[(1 << 3) | (1 << 4), 2] == pytest.approx(0.0, abs=1e-10)
    assert omitted[1 << 3, 2] > 0
    assert omitted[1 << 4, 2] > 0
    # XOR of factors zero and two cannot be explained by either alone.
    assert omitted[(1 << 0) | (1 << 2), 4] == pytest.approx(0.0, abs=1e-10)
    assert omitted[1 << 0, 4] > 0
    assert omitted[1 << 2, 4] > 0

    result = exact_scope_curve(grid, n_factors=6)
    assert result.n_factors == 6
    assert result.n_features == 6
    assert len(result.curve) == 37
    assert result.total_variance > 0
    assert result.curve[0] == pytest.approx(result.total_variance, abs=1e-10)
    assert result.curve[-1] == pytest.approx(0.0, abs=1e-10)
    assert np.all(np.diff(result.curve) <= 1e-10)
    assert 0 < result.budget(0.1 * result.total_variance) <= 36
