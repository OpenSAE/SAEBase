"""Regression checks for M0 stored activation probes used by real M2 runs."""
import numpy as np
import pytest

from saebase.tools.m2_real_smoke import compare_m0_activation_probe


def test_m0_probe_combines_absolute_and_relative_tolerances():
    baseline = [1.0, -2.0, 0.0]
    current = [1.00015, -2.00025, 0.00008]
    # Coordinate tolerances: 2e-4, 3e-4, 1e-4.
    assert 0.00024 < compare_m0_activation_probe(
        baseline, current, atol=1e-4, rtol=1e-4) < 0.00026


def test_m0_probe_rejects_excessive_or_nonfinite_drift():
    with pytest.raises(RuntimeError, match="coordinate 0"):
        compare_m0_activation_probe([0.0, 100.0], [0.00010001, 100.0],
                                    atol=1e-4, rtol=1e-4)
    with pytest.raises(RuntimeError, match="non-finite"):
        compare_m0_activation_probe([1.0], [float("nan")],
                                    atol=1e-4, rtol=1e-4)
    with pytest.raises(RuntimeError, match="dimensions"):
        compare_m0_activation_probe([1.0], [1.0, 2.0],
                                    atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="tolerances"):
        compare_m0_activation_probe([1.0], [1.0], atol=-1e-4, rtol=0.0)
