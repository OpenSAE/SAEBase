"""Compare recorded activation probes using their numerical tolerances."""

import numpy as np


def compare_reference_activation_probe(reference, observed, *, atol: float, rtol: float) -> float:
    """Compare an activation probe with the saved reference using recorded tolerances.

    The comparison uses numpy-style |observed-reference| <= atol + rtol*|reference|.
    The returned max *absolute* difference is a diagnostic, not the acceptance
    criterion. Reject changed probe geometry or non-finite values explicitly.
    """
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("reference numerical tolerances must be finite and nonnegative")
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(observed, dtype=np.float64)
    if expected.ndim != 1 or expected.size == 0 or expected.shape != actual.shape:
        raise RuntimeError("reference activation probe dimensions differ")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError("reference activation probe contains non-finite values")
    delta = np.abs(actual - expected)
    allowed = atol + rtol * np.abs(expected)
    if np.any(delta > allowed):
        pos = int(np.argmax(delta > allowed))
        raise RuntimeError(
            f"reference activation probe numerical drift at coordinate {pos}: "
            f"abs_error={delta[pos]:.9g}, allowed={allowed[pos]:.9g}, "
            f"reference={expected[pos]:.9g}"
        )
    return float(delta.max())
