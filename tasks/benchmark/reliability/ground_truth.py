"""Independent synthetic truths for SAE feature factors and dictionary recovery.

These fixtures are analytical controls, never claims about learned model SAE
features. Calibration chooses a feature matching; test never re-fits it.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class FactorTruth:
    scenario: str
    seed: int
    dictionary: np.ndarray  # [d_in, n_features]
    calibration: np.ndarray  # known factors [calibration_groups, n_features]
    test: np.ndarray  # unseen held-out factor samples
    calibration_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def __post_init__(self):
        if set(self.calibration_ids) & set(self.test_ids):
            raise ValueError("calibration/test group leakage")
        if (
            self.calibration.shape[1] != self.dictionary.shape[1]
            or self.test.shape[1] != self.dictionary.shape[1]
        ):
            raise ValueError("factor dictionary shape mismatch")
        if len(self.calibration) != len(self.calibration_ids) or len(self.test) != len(
            self.test_ids
        ):
            raise ValueError("factor group lineage mismatch")
        if not all(np.isfinite(x).all() for x in (self.dictionary, self.calibration, self.test)):
            raise ValueError("nonfinite known truth")


def make_factors(
    scenario: str, *, seed: int = 0, n_calibration: int = 96, n_test: int = 128
) -> FactorTruth:
    if scenario not in {"independent", "correlated", "hierarchical", "superposition"}:
        raise ValueError(f"unknown factor scenario {scenario!r}")
    if n_calibration < 16 or n_test < 16:
        raise ValueError("calibration/test require at least 16 independent groups")
    rng = np.random.default_rng(seed)
    n_features = 6
    d_in = 4 if scenario == "superposition" else 8
    D = rng.normal(size=(d_in, n_features))
    D /= np.linalg.norm(D, axis=0)

    def draw(n):
        sparsity = 0.15 if scenario == "superposition" else 0.42
        z = (rng.random((n, n_features)) < sparsity) * rng.normal(1.0, 0.35, size=(n, n_features))
        if scenario == "correlated":
            z[:, 1] = 0.8 * z[:, 0] + 0.2 * z[:, 1]
        elif scenario == "hierarchical":
            z[:, 1] *= z[:, 0] != 0
            z[:, 3] *= z[:, 2] != 0
        return z.astype(np.float64)

    return FactorTruth(
        scenario,
        seed,
        D,
        draw(n_calibration),
        draw(n_test),
        tuple(f"calibration-{i}" for i in range(n_calibration)),
        tuple(f"test-{i}" for i in range(n_test)),
    )


def _corr(left, right):
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[0] < 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("matching requires finite equal-width [samples, features] arrays")
    aa = a - a.mean(axis=0)
    bb = b - b.mean(axis=0)
    den = np.linalg.norm(aa, axis=0)
    other = np.linalg.norm(bb, axis=0)
    if np.any(den < 1e-12) or np.any(other < 1e-12):
        raise ValueError("undefined feature correlation for constant channel")
    return (aa.T @ bb) / np.outer(den, other)


def fit_matching(reference_calibration, proposed_calibration):
    """Exact maximum-absolute-correlation assignment on calibration only.

    Exponential DP intentionally supports only <= 10 synthetic factors.
    Real large SAEs should use their own auditable calibration matching.
    """
    scores = _corr(reference_calibration, proposed_calibration)
    n = scores.shape[0]
    if n > 10:
        raise ValueError("exact matching accepts at most ten synthetic features")

    @lru_cache(None)
    def solve(i, mask):
        if i == n:
            return 0.0, ()
        choices = []
        for j in range(n):
            if mask & (1 << j):
                continue
            value, tail = solve(i + 1, mask | (1 << j))
            choices.append((value + abs(float(scores[i, j])), (j,) + tail))
        return max(choices, key=lambda pair: (pair[0], tuple(-v for v in pair[1])))

    _, assignment = solve(0, 0)
    signs = tuple(int(np.sign(scores[i, j])) for i, j in enumerate(assignment))
    return {
        "permutation": list(assignment),
        "signs": list(signs),
        "matching_split": "calibration",
        "n_calibration": len(reference_calibration),
    }


def evaluate_matching(reference_test, proposed_test, matching):
    """Held-out performance: average sign-corrected Pearson correlation."""
    mapping = matching["permutation"]
    sign = np.asarray(matching["signs"], dtype=float)
    if len(mapping) != len(set(mapping)) or not np.isin(sign, (-1, 1)).all():
        raise ValueError("invalid frozen feature mapping")
    corr = _corr(np.asarray(reference_test), np.asarray(proposed_test)[:, mapping])
    values = sign * np.diag(corr)
    return {
        "score": float(np.mean(values)),
        "feature_correlations": values.tolist(),
        "n_test": len(reference_test),
        "matching_split": "calibration",
        "evaluation_split": "test",
    }


def reconstruction_mse(z, dictionary, reconstructed_codes=None, reconstructed_dictionary=None):
    z = np.asarray(z, dtype=float)
    D = np.asarray(dictionary, dtype=float)
    z_hat = z if reconstructed_codes is None else np.asarray(reconstructed_codes, dtype=float)
    D_hat = (
        D if reconstructed_dictionary is None else np.asarray(reconstructed_dictionary, dtype=float)
    )
    if z.shape != z_hat.shape or D.shape != D_hat.shape or D.shape[1] != z.shape[1]:
        raise ValueError("reconstruction factor/dictionary dimensions differ")
    return float(np.square(z @ D.T - z_hat @ D_hat.T).mean())
