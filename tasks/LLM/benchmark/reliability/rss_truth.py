"""Selection/held-out known-support RSS fixtures and false-positive controls."""

from __future__ import annotations
import numpy as np


def fixture(*, seed: int = 11, n_select: int = 40, n_held_out: int = 40):
    rng = np.random.default_rng(seed)
    if n_select < 5 or n_held_out < 5:
        raise ValueError("need independent selection and held-out variants")
    K = 12
    support = np.asarray([1, 4, 9], dtype=int)
    direction = np.asarray([1, -1, 1], dtype=int)
    scale = np.asarray([2.0, 1.6, 1.2])
    selection = rng.normal(0, 0.005, (n_select, K))
    held = rng.normal(0, 0.005, (n_held_out, K))
    selection[:, support] += scale * direction
    held[:, support] += scale * direction
    return {
        "selection": selection,
        "held_out": held,
        "support": support,
        "direction": direction,
        "seed": seed,
        "selection_ids": tuple(f"selection-{i}" for i in range(n_select)),
        "held_out_ids": tuple(f"held_out-{i}" for i in range(n_held_out)),
    }


def verify_original_rss(paper, *, seed: int = 11) -> dict:
    if not callable(getattr(paper, "rss", None)) or not callable(
        getattr(paper, "held_out_rate", None)
    ):
        raise TypeError("trusted original RSS module requires rss and held_out_rate")
    f = fixture(seed=seed)
    if set(f["selection_ids"]) & set(f["held_out_ids"]):
        raise ValueError("RSS selection/held-out leakage")
    support, direction, mean = paper.rss(f["selection"], rho=0.8, kappa=0.9, tau=0.1, M=12, K=3)
    rates = paper.held_out_rate(f["held_out"], support, direction, tau=0.1)
    passed = (
        np.array_equal(support, f["support"])
        and np.array_equal(direction, f["direction"])
        and bool(np.allclose(rates, 1.0))
        and np.isfinite(mean)
    )
    # Modify held-out only; selection-derived support/direction may not change.
    flipped = f["held_out"].copy()
    flipped[:, f["support"]] *= -1
    flipped_rates = paper.held_out_rate(flipped, support, direction, tau=0.1)
    passed = passed and bool(np.allclose(flipped_rates, 0.0))
    empty, empty_direction, undefined = paper.rss(
        np.zeros_like(f["selection"]), rho=0.8, kappa=0.9, tau=0.1, M=12, K=3
    )
    passed = passed and len(empty) == 0 and len(empty_direction) == 0 and np.isnan(undefined)
    return {
        "status": "passed" if passed else "failed",
        "source_estimator": "original RSS / held_out_rate",
        "expected_support": f["support"].tolist(),
        "selected_support": support.tolist(),
        "expected_direction": f["direction"].tolist(),
        "selected_direction": direction.tolist(),
        "mean_rss": float(mean) if np.isfinite(mean) else None,
        "held_out_persistence": rates.tolist(),
        "flipped_held_out_persistence": flipped_rates.tolist(),
        "n_selection": len(f["selection"]),
        "n_held_out": len(f["held_out"]),
        "empty_support_status": "undefined" if np.isnan(undefined) else "invalid",
        "selection_seed": seed,
    }
