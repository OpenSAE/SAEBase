"""Pinned *original* SAR/RSS parity on CPU-only measured-array interfaces."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "saebase" / "research_lab"))

from backend.core import digest, has_complete_identity, load_sites  # noqa: E402
from backend.methods import run_rss, run_sar  # noqa: E402


def _original(path: Path, name: str):
    if not path.is_file():
        pytest.skip("Pinned original research source was not checked out")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site():
    return next(site for site in load_sites() if has_complete_identity(site))


def _identity(site, protocol):
    return {
        "source_type": "real_checkpoint",
        "site_id": site["site_id"],
        "model_sha256": site["model_sha256"],
        "sae_sha256": site["sae_sha256"],
        "prompt_group_id": "synthetic:parity-test-only",
        "dataset_release": "synthetic_reference_not_public_model_evidence",
        "protocol_id": protocol,
    }


def test_original_sar_and_lab_adapter_are_numerically_identical():
    source = Path(os.environ.get("SAEBASE_SAR_SOURCE", "originals/sar/sar.py"))
    original = _original(source, "sar_pinned_reference")
    site = _site()
    masks, fit, held = original.coalitions(K=16, M=384, p=0.5, n_fit=256, seed=7)
    weights = np.linspace(-0.8, 1.1, 16)
    g = (
        masks.astype(np.float64) @ weights
        + 0.7 * (masks[:, 0] & masks[:, 1])
        - 0.2 * (masks[:, 3] & masks[:, 7])
    )
    g_full = float(weights.sum() + 0.7 - 0.2)
    expected = original.sar(g, masks, fit, held, g_full=g_full)
    measured = run_sar(
        {
            **_identity(site, "sar.original.k16.m384.fit256.v1"),
            "masks": masks.tolist(),
            "fit": fit.tolist(),
            "held": held.tolist(),
            "g": g.tolist(),
            "g_full": g_full,
        },
        original=source,
        expected_sha=digest(source),
    )
    mapped = {row["metric"]: row for row in measured["results"]}
    assert set(mapped) == set(expected)
    for key, reference in expected.items():
        if math.isnan(reference):
            assert mapped[key]["status"] == "undefined"
            assert mapped[key]["value"] is None
        else:
            assert mapped[key]["status"] == "completed"
            assert mapped[key]["value"] == pytest.approx(reference, abs=1e-12, rel=1e-12)


def test_original_rss_and_lab_adapter_have_same_support_and_persistence():
    source = Path(os.environ.get("SAEBASE_RSS_SOURCE", "originals/rss/rss.py"))
    original = _original(source, "rss_pinned_reference")
    site = _site()
    delta = np.zeros((5, 28), dtype=np.float64)
    held = np.zeros((10, 28), dtype=np.float64)
    delta[:, 4] = [0.8, 0.9, 1.0, 1.1, 0.9]
    delta[:, 17] = [-0.4, -0.5, -0.7, -0.6, -0.4]
    held[:8, 4] = 1.0
    held[8:, 4] = -1.0
    held[:7, 17] = -1.0
    held[7:, 17] = 1.0
    support, direction, mean_rss = original.rss(delta)
    expected_rate = original.held_out_rate(held, support, direction)
    measured = run_rss(
        {
            **_identity(site, "rss.transform.v1"),
            "selection_variant_ids": [str(i) for i in range(5)],
            "held_variant_ids": [str(i) for i in range(5, 15)],
            "delta_selection": delta.tolist(),
            "delta_held": held.tolist(),
        },
        original=source,
        expected_sha=digest(source),
    )
    assert measured["support_ids"] == support.tolist()
    assert measured["directions"] == direction.tolist()
    mapped = {row["metric"]: row for row in measured["results"]}
    if math.isnan(mean_rss):
        assert mapped["mean_rss"]["status"] == "undefined"
    else:
        assert mapped["mean_rss"]["value"] == pytest.approx(mean_rss)
    expected_mean = float(expected_rate.mean()) if len(expected_rate) else None
    if expected_mean is None:
        assert mapped["held_out_direction_rate"]["status"] == "undefined"
    else:
        assert mapped["held_out_direction_rate"]["value"] == pytest.approx(expected_mean)
