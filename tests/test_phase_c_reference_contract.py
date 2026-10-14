"""Optional integration tests against untouched SAR/RSS paper sources.

Set SAEBASE_SAR_SOURCE and SAEBASE_RSS_SOURCE to the original checkout dirs.
These tests require no model downloads and are skipped for standalone installs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.integrations import evaluate_sar, evaluate_rss, collect_rss_features, load_paper_module


def _source(name: str) -> Path:
    path = os.getenv(f"SAEBASE_{name.upper()}_SOURCE")
    if not path:
        pytest.skip(f"set SAEBASE_{name.upper()}_SOURCE to the original {name}.py or checkout")
    return Path(path)


def test_original_sar_module_with_saebase_patching():
    paper = load_paper_module(_source("sar"), paper="sar")
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    result = evaluate_sar(model, sae, site=ActivationSite(0, "blocks.0.hook_resid_post"),
                          paper=paper, prompt="2 3 4", readout=lambda z: z[0] - z[1],
                          K=3, M=80, n_fit=55, seed=3)
    assert result.metrics["sar1"] == pytest.approx(0.2156495440166672, abs=1e-6)
    assert result.metrics["sar2"] < 1e-8
    assert result.metrics["rho2"] == pytest.approx(1., abs=1e-6)
    assert result.metrics["rank2"] > result.metrics["rank1"]
    assert np.array_equal(result.feature_ids, [1, 6, 0])


def test_original_rss_module_with_saebase_capture():
    paper = load_paper_module(_source("rss"), paper="rss")
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    result = evaluate_rss(
        collect_rss_features(model, sae, site=ActivationSite(0, "blocks.0.hook_resid_post"),
                             clean_prompt="1 2 3",
                             selection_variants=[f"1 2 {x}" for x in range(4, 9)],
                             held_out_variants=[f"1 2 {x}" for x in range(9, 14)]),
        paper=paper, rho=.8, kappa=.8, M=20, K=5)
    assert result.support.tolist() == [3, 0, 1, 4, 5]
    assert result.direction.tolist() == [1, 1, 1, 1, 1]
    assert result.mean_rss == pytest.approx(0.513370703458786, rel=1e-5)
    assert np.allclose(result.held_out_rates, 1.)
