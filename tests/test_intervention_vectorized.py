"""Edited activations keep the residual formula and skip a repeated original decode."""

from __future__ import annotations

import torch

from saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter, capture_features
from saebase.api.features import _decode
from saebase.interventions import Condition, apply_condition


def test_edits_match_two_decode_oracle_and_identity_skips_decode():
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    snapshot = capture_features(
        model, sae, site=ActivationSite(0, "blocks.0.hook_resid_post"), prompt="2 3 4"
    )
    calls = {"n": 0}
    original = sae.decode

    def counted(codes: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return original(codes)

    sae.decode = counted
    identity = apply_condition(sae, snapshot, Condition("i", "identity"))
    assert calls["n"] == 0
    torch.testing.assert_close(identity.activation, snapshot.activation)
    reconstruction = apply_condition(sae, snapshot, Condition("r", "reconstruction"))
    assert calls["n"] == 0
    torch.testing.assert_close(reconstruction.activation, snapshot.reconstruction)
    for condition in (
        Condition("ablate", "ablate", (0,)),
        Condition("scale", "scale", (0, 1), 0.5),
        Condition("clamp", "clamp", (0,), 1.0),
        Condition("add", "add", (0, 2), 0.1),
        Condition("coalition", "coalition", (0, 1), retained_ids=(1,)),
    ):
        calls["n"] = 0
        patch = apply_condition(sae, snapshot, condition)
        assert calls["n"] == 2
        sae.decode = original
        before = _decode(sae, snapshot.codes, 4)
        after = _decode(sae, patch.codes, 4)
        sae.decode = counted
        expected = snapshot.activation + (after - before).to(snapshot.activation.dtype)
        torch.testing.assert_close(patch.activation, expected)
