"""A failed content gate does not commit intervention rows."""

from __future__ import annotations

import pytest
import torch

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.interventions import (
    BatchEngine,
    Condition,
    ExperimentIdentity,
    MeasurementSample,
    ResultWriter,
)
from saebase.store import (
    CheckpointLeaseError,
    SampleRecord,
    TokenSelection,
    VerifiedCheckpointSession,
    sha256_state_dict,
)


def test_mutated_weights_fail_the_release_gate_before_commit(tmp_path):
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    identity = ExperimentIdentity(
        sha256_state_dict(model.model),
        sha256_state_dict(sae),
        site,
        model_key="toy",
        sae_key="toy",
        backend=model.backend,
    )
    session = VerifiedCheckpointSession.mint(model.model, sae)
    with torch.no_grad():
        next(model.model.parameters()).data.add_(1)
    sample = MeasurementSample(
        SampleRecord(
            "s", (1, 2, 3), selection=TokenSelection("last_valid"), prompt_group_id="g", split="t"
        ),
        target_token_id=1,
    )
    engine = BatchEngine(model, sae, identity, pad_token_id=0, identity_lease=session)
    with ResultWriter(tmp_path, identity) as writer:
        with pytest.raises(CheckpointLeaseError, match="bytes differ"):
            engine.run([sample], [Condition("identity", "identity")], writer, verify_identity=False)
    assert list(tmp_path.glob("shard-*.jsonl")) == []
