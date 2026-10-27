"""Deterministic CPU feature-intervention demo: real batch hooks on toy models, no checkpoints.

Run from repository root:
    python saebase/examples/interventions/batch_interventions.py
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict
from saebase.interventions import (BatchEngine, Condition, ExperimentIdentity,
                                   MeasurementSample, ResultWriter, GenerationSpec,
                                   RunBudget, dry_run, read_results, run_steering)


def main() -> None:
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    identity = ExperimentIdentity(
        model_state_sha256=sha256_state_dict(model.model),
        sae_state_sha256=sha256_state_dict(sae), site=site,
        model_key="synthetic", sae_key="synthetic", backend=model.backend,
    )
    samples = [MeasurementSample(SampleRecord(
        sample_id=f"toy-{i}", token_ids=tuple(range(1, length + 1)),
        selection=TokenSelection("last_valid"), split="demo"),
        target_token_id=1, reference_token_id=0)
        for i, length in enumerate((3, 5, 4))]
    conditions = [Condition("identity", "identity"),
                  Condition("feature_0_ablation", "ablate", (0,)),
                  Condition("half_strength", "scale", (0, 1), value=0.5),
                  Condition("clamp_feature_0", "clamp", (0,), value=1.0),
                  Condition("add_feature_0", "add", (0,), value=0.5),
                  Condition("keep_feature_1", "coalition", (0,1),retained_ids=(1,)),
                  Condition("full_reconstruction", "reconstruction")]
    print("M2 offline dry run:")
    print(json.dumps(dry_run(samples, conditions), indent=2))
    with tempfile.TemporaryDirectory() as d:
        run_path = Path(d)/"measurements"
        with ResultWriter(run_path,identity) as writer:
            engine = BatchEngine(model,sae,identity,pad_token_id=0)
            stats = engine.run(samples,conditions,writer, sample_batch=3, microbatch=4)
            reused = engine.run(samples,conditions,writer, sample_batch=3, microbatch=4)
        rows=read_results(run_path,identity)
        with ResultWriter(Path(d)/"steering",identity) as writer:
            steering=run_steering(model,sae,identity,samples[:1],conditions[1],
                                 GenerationSpec(max_new_tokens=3),writer,
                                 budget=RunBudget(max_generated_tokens=10))
        assert len(rows)==len(samples)*len(conditions)
        assert reused.measured==0 and reused.reused==len(rows)
        print(json.dumps({"dataset":"synthetic", "stats":asdict(stats),
                          "resumed":reused.reused,"interventions":len(rows),
                          "first_metrics":rows[0]["metrics"], "steering":steering},indent=2))

if __name__ == "__main__":
    main()
