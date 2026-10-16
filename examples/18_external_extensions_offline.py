"""Synthetic third-party SAE/model/metric demo; no checkpoint downloads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from _synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite
from saebase.benchmarks import BenchmarkConfig, MetricSpec, build_plan
from saebase.benchmarks.runner import execute
from saebase.extensions import (
    ExtensionBundle,
    ExternalMetric,
    ExternalModel,
    ExternalObservation,
    ExternalSAE,
    HookSpec,
    load_external_pair,
    register_extension,
)

MODEL = "ext.opensae_demo.model"
SAE = "ext.opensae_demo.sae"
METRIC = "ext.opensae_demo.score"
SITE = ActivationSite(0, "blocks.0.hook_resid_post")


def evaluate(context):
    """Tiny demonstration, not a scientific metric or a checkpoint benchmark."""
    assert context.target.site.hook_name == SITE.hook_name
    return ExternalObservation(value=0.5, n_valid=1)


def make_bundle() -> ExtensionBundle:
    return ExtensionBundle(
        name="ext.opensae_demo.bundle",
        models=(
            ExternalModel(
                key=MODEL,
                source="https://github.com/OpenSAE/SAEBase",
                revision="synthetic-demo-v1",
                hooks=(HookSpec(SITE, 4),),
                factory=lambda device: ResearchToyModel().eval(),
            ),
        ),
        saes=(
            ExternalSAE(
                key=SAE,
                model_key=MODEL,
                source="https://github.com/OpenSAE/SAEBase",
                revision="synthetic-demo-v1",
                site=SITE,
                d_in=4,
                d_sae=7,
                architecture="demo_linear",
                normalization="none",
                dtype="float32",
                factory=lambda device: ResearchToySAE().eval(),
            ),
        ),
        metrics=(
            ExternalMetric(
                spec=MetricSpec(
                    METRIC,
                    "1",
                    ("external_pair",),
                    "dataset",
                    direction="contextual",
                    protocol_id="ext.opensae_demo.v1",
                ),
                source="https://github.com/OpenSAE/SAEBase",
                revision="synthetic-demo-v1",
                source_file=Path(__file__).resolve(),
                evaluator=evaluate,
            ),
        ),
    )


def main() -> int:
    register_extension(make_bundle())
    pair = load_external_pair(
        model_key=MODEL,
        sae_key=SAE,
        layer=0,
        probe_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )
    print("Adapter probe:", pair.verification)
    print("SAE hook:", pair.identity.hook_name)
    with tempfile.TemporaryDirectory(prefix="saebase-external-demo-") as run:
        config = BenchmarkConfig.from_dict(
            {
                "schema": "saebase.benchmark.config.v1",
                "name": "external-demo",
                "targets": [
                    {
                        "model_key": MODEL,
                        "sae_keys": [SAE],
                        "layer": 0,
                        "hook_name": SITE.hook_name,
                    }
                ],
                "dataset": {
                    "id": "synthetic",
                    "samples": [{"sample_id": "demo1", "token_ids": [1, 2, 3], "split": "test"}],
                },
                "metrics": [METRIC],
                "sources": {METRIC: str(Path(__file__).resolve())},
                "options": {"tokenizer_sha256": "a" * 64},
            }
        )
        plan = build_plan(config)
        print("Planned metric evaluations:", plan["metric_jobs"])
        # The real=True flag enables code execution; this is still toy data.
        result = execute(config, run, real=True)
        assert result[0]["status"] == "completed"
        print("Demonstration metric:", result[0]["value"])
        print("Original source SHA256:", result[0]["source_sha256"])
    print("Synthetic demo completed (no scientific checkpoint acceptance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
