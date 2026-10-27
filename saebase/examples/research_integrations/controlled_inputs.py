"""Full offline controlled-input workflow with reusable SAEBase caches.

    python saebase/examples/research_integrations/controlled_inputs.py
    python saebase/examples/research_integrations/controlled_inputs.py --evaluate

The optional --evaluate mode requires the original sae_localize package:
    pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git
All data in this example is synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.data import (
    BinaryFactor, ControlledInputDesign, collect_controlled_input_activations,
)
from saebase.integrations import localization_grid_from_code_store, evaluate_localization
from saebase.store import (
    ActivationIdentity, ActivationWriter, CodeIdentity, encode_from_store, sha256_state_dict,
)


class FactorialToyModel(ResearchToyModel):
    """A deterministic model representation that uses all three input factors."""

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != 3:
            raise ValueError("expected three independently controlled input tokens")
        first = (tokens[:, 0] - 1).float()
        second = (tokens[:, 1] - 3).float()
        third = (tokens[:, 2] - 5).float()
        h = torch.stack((first, second, third, first * second), dim=-1)
        return h[:, None, :].expand(-1, tokens.shape[1], -1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the original Localize estimator (optional dependency)")
    args = parser.parse_args()

    factors = (
        BinaryFactor("first", ("1", "2")),
        BinaryFactor("second", ("3", "4")),
        BinaryFactor("third", ("5", "6")),
    )
    identity_design = ControlledInputDesign.from_renderer(
        factors=factors, name="three-factor-toy",
        render=lambda x: f'{x["first"]} {x["second"]} {x["third"]}',
    )
    # new state mask -> old state mask. The same eight prompt/SAE pairs are reused.
    xor_map = {i: ((i & 1) ^ ((i >> 1) & 1)) | (i & ~1) for i in range(8)}
    mixed_design = identity_design.reindex(xor_map, name="three-factor-xor")

    model = ModelAdapter(FactorialToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    activation_id = ActivationIdentity(
        model_key="controlled-toy", model_state_sha256=sha256_state_dict(model.model),
        tokenizer_sha256=hashlib.sha256(b"FactorialToyModel.to_tokens.v1").hexdigest(),
        hook_name=site.hook_name, layer=site.layer,
    )
    sae_id = CodeIdentity(
        "research-toy", sha256_state_dict(sae),
        d_in=sae.cfg.d_in, d_sae=sae.cfg.d_sae,
    )

    with TemporaryDirectory() as tmp:
        with ActivationWriter(Path(tmp), activation_id, width=sae.cfg.d_in) as writer:
            identity_run = collect_controlled_input_activations(
                model, identity_design, writer, site=site
            )
            mixed_run = collect_controlled_input_activations(
                model, mixed_design, writer, site=site
            )
            activation_path = writer.path

        code_path, code_stats = encode_from_store(activation_path, sae, sae_id)
        original = localization_grid_from_code_store(
            code_path, manifest=identity_run.manifest_path, site=site
        )
        mixed = localization_grid_from_code_store(
            code_path, manifest=mixed_run.manifest_path, site=site
        )

        output = {
            "data_type": "synthetic demonstration",
            "input_states": identity_design.n_states,
            "original_activation_samples": identity_run.stats.collected,
            "reindexed_activation_samples_reused": mixed_run.stats.reused,
            "encoded_code_samples": code_stats.collected,
            "same_SAE_code_rows_different_input_order": (
                sorted(original.state_sample_ids) == sorted(mixed.state_sample_ids)
            ),
            "original_design_key": original.design_key,
            "reindexed_design_key": mixed.design_key,
        }
        if args.evaluate:
            output["identity_budget"] = evaluate_localization(original).budget
            output["xor_budget"] = evaluate_localization(mixed).budget
        print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
