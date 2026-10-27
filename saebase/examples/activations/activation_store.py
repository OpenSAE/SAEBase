"""Offline activation-store demonstration. All values are from a small synthetic model."""

from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE
from saebase.api import ActivationSite, ModelAdapter
from saebase.store import (
    ActivationIdentity,
    ActivationWriter,
    CodeIdentity,
    SampleRecord,
    TokenSelection,
    ShardedReader,
    collect_activations,
    encode_from_store,
)
from saebase.tools.m0_reference_smoke import _sha_weights


def main():
    torch.manual_seed(0)
    model = ModelAdapter(ResearchToyModel().eval())
    sae1, sae2 = ResearchToySAE().eval(), ResearchToySAE().eval()
    with torch.no_grad():
        sae2.bias.add_(0.125)  # distinct checkpoint; same width
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    activation_id = ActivationIdentity(
        model_key="synthetic-toy",
        model_state_sha256=_sha_weights(model.model),
        tokenizer_sha256="b" * 64,
        hook_name=site.hook_name,
        layer=site.layer,
        execution={"backend": "synthetic", "seed": 0},
    )
    items = [
        SampleRecord("clean-A", (1, 2, 3), prompt_group_id="A"),
        SampleRecord(
            "variant-A",
            (4, 5, 6),
            prompt_group_id="A",
            variant_role="selection",
            selection=TokenSelection("all_valid"),
        ),
    ]
    with tempfile.TemporaryDirectory() as directory:
        with ActivationWriter(directory, activation_id, width=4, shard_target_bytes=256) as w:
            stats = collect_activations(model, items, w, site=site)
            raw = w.path
        for label, sae in (("A", sae1), ("B", sae2)):
            code_id = CodeIdentity(label, _sha_weights(sae), 4, 7)
            path, result = encode_from_store(
                raw, sae, code_id, layout="csr", max_rows=8, shard_target_bytes=256
            )
            reader = ShardedReader(path)
            assert reader.sample_count == 2
            print(
                f"SAE {label}: {result.collected} encoded, {reader.row_count} positions, {len(reader.manifest['shards'])} shard(s)"
            )
        assert ShardedReader(raw).sample_count == stats.collected == 2
        print(f"Activation Store: {stats.collected} samples, no model rerun for second SAE")
        sample = ShardedReader(raw).get("variant-A")
        print(
            "positions:",
            sample.positions,
            "first activation:",
            np.array(sample.values[0]).round(3).tolist(),
        )


if __name__ == "__main__":
    main()
