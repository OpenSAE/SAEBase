"""Real checkpoint Phase B smoke, needs model/SAE dependencies and downloads.

Run: python examples/06_public_api_pythia.py
On real checkpoints this checks a baseline identity patch, SAE feature removal,
and prints a reproducible model/SAE record and observed logit-change summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saebase.api import capture_features, coalition_replacements, evaluate_edit, load_registered


def main():
    bundle = load_registered(
        model_key="pythia70m", sae_key="pythia70m_resid_sm", layer=3,
        device="cpu",
    )
    tokens = bundle.model.tokens(prompt="Alice Emma Alice")
    site = bundle.site
    snapshot = capture_features(bundle.model, bundle.sae, site=site, tokens=tokens)
    top = snapshot.codes.abs().topk(min(4, snapshot.codes.numel())).indices.tolist()
    drop = coalition_replacements(
        snapshot.codes, selected_indices=top, retained_indices=top[:2]
    )
    result = evaluate_edit(
        bundle.model, bundle.sae, tokens=tokens, site=site,
        replacements=drop,
    )
    print(json.dumps({
        "identity": bundle.identity.to_dict(),
        "prompt": "Alice Emma Alice", "site": site.hook_name,
        "selected_features": top, "removed_features": sorted(drop),
        "logit_change_norm": float(result.logit_delta.float().norm()),
    }, indent=2))


if __name__ == "__main__":
    main()
