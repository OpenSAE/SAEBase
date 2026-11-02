"""Optional two-checkpoint SAEBase public API integration smoke (downloads weights).

python saebase/examples/quickstart/model_checkpoint_probe.py --model gemma270m --sae gemma270m_scope_resid --layer 5
python saebase/examples/quickstart/model_checkpoint_probe.py --model pythia70m --sae pythia70m_resid_sm --layer 3
Both choices are configured; real model compatibility must be checked on the
machine and in the exact environment used by the researcher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from saebase.api import capture_features, evaluate_edit, load_registered
from saebase.api.sae_profiles import get_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gemma270m", "pythia70m"), default="gemma270m")
    parser.add_argument("--sae", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    pairs = {
        "gemma270m": ("gemma270m_scope_resid", 5, "The capital of France is"),
        "pythia70m": ("pythia70m_resid_sm", 3, "The capital of France is"),
    }
    sae_key, layer, prompt = pairs[args.model]
    sae_key, layer = args.sae or sae_key, args.layer if args.layer is not None else layer
    profile = get_profile(sae_key)
    if args.model not in profile.allowed_model_keys() or layer not in profile.layers:
        parser.error("Model, SAE and layer do not match a registered combination")
    pair = load_registered(model_key=args.model, sae_key=sae_key, layer=layer, device=args.device)
    tokens = pair.model.tokens(prompt=prompt)
    snap = capture_features(pair.model, pair.sae, tokens=tokens, site=pair.site)
    rank = int(snap.codes.abs().argmax())
    edit = evaluate_edit(
        pair.model,
        pair.sae,
        site=pair.site,
        tokens=tokens,
        replacements={rank: 0.0},
        verify_identity=True,
    )
    print(
        json.dumps(
            {
                "status": "executed",
                "model": args.model,
                "identity": pair.identity.to_dict(),
                "captured_feature_count": int(snap.codes.numel()),
                "removed_feature": rank,
                "logit_delta_norm": float(edit.logit_delta.norm()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
