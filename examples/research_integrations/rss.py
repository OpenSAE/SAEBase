"""RSS via SAEBase captures + the unchanged original RSS implementation.

Toy: python examples/research_integrations/rss.py --paper-dir /path/to/rss-repo
Real: python examples/research_integrations/rss.py --paper-dir /path/to/rss-repo --real

Selection and held-out variants are distinct. Toy output is synthetic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saebase.api import ActivationSite, ModelAdapter, load_registered
from saebase.integrations import collect_rss_features, evaluate_rss, load_paper_module


def main():
    parser = argparse.ArgumentParser(description="RSS with SAEBase model capture")
    parser.add_argument("--paper-dir", type=Path, required=True, help="Directory containing original rss.py")
    parser.add_argument("--real", action="store_true", help="Use real Pythia-70M (network)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    paper = load_paper_module(args.paper_dir, paper="rss")
    if args.real:
        pair = load_registered(sae_key="pythia70m_resid_sm", model_key="pythia70m",
                               layer=3, device=args.device)
        model, sae, site, identity = pair.model, pair.sae, pair.site, pair.identity
        clean = "Alice Emma Alice"
        selection = [f"{prefix} Alice Emma Alice" for prefix in
                     ("Sequence:", "Continue:", "Pattern:", "Complete:", "Input:")]
        held_out = [f"{prefix} Alice Emma Alice" for prefix in
                    ("Example:", "Finish:", "Repeat:", "Next:", "Text:")]
        # Hand-written exploratory prefixes, with last-token "Alice" aligned.
        # For research use the paper's controlled transformations and split.
    else:
        from examples._synthetic_common import ResearchToyModel, ResearchToySAE
        model = ModelAdapter(ResearchToyModel().eval())
        sae = ResearchToySAE().eval()
        site, identity, clean = ActivationSite(0, "blocks.0.hook_resid_post"), None, "1 2 3"
        selection = [f"1 2 {x}" for x in (4, 5, 6, 7, 8)]
        held_out = [f"1 2 {x}" for x in (9, 10, 11, 12, 13)]
    features = collect_rss_features(
        model, sae, site=site, clean_prompt=clean,
        selection_variants=selection, held_out_variants=held_out,
        identity=identity,
    )
    result = evaluate_rss(features, paper=paper, rho=.8, kappa=.8, tau=1e-5, M=20, K=5)
    print(json.dumps({"data_type": "real checkpoint exploratory variants" if args.real else "synthetic toy",
                      **result.summary()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
