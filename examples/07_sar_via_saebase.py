"""SAR via SAEBase + the original paper estimator (offline toy or real checkpoint).

Toy:  python examples/07_sar_via_saebase.py --paper-dir /path/to/sar-repo
Real: python examples/07_sar_via_saebase.py --paper-dir /path/to/sar-repo --real

Real run downloads weights. The toy example is synthetic, not a paper result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saebase.api import ActivationSite, ModelAdapter, load_registered
from saebase.integrations import evaluate_sar, load_paper_module


def main() -> None:
    parser = argparse.ArgumentParser(description="SAR measured with SAEBase interventions")
    parser.add_argument("--paper-dir", type=Path, required=True, help="Directory containing original sar.py")
    parser.add_argument("--real", action="store_true", help="Use real Pythia-70M + published SAE (network)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    paper = load_paper_module(args.paper_dir, paper="sar")
    if args.real:
        pair = load_registered(sae_key="pythia70m_resid_sm", model_key="pythia70m",
                               layer=3, device=args.device)
        model, sae, site, identity = pair.model, pair.sae, pair.site, pair.identity
        prompt = "Alice Emma Alice"
        ids = model.model.to_single_token(" Emma"), model.model.to_single_token(" Rachel")
        readout = lambda logits: logits[ids[0]] - logits[ids[1]]
    else:
        from examples._synthetic_common import ResearchToyModel, ResearchToySAE
        model = ModelAdapter(ResearchToyModel().eval())
        sae = ResearchToySAE().eval()
        site, identity, prompt = ActivationSite(0, "blocks.0.hook_resid_post"), None, "2 3 4"
        readout = lambda logits: logits[0] - logits[1]
    result = evaluate_sar(model, sae, site=site, paper=paper, readout=readout,
                          prompt=prompt, K=3, M=80, n_fit=55, seed=3, identity=identity)
    print(json.dumps({"data_type": "real checkpoint" if args.real else "synthetic toy",
                      **result.summary()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
