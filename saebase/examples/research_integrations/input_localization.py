"""Input Localization through SAEBase capture and the original estimator.

Install the original method first:
    pip install git+https://github.com/OpenSAE/SAE-Input-Localization.git

This example uses a deterministic synthetic model/SAE, not paper results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saebase.api import ActivationSite, ModelAdapter
from saebase.integrations import collect_localization_grid, evaluate_localization
from saebase.saebase.examples._synthetic_common import ResearchToyModel, ResearchToySAE


def main() -> None:
    model = ModelAdapter(ResearchToyModel().eval())
    sae = ResearchToySAE().eval()
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    # Enumerated complete 3-factor Boolean grid. Each factor controls one token.
    prompts = {
        state: f"{1 + (state & 1)} {3 + ((state >> 1) & 1)} {5 + ((state >> 2) & 1)}"
        for state in range(8)
    }
    grid = collect_localization_grid(
        model, sae, site=site, prompts_by_state=prompts, n_factors=3
    )
    result = evaluate_localization(grid)
    print(json.dumps({"data_type": "synthetic demonstration",
                      **result.summary(),
                      "scope_curve": result.curve.tolist(),
                      "feature_scope_masks": result.assignment.tolist()},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
