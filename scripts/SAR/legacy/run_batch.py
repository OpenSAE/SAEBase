"""Run the requested five QA datasets through the existing Capture matrix.

Scope: this recorded experiment selects 100 reference questions and ten SAR
questions per dataset, reusing one Llama 3.1 8B / layer-21 SAE session.
"""

import json
from pathlib import Path

from tasks.capture.matrix import CaptureMatrix


ROOT = Path("/workspace/SAEBase")
REPORT = Path(__file__).resolve().parent


def main():
    plan = json.loads((REPORT / "plan.json").read_text())
    matrix = CaptureMatrix(
        REPORT / "matrix.yaml",
        root=ROOT,
        tag="qa5_L21_ref100_freq50_sar_20261114",
        mode="sar",
        report=REPORT,
    )
    compose = matrix.capture_config

    def selected_config(alias, model, sae, layer):
        cfg = compose(alias, model, sae, layer)
        cfg.dataset.sample_count = 10
        cfg.dataset.prompt_file = plan[alias]["prompt_file"]
        return cfg

    matrix.capture_config = selected_config
    matrix.child("llama31_8b")
    return matrix.summarize()


if __name__ == "__main__":
    raise SystemExit(main())
