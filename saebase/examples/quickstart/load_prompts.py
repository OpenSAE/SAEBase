#!/usr/bin/env python3
"""Example: load prepared prompt artifacts for a registered dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from saebase.datasets.artifact_layout import DEFAULT_DATA_ROOT, prompts_path
from saebase.datasets.prompt_store import load_prompts


def main() -> None:
    behavior = "SimpleCopying"
    model_key = "pythia70m"
    path = prompts_path(behavior, model_key)
    print(f"DEFAULT_DATA_ROOT={DEFAULT_DATA_ROOT}")
    print(f"expected prompts: {path}")
    if not path.is_file():
        print(
            "No artifacts yet. Prepare with:\n"
            "  python3 saebase/tools/sae_prepare_prompts.py "
            f"--model-key {model_key} --behavior {behavior}"
        )
        return
    rows = load_prompts(behavior, model_key)
    print(f"loaded {len(rows)} prompts; first={rows[0].get('prompt', '')[:80]!r}")


if __name__ == "__main__":
    main()
