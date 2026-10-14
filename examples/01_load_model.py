#!/usr/bin/env python3
"""Example: construct a HookedTransformer model handle (needs CUDA + HF cache)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.pythia70m.pythia70m import load_model


def main() -> None:
    handle = load_model()
    print(f"key={handle.key} loaded={handle.loaded_name} family={handle.family}")
    print(f"n_layers={handle.model.cfg.n_layers}")


if __name__ == "__main__":
    main()
