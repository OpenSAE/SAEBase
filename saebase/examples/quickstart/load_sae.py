#!/usr/bin/env python3
"""Example: load an SAE profile at one layer (needs CUDA + SAE weights)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from saebase.sae.loading import load_sae_for_profile
from saebase.sae.registry import get_profile


def main() -> None:
    sae_key = "pythia70m_resid_sm"
    layer = 3
    profile = get_profile(sae_key)
    print(f"profile={profile.key} hook={profile.hook_name(layer)} id={profile.sae_id(layer)}")
    sae, meta = load_sae_for_profile(sae_key, layer=layer)
    print(f"meta hook={meta.get('hook_name')} loader={meta.get('loader')}")
    del sae


if __name__ == "__main__":
    main()
