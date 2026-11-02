"""List SAE profiles from the registry."""

from __future__ import annotations

import argparse

from saebase.api.sae_profiles import list_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or True:
        for p in list_profiles():
            print(
                f"{p.key}: model={p.model_key} release={p.sae_release} "
                f"layers={list(p.layers)} hook={p.hook_kind} loader={p.loader}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
