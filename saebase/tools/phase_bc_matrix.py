"""Inspect the Stage B/C model, SAE and multilayer checkpoint matrix offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.benchmark.coverage_contract import load_coverage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/coverage/stage_bc.json"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    plan = load_coverage(args.config).plan()
    if args.json:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    print(
        f"Stage B/C: {plan['core_models']} core models, "
        f"{plan['extra_architectures']} architecture extensions, "
        f"{plan['planned_checkpoints']} registered layer checkpoints"
    )
    print(
        f"CPU queue: {plan['planned_cpu_sae_probes']} SAE-only checks; "
        f"{plan['planned_cpu_model_pair_probes']} small model pairs"
    )
    print(
        "MODEL                     SAE / ARCHITECTURE                         LAYERS          CPU"
    )
    for entry in plan["entries"]:
        layers = ",".join(str(check["layer"]) for check in entry["checkpoints"])
        cpu_layers = (
            ",".join(
                str(check["layer"])
                for check in entry["checkpoints"]
                if check["planned_cpu_sae_probe"]
            )
            or "—"
        )
        print(f"{entry['model_key']:25} {entry['sae_key'][:42]:42} {layers:15} {cpu_layers}")
    print("All entries are registry plans until real checkpoint artifacts prove otherwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
