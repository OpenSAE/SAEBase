"""Prepare a generic study from its existing selected Dataset and SAE configs.

Static setup is not a scientific pilot. Real smoke is opt-in separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import setup_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--max-sites", type=int, default=10)
    p.add_argument("--total-samples", type=int, default=0)
    p.add_argument("--pilot-timings", type=Path)
    p.add_argument("--cache", type=Path, default=Path.cwd())
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--probe-gpu", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    datasets = json.loads(args.datasets.read_text(encoding="utf-8"))
    site_data = json.loads(args.sites.read_text(encoding="utf-8"))
    if not isinstance(datasets.get("dataset_ids"), list) or not isinstance(
        site_data.get("models"), list
    ):
        p.error("Expected existing dataset_ids and models selectors")
    sites = [
        {"model_key": model["model_key"], "sae_key": model["sae_key"], "layer": layer}
        for model in site_data["models"]
        for layer in model["layers"]
    ]
    timings = (
        json.loads(args.pilot_timings.read_text(encoding="utf-8")) if args.pilot_timings else []
    )
    report = setup_report(
        dataset_ids=datasets["dataset_ids"],
        sites=sites,
        representative_limit=args.max_sites,
        require_cuda=args.require_cuda,
        cache_path=args.cache,
        real_gpu_probe=args.probe_gpu,
        pilot_rows=timings,
        total_samples=args.total_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Distinct runs must use distinct output files; never overwrite evidence.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"status": report["status"], "receipt": str(args.output)}))
    return 0 if report["status"] != "BLOCKED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
