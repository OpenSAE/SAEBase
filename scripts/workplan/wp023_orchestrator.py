"""Independent Precheck target discovery; never claim execution without receipts.

This entrypoint is an offline, bounded preparation unit. Native execution,
resume, audited delivery and Agent Ready have separate implementation gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from tasks.precheck.compatibility_checks import declared_site, plan_compatibility

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,63}\Z")


def inventory(root: Path = ROOT) -> dict[str, Any]:
    """Discover currently active Registry components and their owned SAE sites."""
    from omegaconf import OmegaConf
    from saebase.api.sae_profiles import get_profile
    from saebase.registry import list_registered_dataset_keys, list_registered_model_keys

    datasets = sorted(list_registered_dataset_keys())
    models = sorted(list_registered_model_keys())
    if not datasets or not models:
        raise ValueError("Active Registry must declare at least one Dataset and Model")
    selected = []
    excluded = []
    for model in models:
        config_path = root / "conf" / "model" / f"{model}.yaml"
        if not config_path.is_file():
            excluded.append({"model_key": model, "reason": "Model config unavailable"})
            continue
        config = OmegaConf.load(config_path)
        if str(config.model.name) != model:
            raise ValueError(f"Model config identity mismatch: {model}")
        raw_profiles = list(config.model.get("sae_profiles", []))
        sites = []
        for key in raw_profiles:
            try:
                profile = get_profile(str(key))
                if model not in profile.allowed_model_keys():
                    raise ValueError("SAE not owned by selected Model")
                for layer in sorted(set(profile.layers)):
                    sites.append(declared_site(
                        model, profile.key, layer, profile.hook_name(layer)
                    ))
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({
                    "model_key": model,
                    "sae_key": str(key),
                    "reason": str(exc)[:160],
                })
        if not sites:
            excluded.append({"model_key": model, "reason": "No declared valid SAE Site"})
        selected.extend(sorted(sites, key=lambda item: item["site_id"])[:2])
    return {
        "schema": "saebase.precheck.inventory.v1",
        "status": "DECLARED_ONLY",
        "dataset_keys": datasets,
        "model_keys": models,
        "sae_sites": selected,
        "exclusions": excluded,
        "source": "active Registry and Model-owned Profile configuration",
        "scientific_acceptance": "NOT_COVERED",
    }


def prepare_inventory(run_id: str, root: Path = ROOT) -> dict[str, Any]:
    """Create immutable first-pass inventory under the existing Runtime owner."""
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    out = root / "experiment" / "_studies" / "wp23_component_precheck" / "reports" / run_id
    if out.exists():
        raise FileExistsError("Existing run must not be overwritten")
    data = inventory(root)
    matrix = plan_compatibility(
        data["dataset_keys"], data["model_keys"], data["sae_sites"]
    )
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, timeout=20, check=False,
    )
    commit = source.stdout.strip()
    if source.returncode != 0 or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise RuntimeError("Could not establish exact source identity")
    config = root / "conf" / "experiment" / "wp23_precheck.yaml"
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in (("inventory.json", data), ("targets.json", matrix)):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump({
                **payload,
                "run_id": run_id,
                "source_sha": commit,
                "config_sha256": config_sha,
            }, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {
        "status": "INVENTORY_WRITTEN",
        "run_root": str(out),
        "datasets": len(data["dataset_keys"]),
        "models": len(data["model_keys"]),
        "sites": len(data["sae_sites"]),
        "compatibility_targets": len(matrix["pairs"]),
        "real_checks": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "inventory"))
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if args.stage == "inspect":
        data = inventory()
        matrix = plan_compatibility(
            data["dataset_keys"], data["model_keys"], data["sae_sites"]
        )
        print(json.dumps({
            "status": "PLAN_ONLY",
            "datasets": len(data["dataset_keys"]),
            "models": len(data["model_keys"]),
            "sites": len(data["sae_sites"]),
            "pairs": len(matrix["pairs"]),
            "scientific_acceptance": "NOT_COVERED",
        }, ensure_ascii=False))
        return 0
    if not args.run_id:
        parser.error("--run-id required for inventory")
    result = prepare_inventory(args.run_id)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
