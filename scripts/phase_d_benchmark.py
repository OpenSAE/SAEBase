"""Stage D real M3/M4 runner for any catalog-selected SAE site.

Builds a BenchmarkConfig with registered metric IDs, freezes token IDs, runs
native M3, then M4 against stored activations (no second model load for audit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from saebase.api import ActivationSite, ModelAdapter, capture_features
from saebase.benchmarks import BenchmarkConfig, build_plan
from saebase.benchmarks.report import write_report
from saebase.benchmarks.runner import execute
from saebase.reliability import AuditConfig, evaluate_m3, read_audit, write_audit
from saebase.sae.coverage import load_coverage
from saebase.sae.loading import load_model_and_sae
from saebase.sae.registry import get_profile
from saebase.sae.stage_d import PROMPTS, update_status
from saebase.benchmarks.results import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


def run_site(
    *,
    sae_key: str,
    layer: int,
    device: str,
    output: Path,
    smoke_only: bool = True,
) -> dict:
    catalog = load_coverage(CATALOG)
    entry = catalog.find(sae_key)
    if layer not in entry.selected_layers:
        raise ValueError(f"layer {layer} not selected for {sae_key}")
    profile = get_profile(sae_key)
    raw_model, sae, meta = load_model_and_sae(sae_key, layer, device=device)
    model = ModelAdapter(raw_model)
    site = ActivationSite(layer, meta["hook_name"], position=-1)
    if site.hook_name != profile.hook_name(layer):
        raise ValueError("loaded SAE hook disagrees with registry")

    selection = capture_features(model, sae, site=site, prompt=PROMPTS[0])
    chosen = int(torch.argmax(selection.codes.abs()).item())
    if float(selection.codes[chosen].abs()) == 0:
        raise ValueError("no live SAE feature to ablate")

    prompts = PROMPTS if smoke_only else PROMPTS  # population sets can replace later
    samples = []
    for i, prompt in enumerate(prompts):
        tokens = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        samples.append({
            "sample_id": f"stage-d-{entry.model_key}-L{layer}-{i:02d}",
            "token_ids": tokens,
            "prompt_group_id": f"stage-d-sample-{i:02d}",
            "split": "test",
            "selection": {"mode": "indices", "indices": [len(tokens) - 1]},
        })

    config = BenchmarkConfig.from_dict({
        "schema": "saebase.benchmark.config.v1",
        "name": f"Stage D {entry.model_key} {sae_key} L{layer}",
        "targets": [{
            "model_key": entry.model_key,
            "sae_keys": [sae_key],
            "layer": layer,
            "hook_name": site.hook_name,
        }],
        "dataset": {
            "id": f"saebase-stage-d-{entry.model_key}-L{layer}",
            "samples": samples,
        },
        "metrics": [
            "core.l0",
            "core.reconstruction_mse",
            "core.explained_variance",
            "intervention.logit_delta_l2",
        ],
        "conditions": [
            {"name": "identity", "operation": "identity"},
            {"name": "feature-ablation", "operation": "ablate", "feature_ids": [chosen]},
        ],
        "max_activation_rows": 24,
        "max_measurements": 24,
        "sample_batch": 1,
        "microbatch": 1,
        "seed": 0,
        "tolerance": 0.01,
    })
    runs_root = output / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows = execute(
        config,
        runs_root,
        real=True,
        loader=lambda mk, sk, l: SimpleNamespace(model=model, sae=sae),
    )
    if not rows or any(row["status"] != "completed" for row in rows):
        raise ValueError("real benchmark has incomplete metrics")
    root = runs_root / config.run_id
    acfg = AuditConfig(
        evaluation_seeds=(0, 1, 2),
        bootstrap_replicates=32,
        sample_sizes=(2, 3),
        metric_ids=("core.l0", "intervention.logit_delta_l2"),
    )
    audit = evaluate_m3(root, acfg)
    if any(row["status"] != "completed" for row in audit["metric_audits"]):
        raise ValueError("real M4 audit has incomplete metric audits")
    write_audit(root, audit)
    write_report(root, config, build_plan(config), rows)
    if read_audit(root) != audit:
        raise ValueError("real M4 audit failed integrity roundtrip")

    record = {
        "schema": "saebase.stage_d.real_m3m4_acceptance.v1",
        "status": "passed",
        "protocol": "smoke_only" if smoke_only else "population",
        "model_key": entry.model_key,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": site.hook_name,
        "feature_ablated": chosen,
        "sample_count": len(samples),
        "benchmark_metric_rows": len(rows),
        "reliability_metric_audits": len(audit["metric_audits"]),
        "n_model_reloads_for_audit": 0,
        "m3_run_id": config.run_id,
        "m3_metrics_sha256": json.loads((root / "metrics_manifest.json").read_text())["sha256"],
        "m4_audit_sha256": json.loads((root / "reliability_manifest.json").read_text())["audit_sha256"],
        "metric_results": [{
            "metric_id": row["metric_id"],
            "condition": row["condition_id"],
            "value": row["value"],
            "n_valid": row["n_valid"],
        } for row in rows],
    }
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / f"{entry.model_key}_{sae_key}_L{layer}_m3m4.json"
    atomic_json(summary_path, record)
    site_id = f"{entry.model_key}|{sae_key}|L{layer}|{site.hook_name}"
    update_status(output, site_id, {
        "benchmark_verified": True,
        "reliability_verified": True,
        "status": "reliability_verified",
        "m3_run_id": config.run_id,
        "m3m4_summary": str(summary_path),
    })
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-key", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/stage_d_runs"))
    parser.add_argument(
        "--population",
        action="store_true",
        help="Mark protocol as population (still uses the frozen prompt set until expanded)",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    record = run_site(
        sae_key=args.sae_key,
        layer=args.layer,
        device=args.device,
        output=args.output,
        smoke_only=not args.population,
    )
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
