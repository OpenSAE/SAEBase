"""Stage D real M3/M4 runner for any catalog-selected SAE site.

Builds a BenchmarkConfig with registered metric IDs, freezes token IDs, runs
native M3, then M4 against stored activations (no second model load for audit).

``--population`` requires the frozen corpus in
``conf/coverage/stage_d_population.jsonl`` with ≥64 independent evaluation
groups. Feature IDs are chosen only on the selection split.
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
from saebase.benchmarks.results import atomic_json
from saebase.benchmarks.runner import execute
from saebase.reliability import AuditConfig, evaluate_m3, read_audit, write_audit
from saebase.sae.coverage import load_coverage
from saebase.sae.loading import load_model_and_sae
from saebase.sae.registry import get_profile
from saebase.sae.stage_d import (
    POPULATION_MIN_EVAL_GROUPS,
    PROMPTS,
    PROTOCOL_POPULATION,
    PROTOCOL_SMOKE,
    load_population_corpus,
    relative_under,
    resolve_evaluation_host,
    update_status,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"
DEFAULT_POPULATION = ROOT / "conf" / "coverage" / "stage_d_population.jsonl"


def _write_token_ledger(
    output: Path,
    *,
    host: str,
    sae_key: str,
    layer: int,
    split: str,
    rows: list[dict],
    tokenizer_name: str | None,
) -> Path:
    dest_dir = output / split / host / sae_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"L{layer}.json"
    payload = {
        "schema": f"saebase.stage_d.{split}_tokens.v1",
        "model_key": host,
        "sae_key": sae_key,
        "layer": layer,
        "tokenizer_name": tokenizer_name,
        "n_rows": len(rows),
        "rows": rows,
    }
    body = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    payload["sha256"] = __import__("hashlib").sha256(body).hexdigest()
    atomic_json(dest, payload)
    return dest


def run_site(
    *,
    sae_key: str,
    layer: int,
    device: str,
    output: Path,
    smoke_only: bool = True,
    population_path: Path | None = None,
) -> dict:
    catalog = load_coverage(CATALOG)
    entry = catalog.find(sae_key)
    if layer not in entry.selected_layers:
        raise ValueError(f"layer {layer} not selected for {sae_key}")
    profile = get_profile(sae_key)
    evaluation_host, override = resolve_evaluation_host(
        catalog_model_key=entry.model_key,
        sae_key=sae_key,
    )
    dataset_sha = None
    protocol = PROTOCOL_SMOKE
    if smoke_only:
        selection_prompts = list(PROMPTS)
        evaluation_prompts = [
            {"group_id": f"smoke-{i:02d}", "prompt": p} for i, p in enumerate(PROMPTS)
        ]
        selection_for_feature = PROMPTS[0]
    else:
        corpus = load_population_corpus(population_path or DEFAULT_POPULATION)
        if corpus["n_evaluation_groups"] < POPULATION_MIN_EVAL_GROUPS:
            raise ValueError(
                f"--population requires >= {POPULATION_MIN_EVAL_GROUPS} evaluation "
                f"groups; corpus has {corpus['n_evaluation_groups']}"
            )
        selection_prompts = [r["prompt"] for r in corpus["selection"]]
        evaluation_prompts = corpus["evaluation"]
        selection_for_feature = selection_prompts[0]
        dataset_sha = corpus["dataset_sha256"]
        protocol = PROTOCOL_POPULATION

    raw_model, sae, meta = load_model_and_sae(
        sae_key,
        layer,
        device=device,
        model_key_override=override,
    )
    model = ModelAdapter(raw_model)
    site = ActivationSite(layer, meta["hook_name"], position=-1)
    if site.hook_name != profile.hook_name(layer):
        raise ValueError("loaded SAE hook disagrees with registry")
    loaded_name = meta.get("loaded_model_name")
    from saebase.sae.stage_d import assert_host_matches_label

    assert_host_matches_label(
        evaluation_host_model_key=evaluation_host,
        loaded_model_name=loaded_name,
    )

    # Choose feature IDs only on the selection split (never on evaluation).
    # Prefer features with the largest mean |code| across selection prompts so
    # TopK sites remain more likely to fire on held-out evaluation groups.
    acc = None
    for prompt in selection_prompts:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        abs_z = snap.codes.detach().abs().float().cpu()
        acc = abs_z if acc is None else acc + abs_z
    assert acc is not None
    mean_abs = acc / max(len(selection_prompts), 1)
    chosen = int(torch.argmax(mean_abs).item())
    if float(mean_abs[chosen]) == 0:
        raise ValueError("no live SAE feature to ablate on selection split")

    tokenizer_name = getattr(getattr(model, "model", None), "tokenizer_name", None)
    if tokenizer_name is None:
        tokenizer_name = getattr(getattr(model, "model", None), "cfg", None)
        tokenizer_name = getattr(tokenizer_name, "model_name", None) or loaded_name

    sel_rows = []
    for i, prompt in enumerate(selection_prompts):
        tokens = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        sel_rows.append(
            {
                "group_id": f"sel-{i:02d}",
                "prompt": prompt,
                "token_ids": tokens,
            }
        )
    _write_token_ledger(
        output,
        host=evaluation_host,
        sae_key=sae_key,
        layer=layer,
        split="selection",
        rows=sel_rows,
        tokenizer_name=str(tokenizer_name),
    )

    samples = []
    eval_rows = []
    for i, row in enumerate(evaluation_prompts):
        prompt = row["prompt"] if isinstance(row, dict) else row
        group_id = row["group_id"] if isinstance(row, dict) else f"eval-{i:02d}"
        tokens = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        eval_rows.append(
            {
                "group_id": group_id,
                "prompt": prompt,
                "token_ids": tokens,
            }
        )
        samples.append(
            {
                "sample_id": f"stage-d-{evaluation_host}-L{layer}-{group_id}",
                "token_ids": tokens,
                "prompt_group_id": group_id,
                "split": "test",
                "selection": {"mode": "indices", "indices": [len(tokens) - 1]},
            }
        )
    _write_token_ledger(
        output,
        host=evaluation_host,
        sae_key=sae_key,
        layer=layer,
        split="evaluation",
        rows=eval_rows,
        tokenizer_name=str(tokenizer_name),
    )

    if not smoke_only and len({s["prompt_group_id"] for s in samples}) < POPULATION_MIN_EVAL_GROUPS:
        raise ValueError("population evaluation collapsed below 64 independent groups")

    n_samples = len(samples)
    n_conditions = 2
    max_rows = 24 if smoke_only else max(64, n_samples)
    max_meas = 24 if smoke_only else max(128, n_samples * n_conditions)
    config = BenchmarkConfig.from_dict(
        {
            "schema": "saebase.benchmark.config.v1",
            "name": f"Stage D {evaluation_host} {sae_key} L{layer}",
            "targets": [
                {
                    "model_key": evaluation_host,
                    "sae_keys": [sae_key],
                    "layer": layer,
                    "hook_name": site.hook_name,
                }
            ],
            "dataset": {
                "id": f"saebase-stage-d-{protocol}-{evaluation_host}-L{layer}",
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
            "max_activation_rows": max_rows,
            "max_measurements": max_meas,
            "sample_batch": 1,
            "microbatch": 1,
            "seed": 0,
            "tolerance": 0.01,
        }
    )
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
    # High reconstruction error sites: record diagnostics, do not auto-fail.
    recon_diag = {}
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        mse_vals = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metric_id") == "core.reconstruction_mse" and rec.get("value") is not None:
                mse_vals.append(float(rec["value"]))
        if mse_vals:
            mean_mse = sum(mse_vals) / len(mse_vals)
            var_mse = sum((v - mean_mse) ** 2 for v in mse_vals) / len(mse_vals)
            recon_diag = {
                "reconstruction_mse_mean": mean_mse,
                "reconstruction_mse_var": var_mse,
                "reconstruction_mse_n": len(mse_vals),
            }
    acfg = AuditConfig(
        evaluation_seeds=(0, 1, 2),
        bootstrap_replicates=32,
        sample_sizes=(2, 3) if smoke_only else (8, 16, 32),
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
        "protocol_id": protocol,
        "model_key": evaluation_host,
        "evaluation_host_model_key": evaluation_host,
        "trained_on_model_key": profile.trained_on_model_key or profile.model_key,
        "transfer_kind": profile.transfer_kind,
        "sae_key": sae_key,
        "layer": layer,
        "hook_name": site.hook_name,
        "loaded_model_name": loaded_name,
        "feature_ablated": chosen,
        "feature_selection_split": "selection" if not smoke_only else "smoke",
        "sample_count": len(samples),
        "n_independent_evaluation_groups": len({s["prompt_group_id"] for s in samples}),
        "dataset_sha256": dataset_sha
        or __import__("hashlib").sha256("\n".join(PROMPTS).encode("utf-8")).hexdigest(),
        "population_corpus_sha256": dataset_sha,
        "benchmark_metric_rows": len(rows),
        "reliability_metric_audits": len(audit["metric_audits"]),
        "n_model_reloads_for_audit": 0,
        "m3_run_id": config.run_id,
        "m3_metrics_sha256": json.loads((root / "metrics_manifest.json").read_text())["sha256"],
        "m4_audit_sha256": json.loads((root / "reliability_manifest.json").read_text())[
            "audit_sha256"
        ],
        "reconstruction_diagnostics": recon_diag,
        "metric_results": [
            {
                "metric_id": row["metric_id"],
                "condition": row["condition_id"],
                "value": row["value"],
                "n_valid": row["n_valid"],
            }
            for row in rows
        ],
    }
    if not smoke_only:
        # Prefer corpus hash for smoke dataset_sha field when population.
        record["dataset_sha256"] = dataset_sha
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / f"{evaluation_host}_{sae_key}_L{layer}_m3m4.json"
    atomic_json(summary_path, record)
    site_id = f"{evaluation_host}|{sae_key}|L{layer}|{site.hook_name}"
    update_status(
        output,
        site_id,
        {
            "benchmark_verified": True,
            "reliability_verified": True,
            "status": "reliability_verified",
            "protocol": "smoke_only" if smoke_only else "population",
            "m3_run_id": config.run_id,
            "m3m4_summary": relative_under(output, summary_path),
            "dataset_sha256": record["dataset_sha256"],
            "evaluation_host_model_key": evaluation_host,
            "loaded_model_name": loaded_name,
        },
    )
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
        help="Require frozen >=64-group population corpus (fails if unavailable)",
    )
    parser.add_argument(
        "--population-path",
        type=Path,
        default=None,
        help="Override path to stage_d_population.jsonl",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Force three-prompt smoke protocol (default when --population is absent)",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.population:
        # Hard-validate corpus before any model load.
        corpus = load_population_corpus(args.population_path or DEFAULT_POPULATION)
        if corpus["n_evaluation_groups"] < POPULATION_MIN_EVAL_GROUPS:
            raise SystemExit(
                f"--population refused: only {corpus['n_evaluation_groups']} "
                f"evaluation groups (need >= {POPULATION_MIN_EVAL_GROUPS})"
            )
        smoke_only = False
    else:
        smoke_only = True
    record = run_site(
        sae_key=args.sae_key,
        layer=args.layer,
        device=args.device,
        output=args.output,
        smoke_only=smoke_only,
        population_path=args.population_path,
    )
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
