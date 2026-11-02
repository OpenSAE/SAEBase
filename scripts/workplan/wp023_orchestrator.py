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


def _run_root(run_id: str, root: Path) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    path = root / "experiment" / "_studies" / "wp23_component_precheck" / "reports" / run_id
    if not path.is_dir():
        raise FileNotFoundError("Run inventory has not been initialized")
    return path


def _identity(run_root: Path) -> tuple[str, dict[str, Any]]:
    current = json.loads((run_root / "inventory.json").read_text(encoding="utf-8"))
    run_id = run_root.name
    if current.get("run_id") != run_id:
        raise ValueError("Run inventory identity mismatch")
    root = run_root.parents[4]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=False, timeout=20,
    )
    if sha.returncode or sha.stdout.strip() != current.get("source_sha"):
        raise ValueError("Source commit changed; cannot resume this run")
    if hashlib.sha256(
        (root / "conf/experiment/wp23_precheck.yaml").read_bytes()
    ).hexdigest() != current.get("config_sha256"):
        raise ValueError("Configuration changed; a new run is required")
    return current["source_sha"], current


def _receipt_path(run_root: Path, key: str) -> Path:
    label = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return run_root / "receipts" / (label + ".json")


def _store(run_root: Path, key: str, value: dict[str, Any]) -> dict[str, Any]:
    path = _receipt_path(run_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        **value,
        "target_id": key,
        "run_id": run_root.name,
        "source_sha": json.loads((run_root / "inventory.json").read_text())["source_sha"],
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_id") != result["run_id"] or existing.get("source_sha") != result["source_sha"] or existing.get("target_id") != key:
            raise ValueError("Resume receipt identity mismatch")
        return existing
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _load_status(run_root: Path, key: str) -> dict[str, Any]:
    path = _receipt_path(run_root, key)
    if not path.is_file():
        return {"status": "NOT_RUN"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("target_id") != key or value.get("run_id") != run_root.name:
        raise ValueError("Invalid existing receipt")
    return value


def _call_worker(
    run_root: Path, key: str, request: dict[str, Any], root: Path,
    *, max_seconds: int = 180,
) -> dict[str, Any]:
    previous = _load_status(run_root, key)
    if previous["status"] != "NOT_RUN":
        return previous
    work = run_root / "worker_requests"
    work.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    request_path = work / (digest + ".input.json")
    output_path = work / (digest + ".output.json")
    if request_path.exists() or output_path.exists():
        raise ValueError("Incomplete or mismatched previous Worker request; manual audit needed")
    with request_path.open("x", encoding="utf-8") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    import os
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    command = [
        "python", "-m", "tasks.precheck.worker", "--input",
        str(request_path), "--output", str(output_path),
    ]
    try:
        proc = subprocess.run(
            command, cwd=root, env=env, capture_output=True,
            text=True, timeout=max_seconds, check=False,
        )
        if output_path.is_file():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result.get("status") not in {"PASS", "FAIL", "BLOCKED"}:
                raise ValueError("Invalid Worker result")
            result["worker_exit_code"] = proc.returncode
        else:
            result = {
                "status": "FAIL", "code": "WORKER_NO_RECEIPT",
                "reason": "Worker exited without a result receipt",
                "worker_exit_code": proc.returncode,
            }
    except subprocess.TimeoutExpired:
        result = {
            "status": "BLOCKED", "code": "BLOCKED_RESOURCE",
            "reason": "Worker exceeded exact wall-time budget",
        }
    return _store(run_root, key, result)


def _model_meta(root: Path, key: str) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(root / "conf" / "model" / (key + ".yaml"))
    if str(cfg.model.name) != key:
        raise ValueError("Model config identity mismatch")
    return str(cfg.model.pretrained_name)


def _real_prompt(root: Path, dataset: str, model: str) -> str:
    from omegaconf import OmegaConf
    from saebase.datasets.prompts import read_prompts

    cfg = OmegaConf.create({
        "name": dataset,
        "data_path": str(root / "data" / dataset),
        "prompt_file": "",
    })
    rows = read_prompts(cfg, model_key=model)
    prompt = rows[0].get("prompt") if rows else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("No real Dataset prompt available")
    return prompt


def execute_stage(
    run_id: str, stage: str, root: Path = ROOT, *,
    allow_compute: bool = False, allow_gpu: bool = False,
) -> dict[str, Any]:
    """Run a bounded component stage without modifying any product source."""
    run_root = _run_root(run_id, root)
    sha, inv = _identity(run_root)
    pairs = json.loads((run_root / "targets.json").read_text())["pairs"]
    datasets, models, sites = inv["dataset_keys"], inv["model_keys"], inv["sae_sites"]
    if stage not in {"dataset", "model", "sae", "compatibility", "task"}:
        raise ValueError("Unknown execution stage")

    def auth(key: str, gpu: bool) -> dict[str, Any] | None:
        if not allow_compute or (gpu and not allow_gpu):
            return _store(run_root, key, {
                "status": "BLOCKED", "code": "BLOCKED_AUTH",
                "reason": "Real compute/GPU not explicitly authorized",
            })
        return None

    done = []
    if stage == "dataset":
        for dataset in datasets:
            key = f"dataset:{dataset}"
            if _load_status(run_root, key)["status"] != "NOT_RUN":
                done.append(_load_status(run_root, key))
                continue
            # Dataset artifacts are legitimately model-specific. A Dataset is
            # available if at least one source-owned real cohort can be checked.
            actual = None
            for model in models:
                from omegaconf import OmegaConf
                from saebase.datasets.prompts import prompt_file

                cfg = OmegaConf.create({
                    "name": dataset,
                    "data_path": str(root / "data" / dataset),
                    "prompt_file": "",
                })
                try:
                    prompt_file(cfg, model_key=model)
                    actual = model
                    break
                except (FileNotFoundError, ValueError):
                    continue
            if actual is None:
                done.append(_store(run_root, key, {
                    "status": "BLOCKED", "code": "BLOCKED_ASSET",
                    "reason": "No prepared genuine Dataset artifact is present",
                }))
            else:
                done.append(_call_worker(
                    run_root, key, {
                        "kind": "dataset", "dataset_key": dataset,
                        "model_key": actual, "data_root": str(root / "data"),
                    }, root,
                ))
    elif stage == "model":
        for model in models:
            key = f"model:{model}"
            if auth(key, True):
                done.append(_load_status(run_root, key))
                continue
            done.append(_call_worker(
                run_root, key, {
                    "kind": "model", "model_key": model,
                    "model_name": _model_meta(root, model),
                    "prompt": "The capital of France is", "device": "cuda",
                }, root,
            ))
    elif stage == "sae":
        for site in sites:
            key = "sae:" + site["site_id"]
            if auth(key, True):
                done.append(_load_status(run_root, key))
                continue
            if _load_status(run_root, "model:" + site["model_key"])["status"] != "PASS":
                done.append(_store(run_root, key, {
                    "status": "BLOCKED", "code": "BLOCKED_DEPENDENCY",
                    "reason": "Parent real Model Forward has not passed",
                }))
                continue
            done.append(_call_worker(
                run_root, key, {
                    "kind": "sae", "model_key": site["model_key"],
                    "sae_key": site["sae_key"], "layer": site["layer"],
                    "prompt": "The capital of France is", "device": "cuda",
                }, root,
            ))
    elif stage == "compatibility":
        from tasks.precheck.compatibility_checks import plan_compatibility

        parents = {
            target: _load_status(run_root, target)
            for target in (
                [f"dataset:{v}" for v in datasets] +
                [f"model:{v}" for v in models] +
                [f"sae:{v['site_id']}" for v in sites]
            )
        }
        matrix = plan_compatibility(datasets, models, sites, parents)
        sites_by_id = {site["site_id"]: site for site in sites}
        for pair in matrix["pairs"]:
            key = pair["id"]
            if not pair["eligible"]:
                done.append(_store(run_root, key, {
                    "status": "BLOCKED" if pair["status"] == "BLOCKED" else "NOT_RUN",
                    "reason": "Component prerequisite not verified",
                    "dependency_states": pair["dependency_states"],
                }))
                continue
            if auth(key, True):
                done.append(_load_status(run_root, key))
                continue
            site = sites_by_id.get(pair["site_id"])
            req = {
                "kind": "compatibility", "target_id": key,
                "dataset_key": pair["dataset_id"],
                "model_key": pair["model_key"],
                "model_name": _model_meta(root, pair["model_key"]),
                "data_root": str(root / "data"),
                "max_samples": 2, "device": "cuda",
            }
            if site is not None:
                req.update(site)
            done.append(_call_worker(run_root, key, req, root))
    else:
        names = ("capture", "intervene", "generate", "benchmark")
        for name in names:
            key = "task:" + name
            done.append(_store(run_root, key, {
                "status": "BLOCKED",
                "code": "BLOCKED_CONFIGURATION",
                "reason": (
                    "Native Task smoke requires an exact supported site, "
                    "genuine Dataset sample and a bounded Task-owned run configuration; "
                    "no unreviewed default is executed"
                ),
            }))
    return {
        "stage": stage, "run_id": run_id,
        "requested": len(done),
        "status_counts": {
            kind: sum(x.get("status") == kind for x in done)
            for kind in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")
        },
        "source_sha": sha,
    }


def audit_and_handoff(run_id: str, root: Path = ROOT) -> dict[str, Any]:
    """Finalize honest coverage, source evidence and the filled Workplan."""
    import csv
    from saebase.runtime.audit import file_sha256
    from saebase.runtime.audit_report import audit_run
    from saebase.runtime.audit_delivery import deliver
    from harness.github.work_plan.lifecycle import completed

    run_root = _run_root(run_id, root)
    sha, inv = _identity(run_root)
    targets = json.loads((run_root / "targets.json").read_text())["pairs"]
    components = (
        [f"dataset:{x}" for x in inv["dataset_keys"]]
        + [f"model:{x}" for x in inv["model_keys"]]
        + [f"sae:{x['site_id']}" for x in inv["sae_sites"]]
    )
    pair_keys = [x["id"] for x in targets]
    task_keys = [f"task:{x}" for x in ("capture", "intervene", "generate", "benchmark")]
    all_keys = components + pair_keys + task_keys
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Repeated declared component or compatibility target")
    rows = []
    for key in all_keys:
        value = _load_status(run_root, key)
        if value["status"] == "NOT_RUN" and not _receipt_path(run_root, key).exists():
            value = _store(run_root, key, {
                "status": "NOT_RUN", "reason": "Target was not executed",
            })
        rows.append({
            "target_id": key, "status": value["status"],
            "code": value.get("code", ""),
            "reason": value.get("reason", ""),
            "receipt": "receipts/" + _receipt_path(run_root, key).name,
        })
    by_key = {row["target_id"]: row for row in rows}

    def csv_write(name: str, selected: list[dict[str, Any]]) -> None:
        with (run_root / name).open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=["target_id", "status", "code", "reason", "receipt"],
            )
            writer.writeheader()
            writer.writerows(selected)

    csv_write("component_checks.csv", [by_key[k] for k in components])
    csv_write("compatibility_checks.csv", [by_key[k] for k in pair_keys])
    csv_write("task_checks.csv", [by_key[k] for k in task_keys])
    grouped = []
    for group, keys in (
        ("dataset", components[:len(inv["dataset_keys"])]),
        ("model", [x for x in components if x.startswith("model:")]),
        ("sae", [x for x in components if x.startswith("sae:")]),
        ("compatibility", pair_keys), ("task", task_keys),
    ):
        counts = {status: sum(by_key[k]["status"] == status for k in keys)
                  for status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")}
        grouped.append({"group": group, "declared": len(keys), **counts})
    with (run_root / "coverage_summary.csv").open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["group", "declared", "PASS", "FAIL", "BLOCKED", "NOT_RUN"],
        )
        writer.writeheader()
        writer.writerows(grouped)

    def manifest_for(paths: list[Path]) -> dict[str, Any]:
        return {
            "schema": "saebase.artifact_manifest.v1",
            "run_id": run_id, "source_sha": sha,
            "files": [
                {
                    "path": str(path.relative_to(run_root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "required": True,
                }
                for path in sorted(paths)
            ],
            "checks": [
                {
                    "id": key,
                    "kind": "task" if key.startswith("task:") else "precheck",
                    "receipt": "receipts/" + _receipt_path(run_root, key).name,
                }
                for key in all_keys
            ],
        }

    receipts = [_receipt_path(run_root, key) for key in all_keys]
    csv_paths = [
        run_root / name for name in (
            "component_checks.csv", "compatibility_checks.csv",
            "task_checks.csv", "coverage_summary.csv",
        )
    ]
    initial = audit_run(run_root, manifest_for(receipts + csv_paths))
    groups = {
        "E1": [k for k in components if k.startswith("dataset:")],
        "E2": [k for k in components if k.startswith("model:")],
        "E3": [k for k in components if k.startswith("sae:")],
        "E4": [k for k in pair_keys if k.startswith("model_dataset:")],
        "E5": [k for k in pair_keys if k.startswith("sae_site_dataset:")],
        "E6": task_keys,
    }
    evidence = {
        "E1": "component_checks.csv", "E2": "component_checks.csv",
        "E3": "component_checks.csv", "E4": "compatibility_checks.csv",
        "E5": "compatibility_checks.csv", "E6": "task_checks.csv",
        "E7": "audit.json",
    }
    results = {}
    for name, keys in groups.items():
        values = [by_key[k]["status"] for k in keys]
        status = (
            "FAIL" if "FAIL" in values else
            "BLOCKED" if "BLOCKED" in values else
            "NOT_RUN" if "NOT_RUN" in values else
            "PASS" if values else "NOT_RUN"
        )
        results[name] = {
            "status": status, "value": sum(v == "PASS" for v in values),
            "notes": f"PASS {sum(v == 'PASS' for v in values)}/{len(values)}; native scientific acceptance not covered",
            "evidence": evidence[name],
        }
    results["E7"] = {
        "status": "PASS" if initial["evidence_status"] == "PASS" else "BLOCKED",
        "value": len(rows), "notes": "Evidence completeness is distinct from execution success",
        "evidence": evidence["E7"],
    }
    run_receipt = {
        "schema": "saebase.workplan.run.v1",
        "plan_id": "WP23", "run_id": run_id, "source_commit": sha,
        "results": results,
    }
    completed_path = run_root / "workplan_completed.md"
    completed_plan = completed(
        "harness/github/work_plan/0.6.5_wp023_qa_precheck.md",
        run_receipt, root=root,
    )
    completed_path.write_text(completed_plan, encoding="utf-8")
    manifest = manifest_for(receipts + csv_paths + [completed_path])
    report = audit_run(run_root, manifest)
    (run_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (run_root / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    delivered = deliver(run_root, manifest, run_root / "archive", archive=False)
    (run_root / "archive_receipt.json").write_text(
        json.dumps(delivered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return {
        "status": report["execution_status"],
        "evidence_status": report["evidence_status"],
        "scientific_acceptance": "NOT_COVERED",
        "run_id": run_id, "report_root": str(run_root),
        "checked_targets": len(rows),
        "archive_status": delivered["archive_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "inventory", "dataset", "model", "sae", "compatibility", "task", "audit"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--allow-compute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
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
    result = (prepare_inventory(args.run_id) if args.stage == "inventory" else
              audit_and_handoff(args.run_id) if args.stage == "audit" else
              execute_stage(args.run_id, args.stage,
                            allow_compute=args.allow_compute,
                            allow_gpu=args.allow_gpu))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
