"""Task-owned benchmark configuration, execution and completion semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conf import configuration_location, read_configuration
from saebase.results import completion_summary
from saebase.runtime.configuration.task_options import task_value
from saebase.runtime.experiment_paths import configured_experiment_name
from saebase.runtime.sae_operations import _task, input_path


def configured_benchmark(cfg: Any):
    """Parse and validate the actual scientific protocol without loading checkpoints."""
    from omegaconf import OmegaConf

    from saebase.registry import selected_models
    from tasks.benchmark.comparison.schema import MatrixConfig
    from tasks.benchmark.execution.planner import build_plan
    from tasks.benchmark.execution.schema import BenchmarkConfig

    models = selected_models(cfg)
    protocol = _task(cfg, "protocol")
    source = _task(cfg, "benchmark_config", "")
    if protocol is not None and source:
        raise ValueError("Select task.benchmark.protocol or a task parameter file, not both")
    path = input_path(source) if source else None
    if protocol is not None:
        raw = OmegaConf.to_container(protocol, resolve=True)
    else:
        if path is None or not configuration_location(path)[0].is_file():
            raise FileNotFoundError(
                "Provide task.benchmark.protocol or an existing task.benchmark.benchmark_config parameter file"
            )
        raw = read_configuration(path, resolve=True)
    selectors = {"targets", "dataset", "model", "models", "prompts"}
    if not isinstance(raw, dict):
        raise ValueError("Benchmark method parameters must be an object")
    if (protocol is not None and set(raw) & selectors) or set(raw) & {"model", "models"}:
        raise ValueError(
            "task.benchmark.protocol contains method parameters only; select model and dataset globally"
        )
    if not set(raw) & selectors:
        from saebase.datasets.prompts import read_measurement_samples

        from saebase.runtime.sae_operations import resolve_sae

        targets = []
        for model in models:
            selected = resolve_sae(
                model, _task(cfg, "sae_key"), _task(cfg, "layer"), operation="precheck"
            )
            targets.append(
                {
                    "model_key": model.name,
                    "sae_keys": [selected.sae_key],
                    "layer": selected.layer,
                    "hook_name": selected.hook,
                }
            )
        raw = {
            "schema": "saebase.benchmark.config.v1",
            "name": configured_experiment_name(cfg) or "metric_evaluation",
            **raw,
            "targets": targets,
            "dataset": {"id": cfg.dataset.name, "samples": read_measurement_samples(cfg.dataset)},
        }
    if {item.name for item in models} != {target["model_key"] for target in raw["targets"]}:
        raise ValueError("Benchmark targets conflict with the global model selection")
    if "dataset" in raw and str(raw["dataset"]["id"]) != cfg.dataset.name:
        raise ValueError("Benchmark dataset conflicts with the global dataset selection")
    if "dataset" in raw and (
        cfg.dataset.get("samples") is not None or cfg.dataset.get("sample_file")
    ):
        from saebase.datasets.prompts import read_measurement_samples

        if raw["dataset"]["samples"] != read_measurement_samples(cfg.dataset):
            raise ValueError("Benchmark samples conflict with the global dataset input")
    if "prompts" in raw and cfg.dataset.get("prompts") is not None:
        if raw["prompts"] != list(cfg.dataset.prompts):
            raise ValueError("Benchmark prompts conflict with the global dataset input")
    if raw.get("schema") == "saebase.model_comparison":
        from tasks.benchmark.comparison.planner import plan_matrix

        config = MatrixConfig.from_dict(raw)
        plan_matrix(config)
    else:
        config = BenchmarkConfig.from_dict(raw)
        build_plan(config)
    selected = _task(cfg, "sae_key")
    layer = _task(cfg, "layer")
    if selected is not None and any(selected not in t.sae_keys for t in config.targets):
        raise ValueError(
            "task.sae_key conflicts with benchmark targets; select it in benchmark_config"
        )
    if layer is not None and any(layer != t.site.layer for t in config.targets):
        raise ValueError("task.layer conflicts with benchmark targets")
    options = getattr(config, "options", {})
    if options.get("device") is not None and options["device"] != _task(cfg, "device", "cpu"):
        raise ValueError("task.device conflicts with benchmark options.device")
    return path, config


def benchmark_completion(config, values) -> dict:
    """Cover every requested target/SAE/metric, including multiple condition rows."""
    rows = [v.to_dict() if hasattr(v, "to_dict") else v for v in values]
    work = []
    priority = {"completed": 0, "undefined": 1, "unsupported": 2, "blocked": 3, "failed": 4}
    for target in config.targets:
        for sae in target.sae_keys:
            for metric in config.metrics:
                matches = [
                    r
                    for r in rows
                    if r["model_id"] == target.model_key
                    and r["sae_id"] == sae
                    and r["metric_id"] == metric
                    and r["site"]["layer"] == target.site.layer
                    and r["site"]["hook_name"] == target.site.hook_name
                ]
                if (
                    metric.startswith("intervention.")
                    and matches
                    and not (
                        {c.name for c in config.conditions}
                        <= {row.get("condition_id") for row in matches}
                    )
                ):
                    work.append(
                        {"status": "blocked", "reason": f"Missing required condition: {metric}"}
                    )
                elif not matches:
                    work.append(
                        {
                            "status": "blocked",
                            "reason": f"Missing required result: {target.model_key}/{sae}/{metric}",
                        }
                    )
                else:
                    for row in matches:
                        if row.get("status") not in priority:
                            raise ValueError("Unknown metric status")
                    worst = max(matches, key=lambda r: priority[r["status"]])
                    work.append(worst)
    return completion_summary(work, unit="target_sae_metric")


def benchmark_input_identity(config, root: Path) -> dict | None:
    """Bind recorded actual weight hashes to the exact task protocol and samples."""
    from saebase.store import fingerprint

    path = root / "provenance.json"
    if not path.is_file():
        return None
    provenance = json.loads(path.read_text())
    expected = [f"{t.model_key}/{t.site.layer}/{sae}" for t in config.targets for sae in t.sae_keys]
    if any(
        not provenance.get(key, {}).get("model_state_sha256")
        or not provenance.get(key, {}).get("sae_state_sha256")
        for key in expected
    ):
        return None
    return {
        "schema": "saebase.task_input.v1",
        "task": "benchmark",
        "protocol_id": "saebase.benchmark.config.v1",
        "model_sha256": fingerprint(
            {key: provenance[key]["model_state_sha256"] for key in expected}
        ),
        "sae_sha256": fingerprint({key: provenance[key]["sae_state_sha256"] for key in expected}),
        "data_sha256": fingerprint(config.to_dict()["dataset"]),
        "protocol_sha256": fingerprint(config.to_dict()),
        "sites": [
            dict(layer=t.site.layer, hook_name=t.site.hook_name, position=t.site.position)
            for t in config.targets
        ],
    }


def resolved_protocol(config, cfg: Any) -> dict:
    """Keep organizational launch labels separate from actual scientific selections."""
    from saebase.runtime.experiment_paths import run_components

    return {
        "run_components": run_components(cfg),
        "configuration_path": str(input_path(_task(cfg, "benchmark_config")))
        if _task(cfg, "benchmark_config")
        else None,
        "targets": [
            target.to_dict()
            if hasattr(target, "to_dict")
            else {
                "model_key": target.model_key,
                "sae_keys": list(target.sae_keys),
                "layer": target.site.layer,
                "hook_name": target.site.hook_name,
            }
            for target in config.targets
        ],
        "dataset": str(cfg.dataset.name),
        "launch_seed": int(cfg.seed),
        "protocol_seed": int(config.seed),
        "device": str(_task(cfg, "device", "cpu")),
        "configuration": config.to_dict(),
    }


def _benchmark(cfg: Any) -> dict[str, Any]:
    from saebase.runtime.experiment_paths import run_components
    from tasks.benchmark.execution.schema import BenchmarkConfig

    path, config = configured_benchmark(cfg)
    common = {
        "status": "executed",
        "operation": "benchmark",
        "model_key": config.targets[0].model_key
        if len({target.model_key for target in config.targets}) == 1
        else run_components(cfg)["model"],
        "model_keys": list(dict.fromkeys(t.model_key for t in config.targets)),
        "resolved_protocol": resolved_protocol(config, cfg),
    }
    if not isinstance(config, BenchmarkConfig):
        from tasks.benchmark.comparison.runner import execute_matrix

        result = execute_matrix(
            config,
            Path.cwd() / "metrics",
            real=True,
            device=str(_task(cfg, "device", "cpu")),
            resume=bool(_task(cfg, "resume", True)),
        )
        work = [
            {
                "status": "completed" if result["status"] == "passed" else "failed",
                "reason": None if result["status"] == "passed" else result["status"],
            }
            for _ in config.targets
        ]
        return {
            **common,
            "comparison": result,
            "completion": completion_summary(work, unit="model_evaluation_and_audit"),
        }
    from tasks.benchmark.execution.runner import execute

    values = execute(
        config,
        Path.cwd() / "metrics",
        real=True,
        resume=bool(_task(cfg, "resume", True)),
        device=str(_task(cfg, "device", "cpu")),
    )
    return {
        **common,
        "benchmark_id": config.run_id,
        "metric_count": len(values),
        "completion": benchmark_completion(config, values),
        "input_identity": benchmark_input_identity(config, Path.cwd() / "metrics" / config.run_id),
        "metrics": [v.to_dict() if hasattr(v, "to_dict") else v for v in values],
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _benchmark(cfg)


def validate_cfg(cfg: Any) -> None:
    """Validate this Task at its input boundary, before resource loading."""
    from saebase.runtime.sae_operations import validate_fields

    validate_fields(cfg, integers=(("layer", 0),), booleans=("resume",))
    if task_value(cfg.task, "evaluation") == "activation_health":
        return
    if task_value(cfg.task, "benchmark_config") or task_value(cfg.task, "protocol") is not None:
        configured_benchmark(cfg)
    elif cfg.task.get("execute", False):
        raise ValueError("Provide task.benchmark.protocol or task.benchmark.benchmark_config before execution")


def prepare_execution(cfg: Any) -> dict:
    """Prepare the selected protocol without running the research calculation."""
    if task_value(cfg.task, "evaluation") == "activation_health":
        from tasks.benchmark.metrics.health import run_preflight

        run_preflight(cfg)
        return {
            "status": "PASS",
            "scope": "activation-health inputs prepared; no evaluation executed",
        }
    _, config = configured_benchmark(cfg)
    return {
        "status": "PASS",
        "scope": "benchmark protocol and inputs prepared; no metrics executed",
        "targets": len(config.targets),
        "metrics": len(config.metrics),
    }


def run_pair_benchmark(output: Path, *, protocol, sae_key=None, model_name=None, device="auto"):
    """Execute a named bounded pair protocol and audit its saved measurements."""
    from types import SimpleNamespace

    import torch

    from conf import load_group
    from models.loading_policy import resolve_device
    from saebase.api import ActivationSite, ModelAdapter, capture_features
    from saebase.api.sae_loading import load_model_and_sae
    from saebase.api.sae_profiles import get_profile
    from tasks.benchmark.execution.planner import build_plan
    from tasks.benchmark.execution.runner import execute as run_metrics
    from tasks.benchmark.execution.schema import BenchmarkConfig
    from tasks.benchmark.reliability import AuditConfig, evaluate_benchmark, read_audit, write_audit
    from tasks.benchmark.results.report import write_report
    from tasks.precheck.sae_checks import TARGETS, load_cpu_pair

    preset = load_group("benchmark", protocol).benchmark
    if protocol == "cpu_pair":
        if not sae_key:
            raise ValueError("cpu_pair requires sae_key")
        if device not in ("auto", "cpu"):
            raise ValueError("cpu_pair requires CPU execution")
        entry, profile, model, sae, meta, site, model_repo, raw = load_cpu_pair(sae_key)
        model_name = entry.model_key
        run_name = f"Stage B real CPU SAEBase Benchmark {sae_key}"
        dataset_id = f"stage-bc-real-{model_name}"
    elif protocol == "reference_pair":
        model_name = model_name or preset.model_key
        sae_key, layer, _ = TARGETS[model_name]
        profile = get_profile(sae_key)
        raw, sae, meta = load_model_and_sae(sae_key, layer, device=resolve_device(device))
        model = ModelAdapter(raw)
        site = ActivationSite(layer, meta["hook_name"])
        model_repo = meta.get("loaded_model_name", profile.model_name)
        run_name = (
            "Real Qwen3.5 2B Base with 32k TopK SAE"
            if model_name == "qwen35_2b"
            else f"Real {model_name} SAE Benchmark"
        )
        dataset_id = (
            "saebase-phase-a-qwen35-2b-real" if model_name == "qwen35_2b" else f"real-{model_name}"
        )
    else:
        raise ValueError(f"Unknown bounded pair protocol: {protocol}")
    first = capture_features(model, sae, site=site, prompt=preset.prompts[0])
    fid = int(torch.argmax(first.codes.abs()).item())
    if not bool(torch.isfinite(first.codes).all()) or float(first.codes[fid].abs()) == 0:
        raise ValueError("no real SAE feature activated for benchmark")
    samples = []
    for i, prompt in enumerate(preset.prompts):
        ids = model.tokens(prompt=prompt).detach().cpu().tolist()[0]
        samples.append(
            dict(
                sample_id=f"stage-bc-{model_name}-{i}"
                if protocol == "cpu_pair"
                else f"real-qwen35-2b-{i:02d}"
                if model_name == "qwen35_2b"
                else f"real-{model_name}-{i:02d}",
                token_ids=ids,
                split="test",
                prompt_group_id=f"stage-bc-group-{i}"
                if protocol == "cpu_pair"
                else f"real-sample-{i:02d}",
                selection={"mode": "indices", "indices": [len(ids) - 1]},
            )
        )
    config = BenchmarkConfig.from_dict(
        dict(
            schema="saebase.benchmark.config.v1",
            name=run_name,
            targets=[
                dict(
                    model_key=profile.model_key,
                    sae_keys=[sae_key],
                    layer=site.layer,
                    hook_name=site.hook_name,
                )
            ],
            dataset={"id": dataset_id, "samples": samples},
            metrics=[
                "core.l0",
                "core.reconstruction_mse",
                "core.explained_variance",
                "intervention.logit_delta_l2",
            ],
            conditions=[
                {"name": "identity", "operation": "identity"},
                {"name": preset.condition_name, "operation": "ablate", "feature_ids": [fid]},
            ],
            max_activation_rows=int(preset.max_activation_rows),
            max_measurements=int(preset.max_measurements),
            sample_batch=int(preset.sample_batch),
            microbatch=int(preset.microbatch),
            seed=int(preset.seed),
            tolerance=float(preset.tolerance),
        )
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = run_metrics(
        config,
        output,
        real=True,
        loader=lambda mk, sk, layer: SimpleNamespace(model=model, sae=sae),
    )
    if (
        not rows
        or any(row["status"] != "completed" for row in rows)
        or (protocol == "cpu_pair" and len(rows) != 5)
    ):
        raise ValueError("real pair benchmark has incomplete metrics")
    root = output / config.run_id
    audit = evaluate_benchmark(
        root,
        AuditConfig(
            evaluation_seeds=tuple(preset.audit.evaluation_seeds),
            bootstrap_replicates=int(preset.audit.bootstrap_replicates),
            sample_sizes=tuple(preset.sample_sizes),
            metric_ids=tuple(preset.audit.metric_ids),
        ),
    )
    if (
        preset.get("audit_count") is not None and len(audit["metric_audits"]) != preset.audit_count
    ) or any(row["status"] != "completed" for row in audit["metric_audits"]):
        raise ValueError("real pair reliability audit has incomplete metrics")
    write_audit(root, audit)
    write_report(root, config, build_plan(config), rows)
    if read_audit(root) != audit:
        raise ValueError("reliability audit failed integrity roundtrip")
    metric_sha = json.loads((root / "metrics_manifest.json").read_text())["sha256"]
    audit_sha = json.loads((root / "reliability_manifest.json").read_text())["audit_sha256"]
    record = dict(
        schema="saebase.coverage.cpu_pair_m3m4.v1"
        if protocol == "cpu_pair"
        else "saebase.phase_a.real_m3m4_acceptance.v1",
        status="benchmark_and_reliability_verified" if protocol == "cpu_pair" else "passed",
        sae_key=sae_key,
        layer=site.layer,
        feature_ablated=fid,
        sample_count=len(samples),
        m3_run_id=config.run_id,
        metric_results=[
            {
                "metric_id": x["metric_id"],
                "condition": x["condition_id"],
                "value": x["value"],
                "n_valid": x["n_valid"],
            }
            for x in rows
        ],
    )
    if protocol == "cpu_pair":
        record.update(
            model_key=model_name,
            model_repo=model_repo,
            sae_repo=profile.repo_id or profile.sae_release,
            metric_rows=len(rows),
            reliability_audits=len(audit["metric_audits"]),
            model_forwards_in_m4=0,
            m3_sha256=metric_sha,
            m4_sha256=audit_sha,
        )
        name = f"{sae_key}_m3m4_summary.json"
    else:
        record.update(
            model=model_name,
            model_loaded=True,
            sae_loaded=True,
            benchmark_metric_rows=len(rows),
            reliability_metric_audits=len(audit["metric_audits"]),
            reliability_pairs=len(audit["paired_comparisons"]),
            n_model_reloads_for_audit=0,
            m3_metrics_sha256=metric_sha,
            m4_audit_sha256=audit_sha,
        )
        name = f"real_{model_name}_m3m4_summary.json"
    (output / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record
