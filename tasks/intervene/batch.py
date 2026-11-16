"""Task-owned batched feature edits using the existing measurement engine."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict
from tasks.intervene.backend import (
    BatchEngine,
    Condition,
    ExperimentIdentity,
    GenerationSpec,
    MeasurementSample,
    ResultWriter,
    RunBudget,
    dry_run,
    read_results,
    run_steering,
)


def parse_plan(config: dict):
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a JSON object")
    allowed = {
        "model_key",
        "sae_key",
        "layer",
        "samples",
        "conditions",
        "pad_token_id",
        "sample_batch",
        "microbatch",
        "max_measurements",
        "max_generated_tokens",
        "max_seconds",
        "steering",
        "tolerance",
    }
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"unknown configuration fields: {extra}")
    for key in ("model_key", "sae_key", "layer", "samples", "conditions"):
        if key not in config:
            raise ValueError(f"missing required experiment field: {key}")
    samples = []
    for row in config["samples"]:
        data = dict(row)
        position = data.pop("patch_position", -1)
        readout = data.pop("readout_position", -1)
        target = data.pop("target_token_id", None)
        reference = data.pop("reference_token_id", None)
        record = SampleRecord(
            sample_id=data["sample_id"],
            token_ids=tuple(data["token_ids"]),
            selection=TokenSelection("last_valid")
            if position == -1
            else TokenSelection("indices", (position,)),
            attention_mask=tuple(data["attention_mask"])
            if data.get("attention_mask") is not None
            else None,
            split=data.get("split", "evaluation"),
            prompt_group_id=data.get("prompt_group_id"),
        )
        samples.append(
            MeasurementSample(
                record,
                readout_position=readout,
                target_token_id=target,
                reference_token_id=reference,
            )
        )
    conditions = [
        Condition(
            name=c["name"],
            operation=c["operation"],
            feature_ids=tuple(c.get("feature_ids", ())),
            value=c.get("value", 0.0),
            retained_ids=tuple(c.get("retained_ids", ())),
        )
        for c in config["conditions"]
    ]
    budget = RunBudget(
        max_measurements=config.get("max_measurements", 100_000),
        max_generated_tokens=config.get("max_generated_tokens", 10_000),
        max_seconds=config.get("max_seconds"),
    )
    plan = dry_run(samples, conditions, microbatch=config.get("microbatch", 16), budget=budget)
    spec = None
    if "steering" in config:
        raw = config["steering"]
        if not isinstance(raw, dict) or "condition_name" not in raw:
            raise ValueError("steering must specify condition_name")
        candidates = [c for c in conditions if c.name == raw["condition_name"]]
        if len(candidates) != 1:
            raise ValueError("steering condition_name must match exactly one condition")
        spec = GenerationSpec(
            max_new_tokens=raw.get("max_new_tokens", 8),
            seed=raw.get("seed", 0),
            temperature=raw.get("temperature", 0.0),
            top_k=raw.get("top_k"),
            eos_token_ids=tuple(raw.get("eos_token_ids", [])),
            max_context_tokens=raw.get("max_context_tokens"),
        )
        if 2 * len(samples) * spec.max_new_tokens > budget.max_generated_tokens:
            raise ValueError("steering exceeds token budget")
        plan["steering_total_max_generated_tokens"] = 2 * len(samples) * spec.max_new_tokens
    return config, samples, conditions, budget, plan, spec


def configured_plan(cfg):
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(cfg.task.protocol, resolve=True)
    reserved = {"model_key", "sae_key", "layer", "samples"} & set(raw)
    if reserved:
        raise ValueError("Select resources globally and provide inputs in dataset.samples")
    from tasks.sae_runtime import resolve_sae

    selected = resolve_sae(
        cfg.model, cfg.task.get("sae_key"), cfg.task.get("layer"), operation="intervene"
    )
    return {
        **raw,
        "model_key": selected.model_key,
        "sae_key": selected.sae_key,
        "layer": selected.layer,
        "samples": OmegaConf.to_container(cfg.dataset.samples, resolve=True),
    }


def run_batch(pair, config: dict, output: Path, *, resume: bool = True) -> dict:
    config, samples, conditions, budget, plan, generation = parse_plan(config)
    identity = ExperimentIdentity(
        sha256_state_dict(pair.model.model),
        sha256_state_dict(pair.sae),
        pair.site,
        model_key=config["model_key"],
        sae_key=config["sae_key"],
        backend=pair.model.backend,
    )
    engine = BatchEngine(pair.model, pair.sae, identity, pad_token_id=config.get("pad_token_id"))
    with ResultWriter(output / "measurements", identity, resume=resume) as writer:
        measured = engine.run(
            samples,
            conditions,
            writer,
            sample_batch=config.get("sample_batch", 8),
            microbatch=config.get("microbatch", 16),
            budget=budget,
            tolerance=config.get("tolerance", 1e-4),
        )
    rows = read_results(output / "measurements", identity)
    from saebase.results import completion_summary

    completion = completion_summary(
        [{"status": "completed"} for _ in rows], expected=plan["measurements"], unit="measurement"
    )
    result = {
        "status": "executed",
        "operation": "intervene",
        "model_key": config["model_key"],
        "sae_key": config["sae_key"],
        "layer": config["layer"],
        "experiment": identity.to_dict(),
        "measurements": asdict(measured),
        "resolved_plan": plan,
        "completion": completion,
        "input_scope": {"unit": "sample", "requested": len(samples), "processed": len(samples)},
    }
    if generation is not None:
        chosen = next(c for c in conditions if c.name == config["steering"]["condition_name"])
        with ResultWriter(output / "steering", identity, resume=resume) as writer:
            result["steering"] = run_steering(
                pair.model, pair.sae, identity, samples, chosen, generation, writer, budget=budget
            )
    result["required_artifacts"] = [
        {
            "path": str(path.relative_to(Path.cwd()))
            if path.is_relative_to(Path.cwd())
            else str(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.glob("*/manifest.json")) + sorted(output.glob("*/shard-*.jsonl"))
    ]
    return result
