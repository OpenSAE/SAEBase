"""Versioned, deliberately strict benchmark/metric contracts.

The resolved configuration is a deterministic identity; heavyweight model
loading and external estimator imports are never performed during validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from saebase.api import ActivationSite
from saebase.interventions import Condition, MeasurementSample
from saebase.store import SampleRecord, TokenSelection, fingerprint

CONFIG_SCHEMA = "saebase.benchmark.config.v1"
RESULT_SCHEMA = "saebase.benchmark.metric.v1"
STATUSES = frozenset({"completed", "failed", "unsupported", "blocked", "undefined"})
UNITS = frozenset({"prompt", "prompt_group", "variant", "token", "coalition", "measurement", "dataset"})
DIRECTIONS = frozenset({"higher", "lower", "contextual", "none"})

def _fields(value: Any, allowed: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {what} fields: {sorted(extra)}")
    return value

def _str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a nonempty string")
    return value

def _positive(value: Any, what: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or (value < 0 if allow_zero else value < 1):
        raise ValueError(f"{what} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value

@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    version: str
    required_inputs: tuple[str, ...]
    sample_unit: str
    direction: str = "contextual"
    aggregation: str = "mean"
    protocol_id: str = "saebase.m3.v1"

    def __post_init__(self) -> None:
        _str(self.metric_id, "metric_id")
        _str(self.version, "metric_version")
        _str(self.protocol_id, "protocol_id")
        if self.sample_unit not in UNITS or self.direction not in DIRECTIONS:
            raise ValueError("unknown metric unit or direction")
        if not isinstance(self.required_inputs, tuple) or not self.required_inputs:
            raise ValueError("required_inputs must be nonempty")

@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    metric_version: str
    protocol_id: str
    model_id: str
    sae_id: str
    site: dict[str, Any]
    dataset_id: str
    split: str
    seed: int
    sample_unit: str
    n_requested: int
    n_valid: int
    status: str
    value: float | None = None
    condition_id: str | None = None
    source_commit: str | None = None
    source_sha256: str | None = None
    measurement_refs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    reason: str | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES or self.sample_unit not in UNITS:
            raise ValueError("invalid MetricResult status/unit")
        if type(self.n_valid) is not int or type(self.n_requested) is not int or not 0 <= self.n_valid <= self.n_requested:
            raise ValueError("invalid sample count")
        if self.status == "completed" and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("completed metric needs a finite value")
        if self.status != "completed" and self.value is not None:
            raise ValueError("noncompleted metric must have null value")
        if self.status != "completed" and not self.reason:
            raise ValueError("noncompleted metric needs a reason")
        if self.value is not None and (type(self.value) not in (float, int) or not math.isfinite(self.value)):
            raise ValueError("metric value must be finite")
        if self.status == "completed" and self.n_valid == 0:
            raise ValueError("completed metric must contain observations")
        fingerprint(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({"schema": RESULT_SCHEMA, **asdict(self)}, allow_nan=False))

@dataclass(frozen=True)
class BenchmarkTarget:
    model_key: str
    sae_keys: tuple[str, ...]
    site: ActivationSite

    def __post_init__(self) -> None:
        _str(self.model_key, "model_key")
        if not self.sae_keys or len(set(self.sae_keys)) != len(self.sae_keys):
            raise ValueError("sae_keys must be unique and nonempty")
        for key in self.sae_keys:
            _str(key, "sae_key")

@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    targets: tuple[BenchmarkTarget, ...]
    dataset_id: str
    samples: tuple[MeasurementSample, ...]
    metrics: tuple[str, ...]
    conditions: tuple[Condition, ...] = ()
    seed: int = 0
    max_measurements: int = 100000
    max_activation_rows: int = 100000
    microbatch: int = 16
    sample_batch: int = 8
    tolerance: float = 1e-4
    sources: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        _str(self.dataset_id, "dataset_id")
        if not self.targets or not self.samples or not self.metrics:
            raise ValueError("targets, samples and metrics must be nonempty")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("duplicate metric IDs")
        if len({(t.model_key, t.site.hook_name, t.site.layer, s) for t in self.targets for s in t.sae_keys}) != sum(len(t.sae_keys) for t in self.targets):
            raise ValueError("duplicate target/SAE pairing")
        if len({s.record.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("duplicate sample IDs")
        if len({c.name for c in self.conditions}) != len(self.conditions):
            raise ValueError("duplicate condition names")
        for fld in ("max_measurements", "max_activation_rows", "microbatch", "sample_batch"):
            _positive(getattr(self, fld), fld)
        _positive(self.seed, "seed", allow_zero=True)
        if type(self.tolerance) not in (float, int) or not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not isinstance(self.sources, dict) or not isinstance(self.options, dict):
            raise ValueError("sources/options must be objects")
        fingerprint(self.sources)
        fingerprint(self.options)
        # Train/test leakage is detected at the independent prompt-group level.
        groups: dict[str, set[str]] = {}
        for sample in self.samples:
            if sample.record.prompt_group_id:
                groups.setdefault(sample.record.prompt_group_id, set()).add(sample.record.split)
        for group, splits in groups.items():
            if "train" in splits and ("test" in splits or "held_out" in splits):
                raise ValueError(f"train/test prompt group leakage: {group}")

    @property
    def run_id(self) -> str:
        return fingerprint(self.to_dict())[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA, "name": self.name,
            "targets": [{"model_key": t.model_key, "sae_keys": list(t.sae_keys),
                         "layer": t.site.layer, "hook_name": t.site.hook_name,
                         "position": t.site.position} for t in self.targets],
            "dataset": {"id": self.dataset_id, "samples": [
                {**s.record.to_dict(), "readout_position": s.readout_position,
                 "target_token_id": s.target_token_id, "reference_token_id": s.reference_token_id}
                for s in self.samples]},
            "metrics": list(self.metrics),
            "conditions": [c.to_dict() for c in self.conditions],
            "seed": self.seed, "max_measurements": self.max_measurements,
            "max_activation_rows": self.max_activation_rows,
            "microbatch": self.microbatch, "sample_batch": self.sample_batch,
            "tolerance": self.tolerance, "sources": self.sources, "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkConfig":
        raw = _fields(dict(data), {"schema", "name", "targets", "dataset", "metrics", "conditions",
                                 "seed", "max_measurements", "max_activation_rows", "microbatch",
                                 "sample_batch", "tolerance", "sources", "options"}, "benchmark")
        if raw.get("schema") != CONFIG_SCHEMA:
            raise ValueError(f"benchmark schema must be {CONFIG_SCHEMA}")
        targets = []
        for entry in raw["targets"]:
            e = _fields(entry, {"model_key", "sae_keys", "layer", "hook_name", "position"}, "target")
            targets.append(BenchmarkTarget(e["model_key"], tuple(e["sae_keys"]),
                                           ActivationSite(e["layer"], e["hook_name"], e.get("position", -1))))
        dataset = _fields(raw["dataset"], {"id", "samples"}, "dataset")
        samples = []
        for row in dataset["samples"]:
            r = _fields(row, {"sample_id", "token_ids", "selection", "attention_mask", "prompt_group_id",
                              "split", "variant_role", "token_offsets", "source_activation_sha256",
                              "readout_position", "target_token_id", "reference_token_id"}, "sample")
            record = SampleRecord.from_dict({"selection": {"mode": "last_valid"}, **{k:v for k,v in r.items()
                if k not in {"readout_position", "target_token_id", "reference_token_id"}}})
            samples.append(MeasurementSample(record, readout_position=r.get("readout_position", -1),
                                             target_token_id=r.get("target_token_id"),
                                             reference_token_id=r.get("reference_token_id")))
        conditions = []
        for entry in raw.get("conditions", []):
            e = _fields(entry, {"name", "operation", "feature_ids", "value", "retained_ids"}, "condition")
            conditions.append(Condition(name=e["name"], operation=e["operation"],
                                        feature_ids=tuple(e.get("feature_ids", ())),
                                        retained_ids=tuple(e.get("retained_ids", ())),
                                        value=e.get("value", 0.0)))
        from .registry import get_metric
        metrics = tuple(raw["metrics"])
        for metric_id in metrics:
            get_metric(metric_id)
        return cls(name=raw["name"], targets=tuple(targets), dataset_id=dataset["id"],
                   samples=tuple(samples), metrics=metrics, conditions=tuple(conditions),
                   seed=raw.get("seed", 0), max_measurements=raw.get("max_measurements", 100000),
                   max_activation_rows=raw.get("max_activation_rows", 100000),
                   microbatch=raw.get("microbatch", 16), sample_batch=raw.get("sample_batch", 8),
                   tolerance=raw.get("tolerance", 1e-4), sources=raw.get("sources", {}),
                   options=raw.get("options", {}))

def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require the optional PyYAML dependency; JSON is always supported") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("benchmark config must be .json, .yaml or .yml")
    return BenchmarkConfig.from_dict(data)
