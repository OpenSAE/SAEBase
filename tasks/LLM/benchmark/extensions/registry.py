"""Explicit, collision-safe extension registration and entry-point discovery."""

from __future__ import annotations

import hashlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Iterable

from .contracts import (
    ENTRY_POINT_GROUP,
    ExtensionBundle,
    ExternalMetric,
    ExternalModel,
    ExternalSAE,
)

_MODELS: dict[str, ExternalModel] = {}
_SAES: dict[str, ExternalSAE] = {}
_METRICS: dict[str, ExternalMetric] = {}
_METRIC_DIGESTS: dict[str, str] = {}
_BUNDLES: set[str] = set()
_DISCOVERED: set[str] = set()


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def register_extension(bundle: ExtensionBundle) -> None:
    """Register a fully validated bundle, refusing namespace clashes.

    Does not download weights or run external metric code. Only the explicit
    caller executes this function. All preflight checks run before mutation.
    """
    if not isinstance(bundle, ExtensionBundle):
        raise TypeError("expected an ExtensionBundle")
    if bundle.name in _BUNDLES:
        raise ValueError(f"extension already registered: {bundle.name}")
    from dlbase.registry import list_registered_model_keys
    from saebase.sae.registry import SAE_KEYS, DEFAULT_SAE_FOR_MODEL
    from tasks.LLM.benchmark.backend.registry import metric_specs, register_metric

    model_keys = [x.key for x in bundle.models]
    sae_keys = [x.key for x in bundle.saes]
    metric_keys = [x.spec.metric_id for x in bundle.metrics]
    for keys, existing, kind in (
        (model_keys, set(_MODELS) | set(list_registered_model_keys()), "model"),
        (sae_keys, set(_SAES) | set(SAE_KEYS) | set(DEFAULT_SAE_FOR_MODEL) | {"none"}, "SAE"),
        (metric_keys, set(_METRICS) | {x.metric_id for x in metric_specs()}, "metric"),
    ):
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate {kind} keys in extension")
        if set(keys) & existing:
            raise ValueError(f"{kind} collides with registered key: {sorted(set(keys) & existing)}")

    available_models = dict(_MODELS)
    available_models.update({x.key: x for x in bundle.models})
    for sae in bundle.saes:
        model = available_models.get(sae.model_key)
        if model is None:
            raise ValueError(f"external SAE requires registered external model: {sae.model_key}")
        widths = {(hook.site.layer, hook.site.hook_name): hook.width for hook in model.hooks}
        actual = widths.get((sae.site.layer, sae.site.hook_name))
        if actual is None:
            raise ValueError(f"SAE hook {sae.site.hook_name!r} not registered by model")
        if actual != sae.d_in:
            raise ValueError(f"model hook width {actual} differs from SAE d_in {sae.d_in}")

    digest_by_key = {}
    for metric in bundle.metrics:
        if not metric.source_file.is_file():
            raise ValueError(f"external metric source file absent: {metric.source_file}")
        digest_by_key[metric.spec.metric_id] = _digest(metric.source_file)
    for metric in bundle.metrics:
        register_metric(metric.spec)
    _MODELS.update({x.key: x for x in bundle.models})
    _SAES.update({x.key: x for x in bundle.saes})
    _METRICS.update({x.spec.metric_id: x for x in bundle.metrics})
    _METRIC_DIGESTS.update(digest_by_key)
    _BUNDLES.add(bundle.name)


def discover_extensions(names: Iterable[str]) -> tuple[str, ...]:
    """Opt-in to installed entry points by exact name; nothing auto-loads.

    External entry point code is trusted only when the caller explicitly
    requests that distribution. Unlike model metadata, Python code execution
    is not a sandbox.
    """
    requested = tuple(names)
    if not requested or any(not isinstance(x, str) or not x for x in requested):
        raise ValueError("pass nonempty installed entry-point names explicitly")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate requested extension names")
    available = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        if entry.name in available:
            raise ValueError(f"duplicate installed entry-point name {entry.name!r}")
        available[entry.name] = entry
    for name in requested:
        if name not in available:
            raise ValueError(f"extension entry point {name!r} not installed")
        if name in _DISCOVERED:
            raise ValueError(f"extension already discovered: {name}")
    loaded = []
    for name in requested:
        factory = available[name].load()
        if not callable(factory):
            raise TypeError(f"extension {name!r} entry point must be a callable")
        bundle = factory()
        register_extension(bundle)
        _DISCOVERED.add(name)
        loaded.append(name)
    return tuple(loaded)


def external_model(key: str) -> ExternalModel | None:
    return _MODELS.get(key)


def external_sae(key: str) -> ExternalSAE | None:
    return _SAES.get(key)


def external_metric(metric_id: str) -> ExternalMetric | None:
    return _METRICS.get(metric_id)


def is_external_metric(metric_id: str) -> bool:
    return metric_id in _METRICS


def has_external_pair(model_key: str, sae_key: str) -> bool:
    return model_key in _MODELS and sae_key in _SAES


def metric_source_sha256(metric_id: str) -> str:
    metric = _METRICS.get(metric_id)
    if metric is None:
        raise ValueError(f"external metric is not registered: {metric_id}")
    digest = _digest(metric.source_file)
    if digest != _METRIC_DIGESTS[metric_id]:
        raise ValueError(f"external metric source changed after registration: {metric_id}")
    return digest


def list_extensions() -> dict[str, tuple[str, ...]]:
    """Report installed registrations, never inferred checkpoint verification."""
    return {
        "bundles": tuple(sorted(_BUNDLES)),
        "models": tuple(sorted(_MODELS)),
        "saes": tuple(sorted(_SAES)),
        "metrics": tuple(sorted(_METRICS)),
    }
