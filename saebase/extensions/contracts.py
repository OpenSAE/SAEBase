"""Versioned contracts for explicitly installed third-party SAEBase extensions.

Only an explicit registration or a requested entry point discovery may load
external Python code. Declaring a source is NOT checkpoint verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from saebase.api.model import ActivationSite
from saebase.benchmarks.schema import MetricSpec

API_VERSION = 1
ENTRY_POINT_GROUP = "opensae.saebase.extensions.v1"
_KEY = re.compile(r"ext\.[a-z][a-z0-9_]+(?:\.[a-z][a-z0-9_]*)+")
_DTYPES = frozenset({"float16", "float32", "float64", "bfloat16"})
_NORMALIZATIONS = frozenset({"none", "layer_norm", "unit_norm", "folded", "custom"})


def _key(value: str) -> None:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise ValueError("extension IDs must be namespaced as ext.vendor.name")


def _nonempty(value: str, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a nonempty string")


def _revision(value: str, what: str) -> None:
    _nonempty(value, what)
    if value.strip().lower() in {"main", "master", "head", "latest"}:
        raise ValueError(f"{what} must identify an immutable source revision")


@dataclass(frozen=True)
class HookSpec:
    site: ActivationSite
    width: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("hook width must be a positive integer")


@dataclass(frozen=True)
class ExternalModel:
    key: str
    source: str
    revision: str
    hooks: tuple[HookSpec, ...]
    factory: Callable[[str], Any] = field(repr=False)
    backend: str = "auto"

    def __post_init__(self) -> None:
        _key(self.key)
        _nonempty(self.source, "model source")
        _revision(self.revision, "model revision")
        if not callable(self.factory):
            raise TypeError("model factory must be callable")
        if not isinstance(self.hooks, tuple) or not self.hooks:
            raise ValueError("model hooks must be a nonempty tuple")
        keys = []
        for item in self.hooks:
            if not isinstance(item, HookSpec):
                raise TypeError("model hooks must be HookSpec records")
            keys.append((item.site.layer, item.site.hook_name))
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate model hook site")


@dataclass(frozen=True)
class ExternalSAE:
    key: str
    model_key: str
    source: str
    revision: str
    site: ActivationSite
    d_in: int
    d_sae: int
    architecture: str
    normalization: str
    dtype: str
    factory: Callable[[str], Any] = field(repr=False)
    sae_id: str = "default"

    def __post_init__(self) -> None:
        _key(self.key)
        _key(self.model_key)
        _nonempty(self.source, "SAE checkpoint source")
        _revision(self.revision, "SAE checkpoint revision")
        _nonempty(self.architecture, "SAE architecture")
        _nonempty(self.sae_id, "SAE checkpoint identifier")
        if not isinstance(self.site, ActivationSite):
            raise TypeError("SAE site must be an ActivationSite")
        for name in ("d_in", "d_sae"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError("SAE normalization must be explicit and supported")
        if self.dtype not in _DTYPES:
            raise ValueError("SAE dtype must be an explicit floating-point dtype")
        if not callable(self.factory):
            raise TypeError("SAE factory must be callable")


@dataclass(frozen=True)
class ExternalObservation:
    """A result value from original third-party metric code, not a global score."""

    value: float
    n_valid: int
    measurement_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        import math

        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("external benchmark result must be finite")
        if type(self.n_valid) is not int or self.n_valid <= 0:
            raise ValueError("external benchmark needs a positive observation count")
        if not isinstance(self.measurement_refs, tuple) or not all(
            isinstance(ref, str) and ref for ref in self.measurement_refs
        ):
            raise ValueError("measurement_refs must be a tuple of nonempty paths")


@dataclass(frozen=True)
class ExternalMetricContext:
    config: Any
    target: Any
    sae_key: str
    model: Any
    sae: Any
    activation_store: Path | None
    code_store: Path | None


@dataclass(frozen=True)
class ExternalMetric:
    spec: MetricSpec
    source: str
    revision: str
    source_file: Path
    evaluator: Callable[[ExternalMetricContext], ExternalObservation] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, MetricSpec):
            raise TypeError("external benchmark must provide a MetricSpec")
        _key(self.spec.metric_id)
        _nonempty(self.source, "benchmark source")
        _revision(self.revision, "benchmark revision")
        if not isinstance(self.source_file, Path):
            raise TypeError("source_file must be a Path")
        if not callable(self.evaluator):
            raise TypeError("benchmark evaluator must be callable")
        permitted = {"external_pair", "m1_activations", "m1_codes"}
        if not set(self.spec.required_inputs) <= permitted:
            raise ValueError("unsupported external metric inputs")
        if not self.spec.required_inputs:
            raise ValueError("external metric must declare required inputs")


@dataclass(frozen=True)
class ExtensionBundle:
    name: str
    models: tuple[ExternalModel, ...] = ()
    saes: tuple[ExternalSAE, ...] = ()
    metrics: tuple[ExternalMetric, ...] = ()
    api_version: int = API_VERSION

    def __post_init__(self) -> None:
        _key(self.name)
        if self.api_version != API_VERSION:
            raise ValueError(f"extension API version must be {API_VERSION}")
        for field_name, kind in (
            ("models", ExternalModel),
            ("saes", ExternalSAE),
            ("metrics", ExternalMetric),
        ):
            seq = getattr(self, field_name)
            if not isinstance(seq, tuple) or not all(isinstance(x, kind) for x in seq):
                raise TypeError(f"{field_name} must be a tuple of {kind.__name__}")
        if not (self.models or self.saes or self.metrics):
            raise ValueError("extension bundle has no providers")
