"""Explicit third-party extension API for models, SAEs and benchmark metrics.

To load installed entry-point packages, opt in to discover_extensions(names).
Importing SAEBase never imports arbitrary third-party extension packages.
"""

from .contracts import (
    API_VERSION,
    ENTRY_POINT_GROUP,
    ExtensionBundle,
    ExternalMetric,
    ExternalMetricContext,
    ExternalModel,
    ExternalObservation,
    ExternalSAE,
    HookSpec,
)
from .registry import (
    discover_extensions,
    external_metric,
    has_external_pair,
    is_external_metric,
    list_extensions,
    register_extension,
)
from .runtime import load_external_pair, validate_external_target
from .metrics import execute_external_metric, validate_external_metric_source

__all__ = [
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "ExtensionBundle",
    "ExternalMetric",
    "ExternalMetricContext",
    "ExternalModel",
    "ExternalObservation",
    "ExternalSAE",
    "HookSpec",
    "discover_extensions",
    "external_metric",
    "has_external_pair",
    "is_external_metric",
    "list_extensions",
    "register_extension",
    "load_external_pair",
    "validate_external_target",
    "execute_external_metric",
    "validate_external_metric_source",
]
