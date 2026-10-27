"""Execution and source-integrity checks for explicit third-party metrics."""

from __future__ import annotations

from pathlib import Path

from .contracts import ExternalMetricContext, ExternalObservation
from .registry import external_metric, metric_source_sha256


def execute_external_metric(
    metric_id: str, context: ExternalMetricContext
) -> tuple[ExternalObservation, str, str]:
    """Call the original estimator only for a configured, pinned source file."""
    plugin = external_metric(metric_id)
    if plugin is None:
        raise ValueError(f"no evaluator for external metric {metric_id}")
    configured = context.config.sources.get(metric_id)
    if not isinstance(configured, str) or (
        Path(configured).resolve() != plugin.source_file.resolve()
    ):
        raise ValueError(f"sources.{metric_id} must point to the registered original source")
    digest = metric_source_sha256(metric_id)
    result = plugin.evaluator(context)
    if not isinstance(result, ExternalObservation):
        raise TypeError("external estimator must return ExternalObservation")
    return result, digest, plugin.revision


def validate_external_metric_source(metric_id: str, record: dict, sources: dict) -> None:
    """Read-only resume/report gate: rehash original estimator source."""
    plugin = external_metric(metric_id)
    if plugin is None:
        raise ValueError(f"original external metric is not registered: {metric_id}")
    configured = sources.get(metric_id)
    if not isinstance(configured, str) or (
        Path(configured).resolve() != plugin.source_file.resolve()
    ):
        raise ValueError("external metric provenance/source location changed")
    if (
        record.get("source_sha256") != metric_source_sha256(metric_id)
        or record.get("source_commit") != plugin.revision
    ):
        raise ValueError("external metric source changed after benchmark computation")
