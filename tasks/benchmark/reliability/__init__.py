"""Reliability analysis. Never modifies a benchmark metric or its raw observations."""

from tasks.benchmark.reliability.bootstrap import (
    bootstrap_metric,
    group_samples,
    paired_bootstrap,
    sample_size_curve,
)

from .schema import AuditConfig, Observation

__all__ = [
    "AuditConfig",
    "Observation",
    "bootstrap_metric",
    "paired_bootstrap",
    "sample_size_curve",
    "group_samples",
]
from tasks.benchmark.results.audit import read_audit, write_audit

from .runner import evaluate_benchmark

__all__ += ["evaluate_benchmark", "write_audit", "read_audit"]
from tasks.benchmark.reliability.controls import (  # noqa: E402
    append_ground_truth,
    evaluate_ground_truth,
)

__all__ += ["evaluate_ground_truth", "append_ground_truth"]
