"""Configure, plan and execute scientific benchmarks using task-owned implementations."""

from __future__ import annotations

from tasks.benchmark.execution.planner import build_plan
from tasks.benchmark.execution.schema import (
    BenchmarkConfig,
    BenchmarkTarget,
    MetricResult,
    MetricSpec,
    load_config,
)
from tasks.benchmark.metrics.registry import get_metric, metric_specs, register_metric

__all__ = [
    "BenchmarkConfig",
    "BenchmarkTarget",
    "MetricResult",
    "MetricSpec",
    "load_config",
    "get_metric",
    "metric_specs",
    "register_metric",
    "build_plan",
]
