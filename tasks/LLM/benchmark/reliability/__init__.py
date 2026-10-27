"""M4 reliability audit. Never modifies a benchmark metric or its raw observations."""

from .schema import AuditConfig, Observation
from .bootstrap import bootstrap_metric, paired_bootstrap, sample_size_curve
from .resampling import group_samples

__all__ = [
    "AuditConfig",
    "Observation",
    "bootstrap_metric",
    "paired_bootstrap",
    "sample_size_curve",
    "group_samples",
]
from .runner import evaluate_m3
from .report import write_audit, read_audit

__all__ += ["evaluate_m3", "write_audit", "read_audit"]
from .ground_truth_report import evaluate_ground_truth, append_ground_truth

__all__ += ["evaluate_ground_truth", "append_ground_truth"]
