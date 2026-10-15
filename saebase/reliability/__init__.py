"""M4 reliability audit. Never modifies a benchmark metric or its raw observations."""
from .schema import AuditConfig, Observation
from .bootstrap import bootstrap_metric, paired_bootstrap, sample_size_curve
from .resampling import group_samples
__all__=["AuditConfig","Observation","bootstrap_metric","paired_bootstrap",
         "sample_size_curve","group_samples"]
