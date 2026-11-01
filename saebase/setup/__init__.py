"""Shared, study-agnostic setup orchestration.

Specific experiments choose datasets, models, SAE sites and budgets through
their own existing conf/experiment configurations. This package orchestrates
the existing SAEBase Runtime and never changes scientific parameters.
"""

from .workflow import estimate_runtime, representative_sites, setup_report

__all__ = ["estimate_runtime", "representative_sites", "setup_report"]
