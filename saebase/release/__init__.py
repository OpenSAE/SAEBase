"""M5 two-family publication matrix and release evidence (no model training)."""
from .schema import MatrixConfig, MatrixTarget, load_matrix
from .planner import plan_matrix
__all__=["MatrixConfig","MatrixTarget","load_matrix","plan_matrix"]
from .gates import audit_result, audit_inventory, audit_distributions, generate_release_evidence
__all__ += ["audit_result", "audit_inventory", "audit_distributions", "generate_release_evidence"]
