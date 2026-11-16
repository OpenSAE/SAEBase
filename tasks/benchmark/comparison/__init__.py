"""Compare current model/SAE targets using benchmarks and grouped reliability analysis."""
from .schema import MatrixConfig, MatrixTarget, load_matrix
from .planner import plan_matrix
__all__ = ["MatrixConfig", "MatrixTarget", "load_matrix", "plan_matrix"]
