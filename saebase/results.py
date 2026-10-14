"""Structured evaluation results shared across SAEBase tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Return value of an SAE evaluation run."""

    ok: bool
    task: str
    evaluation: str
    model_key: str
    prompt_source: str
    sae_key: str | None = None
    layer: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    per_prompt: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
