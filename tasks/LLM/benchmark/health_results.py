"""Summaries and optional JSON writers for SAE evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def write_run_summary(payload: dict[str, Any], path: Path | None = None) -> Path:
    out = path or Path.cwd() / "sae_activation_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [r for r in rows if r.get("ok")]
    fails = [r for r in rows if not r.get("ok")]
    l0s = sorted(float(r["l0"]) for r in oks)
    mses = sorted(float(r["reconstruction_mse"]) for r in oks)
    fves = sorted(float(r["fve"]) for r in oks if "fve" in r)

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])

    return {
        "n_ok": len(oks),
        "n_fail": len(fails),
        "l0_median": _median(l0s),
        "reconstruction_mse_median": _median(mses),
        "fve_median": _median(fves) if fves else None,
        "fve_mean": (sum(fves) / len(fves)) if fves else None,
        "failures": [
            {"prompt_idx": r.get("prompt_idx"), "error": r.get("error")} for r in fails
        ],
    }


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
