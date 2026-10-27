"""Concrete benchmark task implementation; SAEBase supplies reusable primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tasks.LLM.sae_runtime import MODEL_KEY, _task


def _benchmark(cfg: Any) -> dict[str, Any]:
    from tasks.LLM.benchmark.backend.schema import load_config
    from tasks.LLM.benchmark.backend.runner import execute

    path = Path(str(_task(cfg, "benchmark_config", "")))
    if not path.is_file():
        raise FileNotFoundError("Provide an existing SAEBase benchmark_config JSON/YAML")
    config = load_config(path)
    if any(target.model_key != MODEL_KEY for target in config.targets):
        raise ValueError("Benchmark targets must use Gemma-2 2B")
    values = execute(
        config, Path.cwd() / "metrics", real=True, resume=bool(_task(cfg, "resume", True))
    )
    return {
        "status": "executed",
        "operation": "benchmark",
        "model_key": MODEL_KEY,
        "benchmark_id": config.run_id,
        "metric_count": len(values),
        "metrics": [v.to_dict() if hasattr(v, "to_dict") else v for v in values],
    }


def execute(cfg: Any) -> dict[str, Any]:
    return _benchmark(cfg)
