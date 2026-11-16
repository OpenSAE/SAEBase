"""Expose configured benchmarks and activation-health evaluation as one task.

The selected evaluation owns its numerical and artifact protocol.
"""

from __future__ import annotations

from saebase.results import EvaluationResult
from tasks.sae_runtime import SAEOperationTask


class BenchmarkTask(SAEOperationTask):
    OPERATION = "benchmark"
    supports_multiple_models = True

    def supports_run_resume(self, cfg) -> bool:
        return cfg.task.get("evaluation") != "activation_health"

    def validate_cfg(self, cfg) -> None:
        super().validate_cfg(cfg)
        if cfg.task.get("evaluation") == "activation_health":
            from saebase.registry import get_prompt_source_plugin, get_sae_plugin

            for group in ("prompt_source", "sae", "evaluation"):
                if group not in cfg:
                    raise ValueError(f"Missing activation-health configuration: {group}")
            if str(cfg.sae.name) == "none":
                raise ValueError("Activation health requires an actual SAE profile")
            get_sae_plugin(str(cfg.sae.name)).assert_compatible(cfg)
            get_prompt_source_plugin(str(cfg.prompt_source.name)).assert_compatible(cfg)

    def compare_results(self, left, right, *, pairs=None) -> dict:
        from tasks.benchmark.results.metrics import compare_benchmark_runs

        return compare_benchmark_runs(left, right, pairs=pairs)

    def run_validated(self, cfg) -> EvaluationResult:
        if cfg.task.get("evaluation") == "activation_health":
            from tasks.benchmark.metrics.health import run_activation_health

            return run_activation_health(cfg)
        configured = cfg.task.get("benchmark_config") or cfg.task.get("protocol") is not None
        if configured and not cfg.task.get("execute", False):
            from saebase.runtime.experiment_paths import run_components
            from tasks.benchmark.operation import configured_benchmark, resolved_protocol

            _, config = configured_benchmark(cfg)
            components = run_components(cfg)
            return EvaluationResult.from_payload(
                "benchmark",
                cfg,
                {
                    "status": "configured_only",
                    "model_key": config.targets[0].model_key
                    if len({target.model_key for target in config.targets}) == 1
                    else components["model"],
                    "resolved_protocol": resolved_protocol(config, cfg),
                },
            )
        return super().run_validated(cfg)

    def required_result_artifacts(self, result):
        declared = super().required_result_artifacts(result)
        if declared:
            return declared
        run_id = result.summary.get("benchmark_id")
        if run_id:
            return [
                {"path": f"metrics/{run_id}/{name}"}
                for name in ("metrics.jsonl", "metrics_manifest.json")
            ]
        comparison = result.summary.get("comparison", {})
        if comparison.get("matrix_id"):
            root = f"metrics/{comparison['matrix_id']}"
            paths = [f"{root}/matrix_manifest.json", f"{root}/matrix_results.json"]
            for model in comparison.get("models", []):
                paths.extend(
                    f"{root}/benchmarks/{model['m3_run_id']}/{name}"
                    for name in ("metrics.jsonl", "metrics_manifest.json", "provenance.json")
                )
            return [{"path": path} for path in paths]
        return []
