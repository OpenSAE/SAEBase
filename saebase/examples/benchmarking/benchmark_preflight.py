"""No downloads: inspect SAEBase deterministic benchmark plan."""

from pathlib import Path
from saebase.benchmarks import load_config, build_plan

if __name__ == "__main__":
    config = load_config(
        Path(__file__).resolve().parents[2] / "conf/benchmark/pythia_benchmark_smoke.json"
    )
    plan = build_plan(config)
    print("Run ID:", plan["run_id"])
    print("SAE evaluations:", plan["sae_evaluations"])
    print("Metric jobs:", plan["metric_jobs"])
    print("Planned intervention measurements:", plan["planned_measurements"])
