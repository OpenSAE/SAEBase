"""No downloads: inspect SAEBase M3 deterministic benchmark plan."""
from pathlib import Path
from saebase.benchmarks import load_config,build_plan
if __name__=="__main__":
    config=load_config(Path(__file__).resolve().parents[1]/"conf/benchmark/p0_small.json")
    plan=build_plan(config)
    print("Run ID:",plan["run_id"])
    print("SAE evaluations:",plan["sae_evaluations"])
    print("Metric jobs:",plan["metric_jobs"])
    print("Planned M2 measurements:",plan["planned_measurements"])
