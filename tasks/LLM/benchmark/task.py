"""SAEBase benchmark task."""
from tasks.LLM.sae_runtime import SAEOperationTask


class BenchmarkTask(SAEOperationTask):
    OPERATION = "benchmark"
