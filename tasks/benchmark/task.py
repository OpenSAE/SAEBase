"""SAEBase benchmark task."""

from tasks.sae_runtime import SAEOperationTask


class BenchmarkTask(SAEOperationTask):
    OPERATION = "benchmark"
