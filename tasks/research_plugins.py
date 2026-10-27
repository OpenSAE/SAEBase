"""Register seven SAEBase LLM tasks."""

from __future__ import annotations
from saebase.research_registry import register_task


def _register_builtin_tasks() -> None:
    from tasks.capture.task import CaptureTask
    from tasks.intervene.task import InterveneTask
    from tasks.generate.task import GenerateTask
    from tasks.circuits.task import CircuitsTask
    from tasks.explain.task import ExplainTask
    from tasks.benchmark.task import BenchmarkTask
    from tasks.train.task import TrainTask

    for key, cls in (
        ("capture", CaptureTask),
        ("intervene", InterveneTask),
        ("generate", GenerateTask),
        ("circuits", CircuitsTask),
        ("explain", ExplainTask),
        ("benchmark", BenchmarkTask),
        ("train", TrainTask),
    ):
        register_task(key, cls)


_register_builtin_tasks()
