"""Register seven SAEBase LLM tasks."""

from __future__ import annotations
from dlbase.registry import register_task


def _register_builtin_tasks() -> None:
    from tasks.LLM.capture.task import CaptureTask
    from tasks.LLM.intervene.task import InterveneTask
    from tasks.LLM.generate.task import GenerateTask
    from tasks.LLM.circuits.task import CircuitsTask
    from tasks.LLM.explain.task import ExplainTask
    from tasks.LLM.benchmark.task import BenchmarkTask
    from tasks.LLM.train.task import TrainTask

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
