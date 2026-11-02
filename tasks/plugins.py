"""Register all shipped SAEBase research tasks in one Registry."""

from saebase.registry import register_task
from tasks.benchmark.legacy_sae_task import SAETask
from tasks.capture.task import CaptureTask
from tasks.precheck.task import PrecheckTask
from tasks.intervene.task import InterveneTask
from tasks.generate.task import GenerateTask
from tasks.circuits.task import CircuitsTask
from tasks.explain.task import ExplainTask
from tasks.benchmark.task import BenchmarkTask
from tasks.train.task import TrainTask

for key, cls in (
    ("precheck", PrecheckTask),
    ("capture", CaptureTask),
    ("intervene", InterveneTask),
    ("generate", GenerateTask),
    ("circuits", CircuitsTask),
    ("explain", ExplainTask),
    ("benchmark", BenchmarkTask),
    ("train", TrainTask),
):
    register_task(key, cls)
