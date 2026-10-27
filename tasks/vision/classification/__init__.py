"""Vision classification task with lazy PyTorch Lightning imports.

Only runtime execution loads torch/torchvision/Lightning. Registration and
Hydra configuration validation remain available on CPU-only workers.
"""

from .classification_task import ClassificationTask

def __getattr__(name: str):
    if name == "run_classification_training":
        from .run import run_classification_training
        return run_classification_training
    raise AttributeError(name)

__all__ = ["ClassificationTask", "run_classification_training"]
