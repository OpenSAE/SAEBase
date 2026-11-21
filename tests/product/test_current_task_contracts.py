"""Purpose and main functions: Cover task contracts and task-specific configuration.

Scope: Offline configuration and task dispatch; no model weights or GPU execution.
"""

from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from saebase.results import EvaluationResult
from saebase.runtime.configuration.config_validate import validate_hydra_cfg


def config(task="capture"):
    return OmegaConf.create(
        {
            "model": {"name": "pythia70m"},
            "dataset": {"name": "IOI"},
            "task": {"name": task, "execute": False},
            "seed": 1,
            "paths": {"root": "."},
            "experiment": {"sub_name": "contract"},
        }
    )


def test_unrelated_tasks_do_not_require_training_parameters():
    for task in (
        "capture",
        "intervene",
        "generate",
        "circuits",
        "explain",
        "benchmark",
        "precheck",
    ):
        validate_hydra_cfg(config(task))


def test_training_requires_a_plan_only_for_execution():
    cfg = config("train")
    validate_hydra_cfg(cfg)
    cfg.task.execute = True
    with pytest.raises(ValueError, match="train_plan"):
        validate_hydra_cfg(cfg)


def test_configuration_result_does_not_claim_execution():
    result = EvaluationResult.from_payload("capture", config(), {"status": "configured_only"})
    assert result.status == "configured_only"
    assert result.status == "configured_only"
    assert result.summary == {"status": "configured_only"}


def test_runner_requires_the_declared_result_object(monkeypatch):
    import saebase.registry as registry
    from saebase.runner import run_task

    cfg = config()
    payload = EvaluationResult.from_payload("capture", cfg, {"status": "configured_only"})

    class Provider:
        def validate_cfg(self, cfg):
            pass

        def run(self, cfg):
            return payload

    monkeypatch.setattr(
        registry, "get_task_plugin", lambda name: SimpleNamespace(task_class=Provider)
    )
    assert run_task(cfg) is payload
    payload = {"status": "configured_only"}
    with pytest.raises(TypeError, match="EvaluationResult"):
        run_task(cfg)
