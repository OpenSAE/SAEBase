"""Register language-model plugins for SAE research."""

from __future__ import annotations

from omegaconf import DictConfig

from saebase.research_registry import register_model


def _register_builtin_models() -> None:
    def assert_gemma(cfg: DictConfig) -> None:
        from models.gemma2_2b_base.model_task_dataset_compatibility import assert_compatible

        assert_compatible(cfg)

    def construct_gemma(cfg: DictConfig):
        from models.gemma2_2b_base.gemma2_2b_base import construct, construct_lazy

        return construct_lazy(cfg) if bool(cfg.model.get("lazy", False)) else construct(cfg)

    register_model("gemma2_2b_base", assert_gemma, construct_gemma)


_register_builtin_models()
