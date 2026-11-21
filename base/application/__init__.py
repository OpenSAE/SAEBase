"""SAEBase: Hydra-driven framework for LLM SAE evaluation."""

from __future__ import annotations

# Register named Hydra presets without loading model or Task implementations.
import conf as _conf  # noqa: F401 - register named configuration presets

__all__ = ["__version__"]
__version__ = "0.8.0"

from .import_aliases import register_aliases as _register_aliases

_register_aliases({
    "saebase.cli": "saebase.commands",
    "saebase.api.neuronpedia": "saebase.api.neuronpedia_client",
    "saebase.api.scope2_clt": "saebase.api.scope_cross_layer",
    "saebase.api.scope2_native": "saebase.api.scope_native",
    "saebase.datasets.datamodule": "saebase.datasets.data_module",
    "saebase.research_lab.backend.api": "saebase.research_lab.backend.service",
    "saebase.runtime.cfg": "saebase.runtime.configuration",
    "saebase.runtime.cfg.config_validate": "saebase.runtime.configuration.config_validate",
    "saebase.runtime.cfg.contracts": "saebase.runtime.configuration.contracts",
    "saebase.runtime.cfg.task_options": "saebase.runtime.configuration.task_options",
})
