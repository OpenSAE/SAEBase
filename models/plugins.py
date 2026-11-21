"""Register model adapters directly from their family-owned declarations.

The registry owns public model keys; series files own weights and loading choices.
"""

from __future__ import annotations
from importlib import import_module
from saebase.registry import register_model

from models.model_adapter import ModelAdapter

SERIES_MODULES = (
    "models.gemma.gemma3.specs",
    "models.gemma.gemma2.specs",
    "models.gpt_family.gpt2_series.specs",
    "models.llama.llama31.specs",
    "models.mistral.mistral_series_7b.specs",
    "models.pythia.pythia.specs",
    "models.qwen_family.qwen_series_25.specs",
    "models.qwen_family.qwen_series_35.specs",
    "models.qwen_family.qwen_series_3.specs",
    "models.llama.r1_distill.specs",
    "models.qwen_family.r1_distill.specs",
)
MODEL_MODULES = {}
_ADAPTERS = {}
for module_name in SERIES_MODULES:
    for value in vars(import_module(module_name)).values():
        if isinstance(value, ModelAdapter):
            key = value.spec.key
            if key in _ADAPTERS:
                raise ValueError(f"Duplicate model declaration: {key}")
            MODEL_MODULES[key] = module_name
            _ADAPTERS[key] = value
MODEL_KEYS = tuple(_ADAPTERS)


def get_model_adapter(key):
    """Return the declared adapter without loading its model weights."""
    return _ADAPTERS[key]


def _register_builtin_models():
    from saebase.datasets.artifacts import assert_available

    def compatibility(key):
        def check(cfg):
            if str(cfg.task.name) in {"capture", "intervene", "generate"}:
                assert_available(str(cfg.dataset.name), key)

        return check

    for key in MODEL_KEYS:
        adapter = get_model_adapter(key)
        register_model(key, compatibility(key), adapter.construct_lazy, load=adapter.construct)


_register_builtin_models()
