"""Register the 15 active Model instances; historical Model adapters remain opt-in.

A Model owns its default SAE choices in conf/model/*.yaml. SAE profiles
are resources, not another core plugin registration.
"""

from __future__ import annotations

import importlib

from saebase.registry import ModelPlugin, register_model

# Active registration list, not a historical resource deletion list.
_HOOKED_KEYS = (
    "gemma270m",
    "gemma1b",
    "gemma2_2b_base",
    "gemma2_9b_it",
    "qwen3_1_7b_base",
    "qwen35_2b",
    "qwen35_4b",
    "qwen35_9b",
    "gpt2small",
    "pythia70m",
    "pythia410m",
    "llama31_8b",
    "llama31_8b_instruct",
    "mistral7b_instruct",
    "r1_distill_llama8b",
)

_LEGACY_KEYS = frozenset({
    "gemma1b_it",
    "r1_distill_qwen15b",
    "gemma2_2b_it",
    "qwen25_3b",
    "qwen25_3b_it",
    "gemma3_4b_pt",
    "gemma3_4b_it",
    "qwen3_8b",
    "qwen3_1_7b",
    "gemma2_9b_pt",
    "gemma3_12b_pt",
    "gemma3_12b_it",
    "gemma3_27b_pt",
    "gemma3_27b_it",
    "qwen35_27b",
})

_ACTIVE_MODULES = {
    "gemma270m": ("models.gemma.gemma3.gemma270m.compatibility", "models.gemma.gemma3.gemma270m.model"),
    "gemma1b": ("models.gemma.gemma3.gemma1b.compatibility", "models.gemma.gemma3.gemma1b.model"),
    "gemma2_2b_base": ("models.gemma.gemma2.gemma2_2b_base.compatibility", "models.gemma.gemma2.gemma2_2b_base.model"),
    "gemma2_9b_it": ("models.gemma.gemma2.gemma2_9b_it.compatibility", "models.gemma.gemma2.gemma2_9b_it.model"),
    "qwen3_1_7b_base": ("models.qwen.qwen3.qwen3_1_7b_base.compatibility", "models.qwen.qwen3.qwen3_1_7b_base.model"),
    "qwen35_2b": ("models.qwen.qwen35.qwen35_2b.compatibility", "models.qwen.qwen35.qwen35_2b.model"),
    "qwen35_4b": ("models.qwen.qwen35.qwen35_4b.compatibility", "models.qwen.qwen35.qwen35_4b.model"),
    "qwen35_9b": ("models.qwen.qwen35.qwen35_9b.compatibility", "models.qwen.qwen35.qwen35_9b.model"),
    "gpt2small": ("models.gpt2.gpt2.gpt2small.compatibility", "models.gpt2.gpt2.gpt2small.model"),
    "pythia70m": ("models.pythia.pythia.pythia70m.compatibility", "models.pythia.pythia.pythia70m.model"),
    "pythia410m": ("models.pythia.pythia.pythia410m.compatibility", "models.pythia.pythia.pythia410m.model"),
    "llama31_8b": ("models.llama.llama31.llama31_8b.compatibility", "models.llama.llama31.llama31_8b.model"),
    "llama31_8b_instruct": ("models.llama.llama31.llama31_8b_instruct.compatibility", "models.llama.llama31.llama31_8b_instruct.model"),
    "mistral7b_instruct": ("models.mistral.mistral7b.mistral7b_instruct.compatibility", "models.mistral.mistral7b.mistral7b_instruct.model"),
    "r1_distill_llama8b": ("models.llama.r1_distill.r1_distill_llama8b.compatibility", "models.llama.r1_distill.r1_distill_llama8b.model"),
}


def _load_plugin(compat_path: str, model_path: str) -> ModelPlugin:
    compat = importlib.import_module(compat_path)
    module = importlib.import_module(model_path)
    return ModelPlugin(assert_compatible=compat.assert_compatible, construct=module.construct_lazy)


def legacy_plugin(key: str) -> ModelPlugin:
    """Resolve a historical Model explicitly without registering 30 defaults."""
    if key not in _LEGACY_KEYS:
        raise ValueError(f"Unknown legacy Model {key!r}")
    return _load_plugin(
        f"models.{key}.model_task_dataset_compatibility", f"models.{key}.{key}"
    )


def _register_builtin_models() -> None:
    for key in _HOOKED_KEYS:
        compat, module = _ACTIVE_MODULES[key]
        plugin = _load_plugin(compat, module)
        register_model(key, plugin.assert_compatible, plugin.construct)


_register_builtin_models()
