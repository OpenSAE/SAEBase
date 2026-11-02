"""Built-in model and SAE plugin registration.

Model implementations retain their historical models.* paths until a
separately validated compatibility migration. Registration runs once
through Python's import cache, even when a legacy shim is imported.
"""

from __future__ import annotations

import importlib

from saebase.registry import register_model

_HOOKED_KEYS = (
    "gpt2small",
    "pythia70m",
    "pythia410m",
    "gemma270m",
    "gemma1b",
    "gemma1b_it",
    "r1_distill_qwen15b",
    "r1_distill_llama8b",
    "gemma2_2b_base",
    "gemma2_2b_it",
    "qwen25_3b",
    "qwen25_3b_it",
    "gemma3_4b_pt",
    "gemma3_4b_it",
    "llama31_8b",
    "llama31_8b_instruct",
    "qwen3_8b",
    "qwen3_1_7b",
    "qwen3_1_7b_base",
    "mistral7b_instruct",
    "qwen35_2b",
    "qwen35_4b",
    "gemma2_9b_pt",
    "gemma2_9b_it",
    "gemma3_12b_pt",
    "gemma3_12b_it",
    "qwen35_9b",
    "gemma3_27b_pt",
    "gemma3_27b_it",
    "qwen35_27b",
)


def _register_builtin_models() -> None:
    def _register_hooked(key: str) -> None:
        compat = importlib.import_module(f"models.{key}.model_task_dataset_compatibility")
        module = importlib.import_module(f"models.{key}.{key}")
        register_model(key, compat.assert_compatible, module.construct_lazy)

    for hooked_key in _HOOKED_KEYS:
        _register_hooked(hooked_key)


_register_builtin_models()
