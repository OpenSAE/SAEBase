"""Explicit imports for Hugging Face and TransformerLens integrations.

SAEBase's prompt modules live under ``saebase.data``.  The top-level
``datasets`` name therefore belongs unambiguously to Hugging Face; loading
the model stack never edits ``sys.path`` or the global ``sys.modules`` cache.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def import_hf_datasets() -> ModuleType:
    """Return the installed Hugging Face datasets package."""
    hf = importlib.import_module("datasets")
    if not hasattr(hf, "load_dataset"):
        raise ImportError(
            "The installed `datasets` package does not expose load_dataset. "
            "Install Hugging Face datasets in this environment."
        )
    return hf


def import_transformer_lens() -> ModuleType:
    """Load TransformerLens on demand."""
    return importlib.import_module("transformer_lens")


def import_sae_lens() -> ModuleType:
    """Load SAE Lens on demand."""
    return importlib.import_module("sae_lens")


def ensure_hf_stack_imported() -> None:
    """Legacy helper: eagerly load TransformerLens and SAE Lens when requested."""
    import_transformer_lens()
    import_sae_lens()


def load_repo_dataset_module(subpkg: str, module_name: str) -> ModuleType:
    """Legacy helper for accessing relocated built-in prompt modules.

    Both segments must be valid import identifiers; no arbitrary path loading.
    """
    if not subpkg.isidentifier() or not module_name.isidentifier():
        raise ValueError("subpkg and module_name must be Python identifiers")
    return importlib.import_module(f"saebase.data.{subpkg}.{module_name}")
