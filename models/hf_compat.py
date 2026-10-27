"""Explicit model backends and safe access to external Hugging Face datasets.

OpenResearch's own top-level datasets/ component tree shadows the third-party
package name. The isolated reader resolves that collision on real data reads.
No model weights, data or provider APIs load during plugin registration.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from models.hf_datasets_bridge import IsolatedHFDatasets


def import_hf_datasets() -> ModuleType | IsolatedHFDatasets:
    """Read Hugging Face datasets while retaining the local datasets/ package.

    When 'datasets' is the local OpenResearch package, use the isolated
    Hugging Face loader. The original package remains untouched.
    """
    candidate = importlib.import_module("datasets")
    if hasattr(candidate, "load_dataset") and hasattr(candidate, "load_from_disk"):
        return candidate
    return IsolatedHFDatasets()


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
    return importlib.import_module(f"datasets.NLP.{subpkg}.{module_name}")
