"""SAEBase: Hydra-driven framework for LLM SAE evaluation."""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.7.0"

# Register named Hydra presets without loading model or Task implementations.
import conf as _conf  # noqa: F401 - register named configuration presets
