"""One-shot import of plugin registration modules."""

from __future__ import annotations

_loaded = False


def ensure_plugins_loaded() -> None:
    global _loaded
    if _loaded:
        return
    import saebase.data.plugins  # noqa: F401
    import saebase.model_plugins  # noqa: F401
    import saebase.tasks.plugins  # noqa: F401

    _loaded = True
