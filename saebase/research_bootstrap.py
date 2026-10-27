"""SAE-specific research registry bootstrap."""
_loaded = False

def ensure_plugins_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        import research_datasets.registry_plugins
        import models.research_plugins
        import tasks.research_plugins
    except Exception:
        _loaded = False
        raise
