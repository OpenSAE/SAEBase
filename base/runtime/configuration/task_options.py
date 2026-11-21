"""Read task-owned configuration and translate historical overrides at merge boundaries."""

from omegaconf import OmegaConf


def task_value(task, key, default=None):
    """Read the selected task's own branch, then common or historical fields."""
    owner = task.get("name")
    owned = task.get(owner, {}) if isinstance(owner, str) else {}
    return owned[key] if key in owned else task.get(key, default)


def merge_task_overrides(task, overrides):
    """Merge flat historical overrides into the owning branch of a current recipe."""
    owner = task.name
    owned = task.get(owner, {})
    legacy = {key: value for key, value in overrides.items() if key in owned}
    current = {key: value for key, value in overrides.items() if key not in owned and key != owner}
    if legacy or owner in overrides:
        current[owner] = OmegaConf.merge(legacy, overrides.get(owner, {}))
    return OmegaConf.merge(task, current)


def task_owner(task):
    """Return the writable task branch; old in-memory tasks stay writable."""
    owner = task.get("name")
    return task[owner] if isinstance(owner, str) and owner in task else task
