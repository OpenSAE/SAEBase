"""Core Dataset/Model/Task registration with compatibility SAE parameter lookup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type

from omegaconf import DictConfig

from saebase.results import EvaluationResult


class TaskModule(ABC):
    """Hydra task contract: config in, structured evaluation result out."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, cfg: DictConfig) -> EvaluationResult: ...


@dataclass(frozen=True)
class PromptSourcePlugin:
    """``prompt_source.name`` → availability gate + artifact helpers."""

    assert_compatible: Callable[[DictConfig], None]
    canonicalize: Callable[[str], str]
    notes: Callable[[], str]


@dataclass(frozen=True)
class DatasetPlugin:
    setup: Callable[[Any, DictConfig, Any], None]
    assert_task_supported: Callable[[DictConfig], None]
    eval_split_note: Callable[[Any], str]


@dataclass(frozen=True)
class ModelPlugin:
    """``model.name`` → compatibility gate + lazy construction metadata."""

    assert_compatible: Callable[[DictConfig], None]
    construct: Callable[[DictConfig], Any]


@dataclass(frozen=True)
class SAEPlugin:
    """Legacy SAE configuration lookup, not a core registered component."""

    assert_compatible: Callable[[DictConfig], None]
    profile_key: str


@dataclass(frozen=True)
class TaskPlugin:
    task_class: Type[TaskModule]


_DATASET_REGISTRY: dict[str, DatasetPlugin] = {}
_DATASET_ALIASES: dict[str, str] = {}
_PROMPT_SOURCE_REGISTRY: dict[str, PromptSourcePlugin] = {}
_PROMPT_SOURCE_ALIASES: dict[str, str] = {}
_MODEL_REGISTRY: dict[str, ModelPlugin] = {}
_SAE_REGISTRY: dict[str, SAEPlugin] = {}
_TASK_REGISTRY: dict[str, TaskPlugin] = {}
_PLUGINS_BOOTSTRAPPED = False


def _ensure_plugins() -> None:
    global _PLUGINS_BOOTSTRAPPED
    if _PLUGINS_BOOTSTRAPPED:
        return
    import saebase.plugin_bootstrap as plugin_bootstrap  # noqa: PLC0415

    plugin_bootstrap.ensure_plugins_loaded()
    _PLUGINS_BOOTSTRAPPED = True


def register_prompt_source(
    name: str,
    assert_compatible: Callable[[DictConfig], None],
    canonicalize: Callable[[str], str],
    notes: Callable[[], str],
) -> None:
    if name in _PROMPT_SOURCE_REGISTRY:
        raise ValueError(f"register_prompt_source: {name!r} is already registered")
    _PROMPT_SOURCE_REGISTRY[name] = PromptSourcePlugin(
        assert_compatible=assert_compatible,
        canonicalize=canonicalize,
        notes=notes,
    )


def register_prompt_source_alias(alias: str, canonical: str) -> None:
    if alias in _PROMPT_SOURCE_ALIASES:
        raise ValueError(
            f"register_prompt_source_alias: {alias!r} already mapped to "
            f"{_PROMPT_SOURCE_ALIASES[alias]!r}"
        )
    _PROMPT_SOURCE_ALIASES[alias] = canonical


def resolve_prompt_source_name(name: str) -> str:
    return _PROMPT_SOURCE_ALIASES.get(name, name)


def register_model(
    name: str,
    assert_compatible: Callable[[DictConfig], None],
    construct: Callable[[DictConfig], Any],
) -> None:
    if name in _MODEL_REGISTRY:
        raise ValueError(f"register_model: {name!r} is already registered")
    _MODEL_REGISTRY[name] = ModelPlugin(assert_compatible=assert_compatible, construct=construct)


def register_sae(
    name: str,
    assert_compatible: Callable[[DictConfig], None],
    *,
    profile_key: str | None = None,
) -> None:
    if name in _SAE_REGISTRY:
        raise ValueError(f"register_sae: {name!r} is already registered")
    _SAE_REGISTRY[name] = SAEPlugin(
        assert_compatible=assert_compatible,
        profile_key=profile_key or name,
    )


def register_task(name: str, task_class: Type[TaskModule]) -> None:
    if name in _TASK_REGISTRY:
        raise ValueError(f"register_task: {name!r} is already registered")
    _TASK_REGISTRY[name] = TaskPlugin(task_class=task_class)


def get_prompt_source_plugin(name: str) -> PromptSourcePlugin:
    _ensure_plugins()
    key = resolve_prompt_source_name(str(name))
    if key not in _PROMPT_SOURCE_REGISTRY:
        raise ValueError(f"Unknown prompt_source.name: {name!r}")
    return _PROMPT_SOURCE_REGISTRY[key]


def get_model_plugin(name: str) -> ModelPlugin:
    _ensure_plugins()
    key = str(name)
    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key]
    # Historical model adapters remain explicitly loadable for reproduction,
    # without contributing to the small active Model registration list.
    from models.plugins import _LEGACY_KEYS, legacy_plugin

    if key in _LEGACY_KEYS:
        return legacy_plugin(key)
    raise ValueError(f"Unknown model.name: {key!r}")


def get_sae_plugin(name: str) -> SAEPlugin:
    _ensure_plugins()
    key = str(name)
    if key in _SAE_REGISTRY:
        return _SAE_REGISTRY[key]  # Deprecated external compatibility extension.
    if key == "none":
        return SAEPlugin(assert_compatible=lambda cfg: None, profile_key="none")
    # SAE is a Model-owned resource parameter, not a fourth core Registry.
    from saebase.sae.registry import get_profile

    try:
        profile = get_profile(key)
    except KeyError as exc:
        raise ValueError(f"Unknown sae.name: {key!r}") from exc

    def check(cfg: DictConfig) -> None:
        model = str(cfg.model.name)
        if model not in profile.allowed_model_keys():
            raise ValueError(
                f"SAE {key!r} is incompatible with model {model!r}"
            )
        value = cfg.get("sae", {}).get("layer")
        if value is not None and int(value) not in profile.layers:
            raise ValueError(f"SAE {key!r} does not include layer {value}")

    return SAEPlugin(assert_compatible=check, profile_key=profile.key)


def get_task_plugin(name: str) -> TaskPlugin:
    _ensure_plugins()
    key = str(name)
    if key not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task.name={key!r}. Add tasks/{key}/ and register in tasks.plugins."
        )
    return _TASK_REGISTRY[key]


def list_registered_prompt_source_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_PROMPT_SOURCE_REGISTRY.keys())


def list_registered_model_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_MODEL_REGISTRY.keys())


def list_registered_sae_keys() -> list[str]:
    _ensure_plugins()
    # Legacy public listing resolves Model-associated SAE profile parameters.
    from saebase.sae.registry import DEFAULT_SAE_FOR_MODEL, SAE_KEYS

    return sorted(set(_SAE_REGISTRY) | set(SAE_KEYS) | set(DEFAULT_SAE_FOR_MODEL) | {"none"})


def list_registered_task_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_TASK_REGISTRY.keys())


def register_dataset(name: str, setup, assert_task_supported, eval_split_note) -> None:
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset already registered: {name}")
    _DATASET_REGISTRY[name] = DatasetPlugin(setup, assert_task_supported, eval_split_note)


def register_dataset_alias(alias: str, canonical: str) -> None:
    if alias in _DATASET_ALIASES:
        raise ValueError(f"Dataset alias already registered: {alias}")
    _DATASET_ALIASES[alias] = canonical


def resolve_dataset_name(name: str) -> str:
    return _DATASET_ALIASES.get(str(name), str(name))


def get_dataset_plugin(name: str) -> DatasetPlugin:
    _ensure_plugins()
    key = resolve_dataset_name(name)
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset.name: {name}")
    return _DATASET_REGISTRY[key]


def list_registered_dataset_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_DATASET_REGISTRY)
