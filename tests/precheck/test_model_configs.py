"""Every built-in Model owns SAE choices; no anonymous global fallback."""

from __future__ import annotations

from importlib.resources import files

from omegaconf import OmegaConf

from saebase.sae.registry import get_profile


def test_model_conf_sae_lists_are_resolvable():
    root = files("conf") / "model"
    for entry in root.iterdir():
        if entry.suffix != ".yaml":
            continue
        model = OmegaConf.load(entry).model
        available = list(model.get("sae_profiles", []))
        defaults = list(model.get("default_saes", []))
        assert len(set(available)) == len(available), entry.name
        assert len(set(defaults)) == len(defaults), entry.name
        assert set(defaults).issubset(available), entry.name
        for key in available:
            assert str(model.name) in get_profile(key).allowed_model_keys(), (entry.name, key)


def test_builtin_sae_profiles_are_parameters_not_registry_entries():
    import saebase.registry as registry
    from saebase.plugin_bootstrap import ensure_plugins_loaded

    ensure_plugins_loaded()
    assert "gemma2_2b_pt_l25_canonical" not in registry._SAE_REGISTRY
    assert "gemma2_2b_pt_l25_canonical" in registry.list_registered_sae_keys()
    assert registry.get_sae_plugin("gemma2_2b_base").profile_key == "gemma2_2b_pt_l25_canonical"

def test_active_model_registry_is_small_and_family_bounded():
    from models.plugins import _HOOKED_KEYS, _LEGACY_KEYS, _ACTIVE_MODULES
    from saebase.registry import list_registered_model_keys, get_model_plugin

    keys = list_registered_model_keys()
    assert len(keys) == 15
    assert set(keys) == set(_HOOKED_KEYS) == set(_ACTIVE_MODULES)
    assert sum(key.startswith("gemma") for key in keys) == 4
    assert sum(key.startswith("qwen") for key in keys) == 4
    assert all(get_model_plugin(key) is not None for key in _LEGACY_KEYS)


def test_migrated_public_model_paths_resolve_same_spec():
    from importlib import import_module
    from models.plugins import _ACTIVE_MODULES

    for key, (_, module) in _ACTIVE_MODULES.items():
        new = import_module(module)
        old = import_module(f"models.{key}.{key}")
        assert old.SPEC is new.SPEC
        assert old.construct_lazy is new.construct_lazy
