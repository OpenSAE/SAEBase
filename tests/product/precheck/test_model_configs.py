"""Every built-in Model owns SAE choices; no anonymous global fallback."""

from __future__ import annotations


from saebase.api.sae_profiles import get_profile


def test_model_conf_sae_lists_are_resolvable():
    from conf import group_names, load_group

    for entry in group_names("model"):
        model = load_group("model", entry).model
        available = list(model.get("sae_profiles", []))
        defaults = list(model.get("default_saes", []))
        assert len(set(available)) == len(available), entry
        assert len(set(defaults)) == len(defaults), entry
        assert set(defaults).issubset(available), entry
        for key in available:
            assert str(model.name) in get_profile(key).allowed_model_keys(), (entry, key)


def test_builtin_sae_profiles_are_parameters_not_registry_entries():
    import saebase.registry as registry
    from saebase.plugin_bootstrap import ensure_plugins_loaded

    ensure_plugins_loaded()
    assert "gemma2_2b_pt_l25_canonical" not in registry._SAE_REGISTRY
    assert "gemma2_2b_pt_l25_canonical" in registry.list_registered_sae_keys()
    assert registry.get_sae_plugin("gemma2_2b_base").profile_key == "gemma2_2b_pt_l25_canonical"


def test_all_implemented_models_use_family_owned_registration():
    from models.plugins import MODEL_KEYS, MODEL_MODULES
    from saebase.registry import list_registered_model_keys, get_model_plugin

    assert set(list_registered_model_keys()) == set(MODEL_KEYS) == set(MODEL_MODULES)
    assert all(get_model_plugin(key) is not None for key in MODEL_KEYS)
    assert all(path.endswith(".specs") for path in MODEL_MODULES.values())


def test_model_specs_are_loaded_from_family_declarations():
    from models.plugins import MODEL_KEYS, get_model_adapter

    for key in MODEL_KEYS:
        adapter = get_model_adapter(key)
        assert adapter.spec.key == key
        assert callable(adapter.construct_lazy)
