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
