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
