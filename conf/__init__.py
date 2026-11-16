"""Named component configurations for source and installed SAEBase.

Catalogs keep shared defaults once. Hydra owns selection and composition;
load_group provides the same presets to direct component consumers.
"""

from importlib.resources import files
from omegaconf import OmegaConf


def group_names(group):
    return tuple(OmegaConf.load(files(__package__) / group / "catalog.yaml").presets)


def load_group(group, name):
    catalog = OmegaConf.load(files(__package__) / group / "catalog.yaml")
    if name not in catalog.presets:
        raise KeyError(f"Unknown {group} preset {name!r}; available: {list(catalog.presets)}")
    return OmegaConf.create({group: OmegaConf.merge(catalog.defaults, catalog.presets[name])})


def register_configs():
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    for group in ("model", "sae", "dataset", "prompt_source"):
        catalog = OmegaConf.load(files(__package__) / group / "catalog.yaml")
        for name, values in catalog.presets.items():
            node = {group: OmegaConf.merge(catalog.defaults, values)}
            store.store(group=group, name=name, node=node, package="_global_", provider="saebase")


register_configs()
