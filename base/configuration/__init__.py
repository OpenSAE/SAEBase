"""Named component configurations for source and installed SAEBase.

Catalogs keep shared defaults once. Hydra owns selection and composition;
load_group provides the same presets to direct component consumers.
"""

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def workspace_at(directory: Path) -> bool:
    """Recognize a source workspace independently of the launch directory."""
    return (
        (directory / "pyproject.toml").is_file()
        and (directory / "base/application/commands.py").is_file()
        and (directory / "conf/config.yaml").is_file()
    )


def configuration_root() -> Path:
    """Locate physical recipes in a checkout and data files in distributions."""
    source = Path(__file__).resolve().parents[2]
    if workspace_at(source):
        return source / "conf"
    package = Path(__file__).absolute().parent
    image = package.parent
    if image.name.startswith("__editable__.") and image.parent.name == ".build":
        source = image.parent.parent
        if workspace_at(source):
            return source / "conf"
    return package


def launch_directory() -> Path:
    """Return the caller's directory before Hydra changes into the run directory."""
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    return Path(get_original_cwd() if HydraConfig.initialized() else Path.cwd()).resolve()


def default_workspace() -> str | None:
    """Prefer the caller's SAEBase checkout, then the checkout supplying the package."""
    launch = launch_directory()
    for directory in (launch, *launch.parents, configuration_root().parent):
        if workspace_at(directory):
            return str(directory)
    return None


def _default_data_directory(root: str | None) -> str | None:
    return (
        str(
            (
                (launch_directory() / root).resolve()
                / "assets/dataset_preparation/prompt_sets/files"
            ).resolve()
        )
        if root is not None
        else None
    )


def _prepared_dataset_dir(root: str, name: str) -> str:
    from saebase.datasets.artifacts import prepared_dataset_root

    return str(prepared_dataset_root(root, name))


OmegaConf.register_new_resolver("saebase_prepared_dir", _prepared_dataset_dir, replace=True)
OmegaConf.register_new_resolver("saebase_workspace", default_workspace, replace=True)
OmegaConf.register_new_resolver("saebase_data_dir", _default_data_directory, replace=True)


CATALOG_PATHS = {
    "model": "models/catalog.yaml",
    "sae": "base/interfaces/sae_profiles.yaml",
    "dataset": "datasets/catalog.yaml",
    "prompt_source": "datasets/catalog.yaml",
    "benchmark": "tasks/precheck/task.yaml",
}


def configuration_location(source: str | Path) -> tuple[Path, str]:
    """Split a physical configuration path and an optional named section."""
    path = Path(source)
    filename, separator, section = str(path).rpartition("#")
    if (
        separator
        and Path(filename).suffix.lower() in {".json", ".yml", ".yaml"}
        and not path.is_file()
    ):
        if not section or any(not part.isidentifier() for part in section.split(".")):
            raise ValueError(f"Invalid configuration section: {source}")
        return Path(filename), section
    return path, ""


def read_configuration(source: str | Path, *, resolve: bool = False):
    """Read a standalone JSON/YAML recipe or a named mapping within a shared file."""
    path, section = configuration_location(source)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = (
            OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            if resolve
            else yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    else:
        raise ValueError(f"Configuration must use JSON or YAML: {path}")
    if section:
        for key in section.split("."):
            if not isinstance(data, dict) or key not in data:
                raise ValueError(f"Configuration section {section!r} is missing in {path}")
            data = data[key]
    return data


def bundled_configuration(relative: str, section: str = "") -> Path:
    """Locate a bundled file in a checkout or an installed package."""
    path = configuration_root() / relative
    return Path(f"{path}#{section}") if section else Path(str(path))


def _catalog(group):
    config = OmegaConf.load(configuration_root() / CATALOG_PATHS[group])
    sections = {"dataset": "dataset", "prompt_source": "prompt_source", "benchmark": "sae_checks"}
    return config[sections[group]] if group in sections else config


def load_task(name):
    """Read a task recipe from the directory matching its implementation."""
    config = OmegaConf.load(bundled_configuration(f"tasks/{name}/task.yaml"))
    return OmegaConf.masked_copy(config, ["task"])


def group_names(group):
    return tuple(_catalog(group).presets)


def load_group(group, name):
    catalog = _catalog(group)
    if name not in catalog.presets:
        raise KeyError(f"Unknown {group} preset {name!r}; available: {list(catalog.presets)}")
    return OmegaConf.create({group: OmegaConf.merge(catalog.defaults, catalog.presets[name])})


def register_configs():
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    for group in ("model", "sae", "dataset", "prompt_source"):
        catalog = _catalog(group)
        for name, values in catalog.presets.items():
            node = {group: OmegaConf.merge(catalog.defaults, values)}
            store.store(group=group, name=name, node=node, package="_global_", provider="saebase")

    for directory in (configuration_root() / "tasks").iterdir():
        if (directory / "task.yaml").is_file():
            store.store(
                group="task",
                name=directory.name,
                node=load_task(directory.name),
                package="_global_",
                provider="saebase",
            )

    for group, name, path in (
        ("paths", "default", "base/runtime/runtime.yaml"),
        ("experiment", "default", "base/runtime/runtime.yaml"),
        ("evaluation", "activation_health", "tasks/benchmark/task.yaml"),
    ):
        store.store(
            group=group,
            name=name,
            node={group: OmegaConf.load(configuration_root() / path)[group]},
            package="_global_",
            provider="saebase",
        )


register_configs()
