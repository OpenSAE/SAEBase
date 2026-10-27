"""
Hydra training entry (module ``dlbase.train``).

From the repository root (or with that root on ``PYTHONPATH``)::

  python -m dlbase.train
  python -m dlbase.train train.max_epochs=50 seed=1

Overrides use Hydra syntax. ``paths.root`` is the parent of this package
(``${dlbase_repo_root:}``), not the process cwd. ``config_path`` points at
the repository ``conf/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dlbase.runtime_check import verify_training_runtime


import dlbase.plugin_bootstrap as _plugin_bootstrap

_plugin_bootstrap.ensure_plugins_loaded()

import hydra  # noqa: E402 - import ordering follows runtime/bootstrap contract
from omegaconf import DictConfig, OmegaConf  # noqa: E402 - import ordering follows runtime/bootstrap contract

from dlbase.cfg.config_validate import validate_hydra_cfg  # noqa: E402 - import ordering follows runtime/bootstrap contract


def _dlbase_repo_root() -> str:
    """Directory that contains the ``dlbase`` package (the repository root)."""
    return str(Path(__file__).resolve().parents[1])


OmegaConf.register_new_resolver("dlbase_repo_root", _dlbase_repo_root, replace=True)


@hydra.main(version_base=None, config_path="../conf", config_name="openresearch")
def main(cfg: DictConfig) -> Optional[float]:
    verify_training_runtime(
        context="dlbase.train",
        profile="llm",
    )
    validate_hydra_cfg(cfg)
    from dlbase.experiment_paths import prepare_run

    prepare_run(cfg, Path.cwd())
    print(OmegaConf.to_yaml(cfg))
    from tasks.runner import run_task

    run_task(cfg)
    return None


if __name__ == "__main__":
    main()
