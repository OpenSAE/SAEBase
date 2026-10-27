"""
Hydra training entry (module ``saebase.runtime.train``).

From the repository root (or with that root on ``PYTHONPATH``)::

  python -m saebase.runtime.train
  python -m saebase.runtime.train train.max_epochs=50 seed=1

Overrides use Hydra syntax. ``paths.root`` is the parent of this package
(``${saebase_repo_root:}``), not the process cwd. ``config_path`` points at
the repository ``conf/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from saebase.runtime.runtime_check import verify_training_runtime


import saebase.plugin_bootstrap as _plugin_bootstrap

_plugin_bootstrap.ensure_plugins_loaded()

import hydra  # noqa: E402 - import ordering follows runtime/bootstrap contract
from omegaconf import DictConfig, OmegaConf  # noqa: E402 - import ordering follows runtime/bootstrap contract

from saebase.runtime.cfg.config_validate import validate_hydra_cfg  # noqa: E402 - import ordering follows runtime/bootstrap contract


def _saebase_repo_root() -> str:
    """Directory containing the SAEBase source workspace."""
    return str(Path(__file__).resolve().parents[2])


OmegaConf.register_new_resolver("saebase_repo_root", _saebase_repo_root, replace=True)
# Legacy Hydra configuration snapshots may still use this resolver name.
OmegaConf.register_new_resolver("dlbase_repo_root", _saebase_repo_root, replace=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="sae_research")
def main(cfg: DictConfig) -> Optional[float]:
    verify_training_runtime(
        context="saebase.runtime.train",
        profile="llm",
    )
    validate_hydra_cfg(cfg)
    from saebase.runtime.experiment_paths import prepare_run

    prepare_run(cfg, Path.cwd())
    print(OmegaConf.to_yaml(cfg))
    from saebase.runner import run_task

    run_task(cfg)
    return None


if __name__ == "__main__":
    main()
