"""
Hydra evaluation entry (lives under ``saebase/``). Prefer ``python main.py``.

Direct use::

  python -m saebase.run
  python -m saebase.run model=gemma1b sae.layer=13
"""

from __future__ import annotations

from typing import Optional

from saebase.runtime import verify_runtime

verify_runtime(context="saebase.run", mode="analysis")

import saebase.plugin_bootstrap as _plugin_bootstrap

_plugin_bootstrap.ensure_plugins_loaded()

import hydra
from omegaconf import DictConfig, OmegaConf

from saebase.cfg.config_validate import validate_hydra_cfg
from saebase.seed import apply_seed


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    validate_hydra_cfg(cfg)
    apply_seed(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))
    from saebase.runner import run_task

    result = run_task(cfg)
    print(
        f"[saebase] ok={result.ok} evaluation={result.evaluation} "
        f"model={result.model_key} sae={result.sae_key} layer={result.layer} "
        f"summary={result.summary}"
    )
    if not result.ok:
        raise RuntimeError(result.error or "evaluation failed")
    return None


if __name__ == "__main__":
    main()
