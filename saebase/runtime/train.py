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
from saebase.runtime.experiment_paths import hydra_run_component  # noqa: E402

OmegaConf.register_new_resolver("saebase_run_component", hydra_run_component, replace=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    verify_training_runtime(
        context="saebase.runtime.train",
        profile="llm",
    )
    outcome = execute_run(cfg, Path.cwd())
    print("Task:", outcome.status, "completion:", outcome.completion)
    if outcome.exit_code:
        import sys

        for issue in outcome.diagnostics:
            print(f"{issue.get('stage', outcome.task)}: {issue['reason']}", file=sys.stderr)
            for step in issue.get("next_steps", []):
                print(f"Next: {step}", file=sys.stderr)
        raise SystemExit(outcome.exit_code)
    return None


def execute_run(cfg: DictConfig, folder: Path):
    """One shared lifecycle for CLI and integration tests, with records before failure exits."""
    import traceback
    from saebase.runner import run_task
    from saebase.seed import apply_seed
    from saebase.runtime.experiment_paths import prepare_run, run_session
    from saebase.runtime.experiment_results import record_run

    task = validate_hydra_cfg(cfg)
    from models.loading_policy import resolve_config_devices

    resolve_config_devices(cfg)
    with run_session(cfg, folder):
        prepare_run(cfg, folder)
        if cfg.get("runtime", {}).get("mode") == "resume":
            from saebase.runtime.experiment_results import inspect_run, read_run
            from saebase.results import EvaluationResult

            if inspect_run(folder, expected_cfg=cfg)["action"] == "skip":
                return EvaluationResult(**read_run(folder)["result"])
        apply_seed(int(cfg.seed))
        try:
            outcome = run_task(cfg, task=task)
        except Exception as exc:
            record_run(
                folder,
                cfg,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        record_run(folder, cfg, result=outcome)
        return outcome


if __name__ == "__main__":
    main()
