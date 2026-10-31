"""Temporary registered task that delegates real execution to CaptureTask."""
from __future__ import annotations
from saebase.registry import TaskModule

KEY = 'wp21_capture_legacy'


class LegacyCaptureTask(TaskModule):
    @property
    def name(self):
        return KEY

    def run(self, cfg):
        from omegaconf import OmegaConf
        from tasks.capture.task import CaptureTask
        if str(cfg.task.name) != KEY:
            raise ValueError('Legacy task must be selected by its registered key')
        delegated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        delegated.task.name = 'capture'
        delegated.task.path = 'capture'
        result = CaptureTask().run(delegated)
        return {**result, 'registered_task': KEY, 'delegated_task': 'capture'}


def register():
    from saebase.plugin_bootstrap import ensure_plugins_loaded
    from saebase.registry import register_task
    ensure_plugins_loaded()
    register_task(KEY, LegacyCaptureTask)
