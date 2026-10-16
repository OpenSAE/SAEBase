"""Register tasks actually shipped with SAEBase.

Historical task configuration files are not standalone implementations.
"""

from saebase.registry import register_task
from saebase.sae.task import SAETask

register_task("sae", SAETask)
