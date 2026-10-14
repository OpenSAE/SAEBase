"""Register tasks actually shipped with the standalone SAEBase package.

Experiment protocol configurations from SAEReasoning are reference data until their
implementations are separately reviewed and distributed.
"""
from saebase.registry import register_task
from saebase.sae.task import SAETask

register_task("sae", SAETask)
