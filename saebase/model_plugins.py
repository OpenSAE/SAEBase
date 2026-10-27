"""Compatibility re-export of the unified model plugin module."""
from models.plugins import *
from models import plugins as _impl
def __getattr__(name):
    return getattr(_impl, name)
