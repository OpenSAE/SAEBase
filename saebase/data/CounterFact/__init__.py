"""Legacy import forwarding to the canonical Dataset implementation."""
from importlib import import_module as _import_module
_impl = _import_module("saebase.datasets.CounterFact")
def __getattr__(name):
    return getattr(_impl, name)
def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)
