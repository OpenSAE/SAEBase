"""Resolve established SAEBase module names to the implementations they named."""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


class AliasLoader(Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        return import_module(self.canonical)

    def exec_module(self, module):
        """The canonical module's loader has already executed its source."""


class ModuleAliases(MetaPathFinder):
    def __init__(self, aliases):
        self.aliases = aliases

    def find_spec(self, fullname, path=None, target=None):
        canonical = self.aliases.get(fullname)
        if canonical is None:
            return None
        spec = find_spec(canonical)
        if spec is None:
            return None
        return spec_from_loader(
            fullname, AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None,
        )


def register_aliases(aliases):
    """Register exact renamed module identities before searching physical paths."""
    sys.meta_path.insert(0, ModuleAliases(aliases))
