"""Expose physical dataset adapters under the SAEBase Python namespace.

Source checkouts load adapters from the project datasets directory. Built
packages bundle the same adapters without shadowing Hugging Face datasets.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path
import sys


_SOURCE_MODULES = {
    "ARCChallenge": "arc_challenge",
    "BBH": "big_bench_hard",
    "CounterFact": "counter_fact",
    "GSM8K": "grade_school_math",
    "GreaterThan": "greater_than",
    "HarmBench": "harm_bench",
    "HotpotQA": "hotpot_qa",
    "HumanEval": "human_eval",
    "IOI": "indirect_object_identification",
    "MMLUPro": "mmlu_professional",
    "RAVEL": "ravel",
    "SimpleCopying": "simple_copying",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "TruthfulQA": "truthful_qa",
    "XSTest": "xs_test",
}


def dataset_source_module(identity: str) -> str:
    """Return the adapter module for a registered dataset identity."""
    return f"{__name__}.{_SOURCE_MODULES[identity]}"


class _DatasetAliasLoader(Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        return import_module(self.target)

    def exec_module(self, module):
        pass


class _DatasetImportAliases(MetaPathFinder):
    """Resolve legacy adapter imports to the same module objects."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = f"{__name__}."
        if not fullname.startswith(prefix):
            return None
        identity, _, member = fullname[len(prefix):].partition(".")
        if identity not in _SOURCE_MODULES:
            return None
        package = dataset_source_module(identity)
        if member == identity:
            member = "benchmark" if identity == "BBH" else "dataset"
        elif member == "tokenize":
            member = "tokenization"
        canonical = f"{package}.{member}" if member else package
        return spec_from_loader(fullname, _DatasetAliasLoader(canonical), is_package=not member)


_root = Path(__file__).resolve().parents[2] / "datasets"
if _root.is_dir() and str(_root) not in __path__:
    __path__.append(str(_root))

for directory in tuple(__path__):
    shared = Path(directory) / "prompt_collection"
    if shared.is_dir():
        __path__.append(str(shared))

sys.meta_path.append(_DatasetImportAliases())
