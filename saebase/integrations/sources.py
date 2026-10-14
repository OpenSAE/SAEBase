"""Load a paper's unmodified estimator from a user-provided local checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_EXPECTED = {"sar": ("coalitions", "sar"), "rss": ("rss", "held_out_rate")}


def load_paper_module(path: str | Path, *, paper: str) -> ModuleType:
    """Load an explicitly supplied ``sar.py`` or ``rss.py`` file.

    Does not modify ``sys.path`` or shadow installed packages. The file is
    trusted local Python code supplied by the caller, not downloaded by SAEBase.
    """
    if paper not in _EXPECTED:
        raise ValueError("paper must be 'sar' or 'rss'")
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / f"{paper}.py"
    if not source.is_file() or source.name != f"{paper}.py":
        raise FileNotFoundError(f"Provide a local {paper}.py from the {paper.upper()} repository; got {source}")
    spec = importlib.util.spec_from_file_location(f"_saebase_reference_{paper}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load estimator file {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in _EXPECTED[paper]:
        if not callable(getattr(module, name, None)):
            raise TypeError(f"{source} must define the paper function {name}()")
    return module
