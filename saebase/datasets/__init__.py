"""Source-tree alias: physical Dataset implementations live in /datasets.
Installed wheels package them under saebase.datasets to avoid shadowing Hugging Face.
"""

from pathlib import Path

_root = Path(__file__).resolve().parents[2] / "datasets"
if _root.is_dir() and str(_root) not in __path__:
    __path__.append(str(_root))
