# OpenSAE Research Lab server and offline data tools.

from pathlib import Path as _Path

# Keep existing module names after grouping their source files by purpose.
__path__.extend(str(_Path(__file__).parent / name) for name in ("audits", "experimental_design"))
