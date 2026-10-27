"""Load Hugging Face datasets in an isolated interpreter.

OpenResearch owns the top-level 'datasets' package for its actual dataset
implementations. The external Hugging Face package uses the same import name.
Run its read-only loader in an isolated child process rather than replacing
sys.modules, changing the global import search path or mixing package objects.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


_LOADER = """
import json, sys
request = json.loads(sys.stdin.read())
import datasets
operation = request["operation"]
if operation == "load_dataset":
    rows = datasets.load_dataset(*request["args"], **request["kwargs"])
elif operation == "load_from_disk":
    rows = datasets.load_from_disk(*request["args"], **request["kwargs"])
else:
    raise ValueError("Unsupported dataset reader")
def serializable(obj):
    if hasattr(obj, "keys"):
        return {key: serializable(obj[key]) for key in obj.keys()}
    return [dict(row) for row in obj]
sys.stdout.write(json.dumps(serializable(rows), ensure_ascii=False, allow_nan=False))
"""


class IsolatedHFDatasets:
    """Return native text rows with normal list/dict behavior.

    This preserves the caller-facing load_dataset/load_from_disk API used by
    the checked-in CounterFact and MMLU-Pro adapters. Loading is opt-in.
    """

    def _load(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        request = json.dumps(
            {"operation": operation, "args": args, "kwargs": kwargs}, allow_nan=False
        )
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _LOADER],
                input=request,
                text=True,
                capture_output=True,
                timeout=900,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "The isolated Hugging Face datasets loader failed: "
                + (exc.stderr or "").strip()[-1800:]
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Hugging Face dataset loading exceeded the time budget") from exc
        data = json.loads(result.stdout)
        if not isinstance(data, (dict, list)):
            raise TypeError("Hugging Face loader returned an unexpected data structure")
        return data

    def load_dataset(self, *args: Any, **kwargs: Any) -> Any:
        return self._load("load_dataset", *args, **kwargs)

    def load_from_disk(self, *args: Any, **kwargs: Any) -> Any:
        return self._load("load_from_disk", *args, **kwargs)
