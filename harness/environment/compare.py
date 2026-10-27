"""Compare two captured machine profiles without inventing missing values."""

import argparse
import json
from pathlib import Path
import sys

FIELDS = (
    "platform.system",
    "platform.machine",
    "cpu.logical_count",
    "cpu.model",
    "memory.total_bytes",
    "gpu.status",
    "gpu.devices",
    "storage.total_bytes",
    "execution.python_version",
    "execution.commands",
)


def at(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    args = p.parse_args()
    a = json.loads(Path(args.before).read_text())
    b = json.loads(Path(args.after).read_text())
    if a.get("synthetic") or b.get("synthetic"):
        print(
            "WARNING: synthetic profile present; results are a visualization example, not observed comparison"
        )
    diff = [
        {"field": f, "before": at(a, f), "after": at(b, f)} for f in FIELDS if at(a, f) != at(b, f)
    ]
    print(json.dumps({"changes": diff, "count": len(diff)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
