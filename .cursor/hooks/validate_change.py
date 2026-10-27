"""Post-edit syntax diagnostic; a failed check does not undo the edit."""

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    event = json.load(sys.stdin)
    raw = (
        event.get("file_path")
        or event.get("path")
        or (event.get("tool_input") or {}).get("file_path")
    )
    if not raw:
        print("{}")
        return
    p = pathlib.Path(raw)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        print("{}")
        return
    if p.suffix == ".py":
        ast.parse(p.read_text(encoding="utf-8"))
    if p.suffix == ".json":
        json.loads(p.read_text(encoding="utf-8"))
    print("{}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {"agent_message": f"Edited file has syntax errors (edit was not reverted): {exc}"}
            )
        )
