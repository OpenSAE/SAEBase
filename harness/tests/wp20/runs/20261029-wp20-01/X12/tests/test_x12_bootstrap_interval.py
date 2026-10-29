from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "fixtures/frozen.json").read_text(encoding="utf-8"))
EXTRA = json.loads((ROOT / "fixtures/extra.json").read_text(encoding="utf-8"))


def _load():
    sys.path[:] = [p for p in sys.path if "wp20/runs" not in Path(p).as_posix()]
    sys.path.insert(0, str(ROOT))
    for name in list(sys.modules):
        if name in {"src", "implementation"} or name.startswith("src."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("implementation", ROOT / "implementation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["implementation"] = module
    spec.loader.exec_module(module)
    return module.run, module.InputError


def _payload(row):
    return copy.deepcopy(row["input"])


@pytest.mark.parametrize("row", FROZEN, ids=lambda row: row["id"])
def test_frozen_scenarios(row, tmp_path):
    run, error = _load()
    payload = _payload(row)
    snapshot = copy.deepcopy(payload)
    if row["valid"]:
        result = run(payload, tmp_path)
        assert isinstance(result, dict)
        assert result not in ({"accepted": True}, {"ok": True})
        assert payload == snapshot
    else:
        with pytest.raises((error, ValueError)):
            run(payload, tmp_path)
        assert payload == snapshot


@pytest.mark.parametrize("row", EXTRA, ids=lambda row: row["id"])
def test_extra_boundary_scenarios(row, tmp_path):
    run, error = _load()
    payload = _payload(row)
    snapshot = copy.deepcopy(payload)
    if row["valid"]:
        result = run(payload, tmp_path)
        assert isinstance(result, dict)
        assert payload == snapshot
    else:
        with pytest.raises((error, ValueError)):
            run(payload, tmp_path)
        assert payload == snapshot


def test_non_object_payload_is_rejected(tmp_path):
    run, error = _load()
    with pytest.raises((error, ValueError)):
        run(["not", "a", "mapping"], tmp_path)


def test_public_reference_is_not_imported():
    import ast

    banned = "harness.contracts.wp20_reference"
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(banned):
                raise AssertionError(path)
            if isinstance(node, ast.Import) and any(
                alias.name.startswith(banned) for alias in node.names
            ):
                raise AssertionError(path)
