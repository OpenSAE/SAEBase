"""H01 自测：稳定摘要、单文件变化，以及越界和类型错误必须拒绝。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_r2_h01", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_named_fixture_order_is_stable() -> None:
    module = _load()
    root = Path(__file__).resolve().parent / "fixtures"
    first = module.index_named_files(root, ["leaf-south.txt", "leaf-north.txt"])
    second = module.index_named_files(root, ["leaf-north.txt", "leaf-south.txt"])
    assert first == second
    assert [row["path"] for row in first] == ["leaf-north.txt", "leaf-south.txt"]
    assert first[0]["sha256"] == hashlib.sha256((root / "leaf-north.txt").read_bytes()).hexdigest()
    assert "marker.json" not in [row["path"] for row in first]


def test_one_byte_change_touches_only_that_row(tmp_path: Path) -> None:
    module = _load()
    (tmp_path / "leaf-north.txt").write_text("north-sample\n", encoding="utf-8")
    (tmp_path / "leaf-south.txt").write_text("south-sample\n", encoding="utf-8")
    (tmp_path / "hidden-token.txt").write_text("TOKEN-VALUE-NOT-INDEXED\n", encoding="utf-8")
    before = module.index_named_files(tmp_path, ["leaf-south.txt", "leaf-north.txt"])
    (tmp_path / "leaf-south.txt").write_text("south-sample-edited\n", encoding="utf-8")
    after = module.index_named_files(tmp_path, ["leaf-north.txt", "leaf-south.txt"])
    assert after[0] == before[0]
    assert after[1]["sha256"] != before[1]["sha256"]
    assert "TOKEN-VALUE-NOT-INDEXED" not in json.dumps(after)


def test_empty_list_and_bad_types() -> None:
    module = _load()
    root = Path(__file__).resolve().parent / "fixtures"
    assert module.index_named_files(root, []) == []
    for bad in ("leaf-north.txt", {"path": "leaf-north.txt"}, [1]):
        try:
            module.index_named_files(root, bad)  # type: ignore[arg-type]
        except module.IndexRefusal:
            pass
        else:
            raise AssertionError(bad)


def test_reject_escape_symlink_duplicate_and_missing(tmp_path: Path) -> None:
    module = _load()
    (tmp_path / "leaf-north.txt").write_text("north-sample\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-sample.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "leak").symlink_to(outside)
    (tmp_path / "alias.txt").symlink_to(tmp_path / "leaf-north.txt")
    for raw in ("../outside-sample.txt", "/tmp/outside-sample.txt", "leaf-north.txt/../../x"):
        try:
            module.index_named_files(tmp_path, [raw])
        except module.IndexRefusal:
            pass
        else:
            raise AssertionError(raw)
    try:
        module.index_named_files(tmp_path, ["leak"])
    except module.IndexRefusal:
        pass
    else:
        raise AssertionError("symlink")
    try:
        module.index_named_files(tmp_path, ["leaf-north.txt", "leaf-north.txt"])
    except module.IndexRefusal:
        pass
    else:
        raise AssertionError("duplicate")
    try:
        module.index_named_files(tmp_path, ["missing-leaf.txt"])
    except module.IndexRefusal:
        pass
    else:
        raise AssertionError("missing")
    alias = module.index_named_files(tmp_path, ["alias.txt"])
    assert alias[0]["path"] == "alias.txt"
    assert alias[0]["sha256"] == hashlib.sha256(b"north-sample\n").hexdigest()


def test_candidate_matches_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert payload["provisional"] is True
    by_name = {
        change["signature"].split("(")[0].split()[-1]: change
        for change in payload["changes"]
        if change["path"].endswith("implementation.py") and change["signature"]
    }
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert by_name[node.name]["line"] == node.lineno
            assert by_name[node.name]["status"] == "unchecked"
            assert any(node.name in item for item in expected["added_ids"])
