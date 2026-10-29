"""H01 自测：稳定摘要、单文件变化，以及越界路径必须拒绝。"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).with_name("implementation.py")
    spec = importlib.util.spec_from_file_location("wp18_h01_implementation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> None:
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (root / "beta.txt").write_text("beta\n", encoding="utf-8")
    (root / "secret.env").write_text("SECRET-TOKEN\n", encoding="utf-8")


def test_static_fixtures_match_bytes() -> None:
    module = _load()
    root = Path(__file__).resolve().parent / "fixtures"
    rows = module.index_files(root, ["beta.txt", "alpha.txt"])
    assert [row["path"] for row in rows] == ["alpha.txt", "beta.txt"]
    assert "manifest.json" not in [row["path"] for row in rows]
    assert rows[0]["sha256"] == hashlib.sha256((root / "alpha.txt").read_bytes()).hexdigest()


def test_repeat_is_byte_stable(tmp_path: Path) -> None:
    module = _load()
    _tree(tmp_path)
    first = module.index_files(tmp_path, ["beta.txt", "alpha.txt"])
    second = module.index_files(tmp_path, ["beta.txt", "alpha.txt"])
    assert first == second
    assert [row["path"] for row in first] == ["alpha.txt", "beta.txt"]
    assert json.dumps(first) == json.dumps(second)


def test_one_edit_changes_only_that_row(tmp_path: Path) -> None:
    module = _load()
    _tree(tmp_path)
    before = {row["path"]: row for row in module.index_files(tmp_path, ["alpha.txt", "beta.txt"])}
    (tmp_path / "beta.txt").write_text("beta-edited\n", encoding="utf-8")
    after = {row["path"]: row for row in module.index_files(tmp_path, ["alpha.txt", "beta.txt"])}
    assert after["alpha.txt"] == before["alpha.txt"]
    assert after["beta.txt"]["sha256"] != before["beta.txt"]["sha256"]
    assert after["beta.txt"]["size"] != before["beta.txt"]["size"]


def test_unlisted_secret_is_not_read(tmp_path: Path) -> None:
    module = _load()
    _tree(tmp_path)
    rows = module.index_files(tmp_path, ["alpha.txt"])
    assert rows[0]["path"] == "alpha.txt"
    assert "SECRET-TOKEN" not in json.dumps(rows)


def test_reject_escape_duplicate_and_missing(tmp_path: Path) -> None:
    module = _load()
    _tree(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    link = tmp_path / "escape"
    link.symlink_to(outside)
    for raw in ("../outside.txt", "alpha.txt/../../outside.txt", "/etc/passwd", "alpha.txt"):
        if raw == "alpha.txt":
            continue
        try:
            module.index_files(tmp_path, [raw])
        except module.EvidenceIndexError:
            pass
        else:
            raise AssertionError(raw)
    try:
        module.index_files(tmp_path, ["escape"])
    except module.EvidenceIndexError:
        pass
    else:
        raise AssertionError("symlink")
    try:
        module.index_files(tmp_path, ["alpha.txt", "alpha.txt"])
    except module.EvidenceIndexError:
        pass
    else:
        raise AssertionError("duplicate")
    try:
        module.index_files(tmp_path, ["missing.txt"])
    except module.EvidenceIndexError:
        pass
    else:
        raise AssertionError("missing")


def test_internal_symlink_uses_requested_path(tmp_path: Path) -> None:
    module = _load()
    _tree(tmp_path)
    (tmp_path / "alias.txt").symlink_to(tmp_path / "alpha.txt")
    rows = module.index_files(tmp_path, ["alias.txt"])
    assert rows[0]["path"] == "alias.txt"
    assert rows[0]["sha256"] == hashlib.sha256(b"alpha\n").hexdigest()


def test_candidate_binds_real_ast() -> None:
    case = Path(__file__).resolve().parent
    source = (case / "implementation.py").read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    payload = json.loads((case / "review_candidate.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "expected_code_diff.json").read_text(encoding="utf-8"))
    assert payload["source_content_sha256"] == digest
    assert payload["provisional"] is True
    tree = ast.parse(source)
    found = {change["signature"].split("(")[0].split()[-1]: change for change in payload["changes"]}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            change = found[node.name]
            assert change["line"] == node.lineno
            assert change["status"] == "unchecked"
            assert node.name in expected["added_ids"][-1] or any(node.name in item for item in expected["added_ids"])
