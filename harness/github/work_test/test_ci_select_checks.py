"""CI classification must stay conservative across PR and main diffs."""

from __future__ import annotations

from harness.github.ci.select_checks import classify


def test_harness_and_atlas_changes_are_lightweight():
    flags = classify([
        "harness/github/human_check.py",
        "harness/visualization/app.js",
        "harness/core/AGENT_FLOW.md",
    ])
    assert flags == {
        "harness": True, "atlas": True, "product": False, "ci_policy": False,
    }


def test_main_push_uses_union_not_one_category():
    flags = classify([
        "tasks/capture/operation.py",
        "harness/core/agent_flow.py",
        "doc/knowledge/glossary.json",
        "pyproject.toml",
    ])
    assert flags["harness"] and flags["atlas"] and flags["product"]


def test_rename_checks_both_old_and_new_owners():
    flags = classify(["models/gemma/old.py", "harness/code/new.py"])
    assert flags["product"] and flags["atlas"]


def test_ci_changes_always_select_harness():
    flags = classify([".github/workflows/ci.yml"])
    assert flags["harness"] and flags["ci_policy"]


def test_docs_only_do_not_install_analysis():
    assert not any(classify(["README.md"]).values())
    docs = classify(["harness/github/development_doc/note.md"])
    assert docs["harness"] and not docs["product"] and not docs["atlas"]


def test_unknown_or_invalid_scope_is_conservative():
    for value in ("unknown/path.txt", "../private", "/absolute", "weird\\path"):
        assert all(classify([value]).values())


def test_missing_git_diff_runs_all_light_checks():
    assert all(classify([], unknown_diff=True).values())


def test_no_path_selection_never_has_release_authority():
    assert "release" not in classify(["pyproject.toml"])
    assert "publish" not in classify([".github/workflows/ci.yml"])
