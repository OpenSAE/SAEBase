"""CI classification must stay conservative across PR and main diffs."""

from __future__ import annotations

from scripts.quality import classify


def test_harness_and_atlas_changes_are_lightweight():
    flags = classify(
        [
            "harness/github/human_check.py",
            "harness/visualization/app.js",
            "harness/core/AGENT_FLOW.md",
        ]
    )
    assert flags == {
        "harness": True,
        "atlas": True,
        "product": False,
        "ci_policy": False,
    }


def test_main_push_uses_union_not_one_category():
    flags = classify(
        [
            "tasks/capture/operation.py",
            "harness/core/agent_flow.py",
            "doc/code/atlas_site/source/knowledge/glossary.json",
            "pyproject.toml",
        ]
    )
    assert flags["harness"] and flags["atlas"] and flags["product"]


def test_rename_checks_both_old_and_new_owners():
    flags = classify(["models/gemma/old.py", "harness/code/new.py"])
    assert flags["product"] and flags["atlas"]


def test_ci_changes_always_select_harness():
    flags = classify([".github/workflows/ci.yml"])
    assert flags["harness"] and flags["ci_policy"]


def test_docs_only_do_not_install_analysis():
    assert not any(
        classify(["README.md", "AGENTS.md", ".agents/skills/add-task/SKILL.md"]).values()
    )
    docs = classify(["doc/code/documentation/source/development/note.md"])
    assert not any(docs.values())


def test_release_metadata_edits_do_not_need_torch():
    flags = classify(["scripts/release.py"])
    assert flags["harness"] and flags["ci_policy"]
    assert not flags["product"] and not flags["atlas"]


def test_unknown_or_invalid_scope_is_conservative():
    for value in ("unknown/path.txt", "../private", "/absolute", "weird\\path"):
        assert all(classify([value]).values())


def test_missing_git_diff_runs_all_light_checks():
    assert all(classify([], unknown_diff=True).values())


def test_no_path_selection_never_has_release_authority():
    assert "release" not in classify(["pyproject.toml"])
    assert "publish" not in classify([".github/workflows/ci.yml"])
