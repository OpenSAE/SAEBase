"""The branch-based Pages publish must keep all five Atlas views available."""

from harness.visualization.sync_pages_fallback import ASSETS, ROOT, VIEWS, expected_files


def test_branch_pages_mirror_matches_canonical_atlas():
    expected = expected_files()
    assert len(expected) == 7
    for path, content in expected.items():
        assert path.is_file(), path
        assert path.read_text(encoding="utf-8") == content, path
    assert (ROOT / ".nojekyll").is_file()


def test_branch_pages_runtime_and_five_view_navigation():
    for view in VIEWS:
        page = (ROOT / "atlas" / (view + ".html")).read_text(encoding="utf-8")
        assert "SAEBase" in page
        for asset in ASSETS:
            assert f"../harness/visualization/{asset}" in page
            assert (ROOT / "harness" / "visualization" / asset).is_file()
        for tab in VIEWS[1:]:
            assert f'href="{tab}.html"' in page


def test_project_home_points_to_public_atlas():
    root_html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert "./atlas/index.html" in root_html
    assert "window.location.replace" in root_html


def test_pages_snapshots_regenerate_branch_fallback():
    workflow = (ROOT / ".github" / "workflows" / "saebase-atlas-snapshot.yml").read_text(
        encoding="utf-8"
    )
    assert "python harness/visualization/sync_pages_fallback.py" in workflow
    assert "python harness/visualization/sync_pages_fallback.py --check" in workflow
    deploy = (ROOT / ".github" / "workflows" / "atlas-project-pages.yml").read_text(
        encoding="utf-8"
    )
    assert "Verify deployed Atlas pages and live assets" in deploy
