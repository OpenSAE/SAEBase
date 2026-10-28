"""Regression for standalone SAEBase Atlas navigation on the index entrypoint."""

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "harness" / "visualization"
spec = importlib.util.spec_from_file_location("saebase_atlas_pages", ATLAS / "build_pages.py")
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


def test_index_and_generated_pages_link_to_all_five_views():
    template = (ATLAS / "index.html").read_text(encoding="utf-8")
    assert '<button class="tab' not in template
    for view in pages.VIEWS:
        direct_link = f'data-graph="{view}" href="{view}.html"'
        assert direct_link in template, f"Index lacks clickable {view} tab"
        html = pages.build_page(template, view)
        assert f'<body data-atlas-view="{view}">' in html
        assert f'<a class="tab active" {direct_link}' in html
        for other in pages.VIEWS:
            assert f'data-graph="{other}" href="{other}.html"' in html
        assert '<button class="tab' not in html
        assert (ATLAS / f"{view}.html").read_text(encoding="utf-8") == html
