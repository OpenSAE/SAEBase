"""Synchronize the Atlas fallback for branch-based GitHub Pages deployments.

SAEBase currently has an optimized Actions Pages publisher and a built-in
Jekyll Pages publisher. The latter can deploy after the former. Both must
serve working /SAEBase/atlas/{index,core,knowledge,code,environment,context}.html.
The fallback reuses canonical source-owned JS/CSS without copying the graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ("index", "core", "knowledge", "code", "environment", "context")
ASSETS = (
    "style.css",
    "atlas-theme.css",
    "data.js",
    "atlas-structure.js",
    "atlas-public-balance.js",
    "app.js",
    "locale-data.js",
    "locale-runtime.js",
    "i18n.js",
)
BRAND = "https://github.com/OpenSAE/SAEBase"
GUIDE = BRAND + "/blob/main/harness/visualization/README.md"

ROOT_INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="0;url=./atlas/index.html">
  <title>SAEBase Atlas</title>
</head>
<body>
  <a href="./atlas/index.html">Open SAEBase Atlas</a>
  <script>window.location.replace("./atlas/index.html"+window.location.search+window.location.hash);</script>
</body>
</html>
"""


def render_page(view: str, source: str) -> str:
    if view not in VIEWS:
        raise ValueError("Unknown Atlas view")
    html = source
    for asset in ASSETS:
        if not (HERE / asset).is_file():
            raise FileNotFoundError("Missing Atlas source asset " + asset)
        html = html.replace(
            f'"./{asset}',
            f'"../harness/visualization/{asset}',
        )
    html = html.replace(
        'href="../../README.md" title="SAEBase project"',
        f'href="{BRAND}" title="SAEBase project"',
    )
    html = html.replace(
        'href="../README.md">Guide',
        f'href="{GUIDE}">Guide',
    )
    html = html.replace(
        "<!doctype html>",
        "<!doctype html>\n<!-- SAEBase Pages fallback, generated from harness/visualization -->",
        1,
    )
    for asset in ("data.js", "app.js", "style.css"):
        if f"../harness/visualization/{asset}" not in html:
            raise ValueError("Missing Atlas runtime asset " + asset)
    for other in VIEWS[1:]:
        if f'href="{other}.html"' not in html:
            raise ValueError("Missing Atlas view navigation " + other)
    return html


def expected_files() -> dict[Path, str]:
    expected = {ROOT / "index.html": ROOT_INDEX}
    for view in VIEWS:
        source = (HERE / f"{view}.html").read_text(encoding="utf-8")
        expected[ROOT / "atlas" / f"{view}.html"] = render_page(view, source)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in expected_files().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if args.check:
        if not (ROOT / ".nojekyll").is_file():
            stale.append(".nojekyll")
    else:
        (ROOT / ".nojekyll").touch(exist_ok=True)
    if stale:
        print("Stale/missing Atlas Pages fallback: " + ", ".join(stale), file=sys.stderr)
        return 1
    print("Verified Atlas Pages fallback" if args.check else "Generated Atlas Pages fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
