"""Generate five direct-link Atlas HTML pages from the canonical index template.

The produced pages are deployable static files, but publishing to a public
Pages repository is a separate deliberate step.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
VIEWS = ("core", "knowledge", "code", "environment", "context")
NAV = re.compile(
    r'<button class="tab(?: active)?" data-graph="(core|knowledge|code|environment|context)">(.*?)</button>',
    flags=re.DOTALL,
)

def build_page(template: str, view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"Unexpected view {view!r}")
    if 'id="atlasLanguageSwitch"' not in template or 'src="./i18n.js' not in template:\n        raise ValueError("Atlas template requires a visible bilingual switch and its runtime")
    html = template.replace("<body>", f'<body data-atlas-view="{view}">', 1)
    html = html.replace(
        "<title>OpenResearch · Atlas</title>",
        f"<title>OpenResearch Atlas · {view.title()}</title>", 1,
    )
    html = html.replace(
        'href="../../README.md" title="OpenResearch project"',
        'href="https://github.com/OpenSAE/OpenResearch" title="OpenResearch project"', 1,
    )
    html = html.replace(
        'href="../README.md">Guide',
        'href="https://github.com/OpenSAE/OpenResearch/blob/main/harness/visualization/README.md">Guide', 1,
    )
    def nav_link(match: re.Match[str]) -> str:
        current = match.group(1)
        cls = "tab active" if current == view else "tab"
        return f'<a class="{cls}" data-graph="{current}" href="{current}.html">{match.group(2)}</a>'
    html, count = NAV.subn(nav_link, html)
    if count != len(VIEWS) or f'data-atlas-view="{view}"' not in html:
        raise ValueError("Atlas template no longer matches expected layout")
    return html

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    template = (HERE / "index.html").read_text(encoding="utf-8")
    stale = []
    for view in VIEWS:
        destination = HERE / f"{view}.html"
        content = build_page(template, view)
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != content:
                stale.append(str(destination.name))
        else:
            destination.write_text(content, encoding="utf-8")
            print(f"Generated {destination}")
    # Retain old bookmarked glossary.html links, including ?focus=term.ID.
    legacy = (HERE / "glossary.html")
    redirect = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url=context.html">
<title>Atlas · Context</title></head>
<body><a href="context.html">查看 Context</a>
<script>window.location.replace("context.html" + window.location.search + window.location.hash);</script>
</body></html>
"""
    if args.check:
        if not legacy.exists() or legacy.read_text(encoding="utf-8") != redirect:
            stale.append("glossary.html")
    else:
        legacy.write_text(redirect, encoding="utf-8")
        print(f"Generated {legacy} (legacy URL redirect)")
    if stale:
        print("Stale Atlas pages: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
