"""Exercise the project-owned SAEBase Atlas preview in Chromium."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = "http://127.0.0.1:8765/preview/"
shots = Path("atlas-preview-screenshots")
shots.mkdir(exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1550, "height": 950})
    context.grant_permissions(["clipboard-read", "clipboard-write"],
        origin="http://127.0.0.1:8765")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(ROOT + "code.html?lang=en", wait_until="networkidle", timeout=90000)
    page.wait_for_function("() => !!window.OPENRESEARCH_ATLAS?.context?.nodes?.length")
    launcher = page.locator("#atlasGlobalSearchButton")
    assert launcher.is_visible()
    assert launcher.inner_text().strip() == "Search all trees", launcher.inner_text()
    page.locator(".graph-node").first.click()
    assert page.locator("#detailDrawer").is_visible(), "Node click did not reveal Details"
    assert page.locator("#atlasCopyFocused").is_visible()
    for _ in range(2):
        page.locator("#zoomOut").click()
    page.wait_for_function("""() => document.querySelectorAll(
        ".graph-node.density-compact").length > 0""")
    assert page.locator(".graph-node.density-compact .attribute-trigger").count() == 0, (
        "Compact mode still renders grey detail bars")
    assert page.locator(".graph-node.density-compact .node-title").count() > 0
    page.screenshot(path=str(shots / "05-compact-no-grey-footer.png"))
    page.locator("#zoomIn").click()
    page.locator("#zoomIn").click()
    page.keyboard.press("Control+k")
    assert page.locator("#atlasGlobalSearch").is_visible()
    page.locator("#atlasGlobalSearchInput").fill("SAE")
    assert page.locator(".atlas-search-result").count() > 0
    assert page.locator(".atlas-search-filter").count() == 6
    page.screenshot(path=str(shots / "01-global-search-en.png"))
    page.keyboard.press("Escape")
    assert page.locator("#atlasGlobalSearch").is_hidden()
    # Select a real node referenced explicitly from another tree.
    pair = page.evaluate("""() => {
      const data = window.OPENRESEARCH_ATLAS;
      const refs = [
        ["context_term_ids","context"],["context_ids","knowledge"],
        ["core_ids","core"],["term_ids","context"]
      ];
      for (const [view,tree] of Object.entries(data)) {
        for (const n of tree.nodes||[]) {
          for (const [field,targetView] of refs) {
            if (targetView === view) continue;
            for (const targetId of n[field]||[]) {
              if(data[targetView]?.nodes?.some(t=>t.id===targetId))
                return {view:targetView,id:targetId,field,sourceView:view};
            }
          }
        }
      }
      return null;
    }""")
    assert pair is not None, "No explicit cross-tree references in source data"
    page.goto(ROOT + pair["view"] + ".html?lang=en&focus=" + pair["id"],
        wait_until="networkidle", timeout=90000)
    inspector = page.locator("#detailDrawer")
    assert inspector.is_visible(), "Details should automatically open for a focused node"
    assert page.locator(".atlas-detail-layer").count() == 3
    assert page.locator(".atlas-relation-row").count() >= 1, (
        "Reverse reference to " + str(pair) + " was omitted")
    shortcut=page.locator("#atlasRelationsShortcut")
    assert shortcut.is_visible(), "Relations shortcut is hidden"
    assert "Relations" in shortcut.inner_text()
    assert page.locator(".atlas-relation-flow").count() > 0
    assert page.locator(".atlas-relation-arrow").first.inner_text() == "→"
    shortcut.click()
    assert page.locator("#atlasRelationsSection").is_visible()
    assert page.locator(".atlas-relation-direction").filter(
        has_text="Referenced by").count() >= 1
    assert page.locator("#atlasCopyFocused").is_visible(), "Toolbar Copy link hidden"
    assert page.locator(".atlas-copy-link").is_visible(), "Inspector Copy link hidden"
    page.screenshot(path=str(shots / "02-auto-open-relations-copy.png"))
    expected_view=page.locator(".atlas-relation-view").first.inner_text().strip().lower()
    page.locator(".atlas-relation-link").first.click()
    page.wait_for_function("""view => document.body.dataset.atlasView === view""",arg=expected_view)
    assert page.locator("#detailDrawer").is_visible()
    page.goto(ROOT + pair["view"] + ".html?lang=en&focus=" + pair["id"],
        wait_until="networkidle", timeout=90000)
    page.locator("#atlasCopyFocused").click()
    copied = page.evaluate("() => navigator.clipboard.readText()")
    assert "focus=" in copied and pair["id"] in copied
    assert "/preview/" in copied
    assert "review=" not in copied and "#key=" not in copied
    page.locator('#atlasLanguageSwitch button[data-lang="zh"]').click()
    page.wait_for_function("() => window.SAEBASE_ATLAS_LANGUAGE === 'zh'")
    assert page.locator("#atlasGlobalSearchButton").inner_text().strip() == "五树搜索"
    assert page.locator("#atlasCopyFocused").inner_text().strip() == "复制链接"
    page.screenshot(path=str(shots / "03-inspector-chinese.png"))
    page.keyboard.press("Control+k")
    assert page.locator("#atlasGlobalSearch").is_visible()
    page.screenshot(path=str(shots / "04-search-chinese.png"))
    assert not errors, "Page errors: " + "; ".join(errors)
    print("PASS: node click auto-Details, compact footer, directional relations, deep links, locales")
    browser.close()
