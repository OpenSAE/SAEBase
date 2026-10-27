# Atlas bilingual language switching specification

## Goal

The same Chinese / English selection governs all five Atlas trees (Core, Knowledge, Code, Environment, Context), node details, Context glossary, review controls, and every document surfaced by the Atlas documentation reader. Language switching must preserve the exact node ID, selected tree, expanded nodes, focus, zoom, review decision, URL review token, and scroll position.

## User behavior

- Put one prominent, always-visible `中文 | English` segmented switch in the **top-right of the Atlas global header**, beside the five-tree navigation and outside any collapsible menu, Details drawer, or node card. Give it strong visual contrast and an accessible text label, and retain it in a sticky header while the tree canvas scrolls. On narrow screens preserve an immediately visible compact switch; never bury it in settings or a hamburger menu. Default from a saved user preference; otherwise use the browser language, defaulting to English for other locales. Support `?lang=zh` and `?lang=en` deep links.
- Switching updates current content immediately without page navigation or tree regeneration.
- Keep the chosen language across the five Tree links and document links. Review forms, progress counters, shortcuts, source evidence, and backend operations remain the same in both languages.
- **Strict English mode:** Every human-facing Atlas UI label and explanatory sentence must be in English, including tree/node titles, summaries, details, glossary, attributes, reviewer controls, empty states, errors and all document prose. Never fall back to Chinese text in English mode. Keep only language-neutral, untranslatable source material such as verbatim code, original file paths, exact identifiers, citations, formal names and quotations in their original form. If a required English translation is missing, show an English-only `English translation unavailable` message for that field and fail the bilingual release gate; do not display Chinese prose.
- **Chinese mode:** Use natural Chinese throughout navigation and explanations, but retain standard English technical vocabulary, API/class/function names, model names, acronyms and established terms when translating would reduce precision. Missing Chinese copy may show a Chinese-only `中文翻译待补充` notice with the original English text as a documented temporary fallback. Full coverage remains a release requirement.

## Data model

All stable IDs, evidence paths, source references, code, AST signatures, config keys, model identifiers, and numeric facts are language-neutral.

Store human-facing data in locale records keyed by stable node/document IDs:

```json
{
  "node_id": "g.registry",
  "i18n": {
    "zh": {"title": "Dataset、Model、Task 插件注册", "summary": "...", "description": "..."},
    "en": {"title": "Dataset, Model, and Task plugin registration", "summary": "...", "description": "..."}
  }
}
```

Nested interface descriptions, parameter meanings, return meanings, attributes, glossary definitions, aliases, reviewer-facing labels, and graph explanations must also be localized. Never translate arbitrary values or code dynamically.

Keep authoritative scientific data and evidence in existing JSON; retain language data as a separate overlay with validation of target IDs and supported fields. Generated AST snapshots stay reproducible and can be regenerated without losing reviewed translations.

## Documents

Use stable documentation IDs and a paired locale representation for every Markdown document shown in Atlas. Original source path remains authoritative; the localized representation is a sibling or an i18n overlay, with explicit mappings in a manifest. Preserve heading anchors, internal links, code fences, citations, tables, equations, file paths and exact commands. A language switch must route to the same document ID and heading in the other locale.

For every `doc/**` Markdown source displayed in the documentation reader, provide both `zh` and `en` versions before marking bilingual coverage complete. Also include any documentation from `harness/**`, README files, and other repository areas if linked from a Tree node. Distinguish technical source evidence from reader-facing explanatory documents: source code is always verbatim.

## Export and frontend

Export localized node overlays and localized documentation manifest together with the existing Atlas graph export. `app.js` reads the active locale through one shared resolver, not independent per-Tree switches. Localize the entire visible UI, including navigation, Details, attributes, review decisions, filters, search, tooltip, empty state and errors. The language switch itself may display `中文 | English` in either mode solely as self-identifying language choices; all other English-mode UI prose must be English. Existing ids/edges/DFS order are unaffected.

The public Atlas lives in the website repository and must receive its corresponding app, locale files, documents, HTML asset revision and cache busting in the same rollout.

## Validation

1. Both locale files have the same stable node/document IDs.
2. Both language variants exist for each document included in the coverage declaration; missing translations fail the release gate.
3. All displayed prose fields are checked, including nested node attributes and glossary definitions.
4. Switching languages preserves focused node, expanded cards, DFS review position, counters, pending edits, and keyboard shortcuts.
5. Every five Tree route and an example document deep link can switch zh -> en -> zh without broken references.
6. English-mode automated checks reject Chinese CJK characters in user-facing prose except explicitly allowlisted verbatim source evidence, proper nouns or citations. Chinese-mode checks allow common technical English terms.
7. Verify the switch is visible without scrolling on desktop, remains visible while scrolling the Atlas canvas, and is directly accessible on mobile without opening a menu.
8. A generated build and published website smoke test verify content after refresh, direct links and stale-cache prevention.

## Deployment status

Specification only. This file does not establish that translations or the published Atlas switch are implemented.
