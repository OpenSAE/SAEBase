> **Normative prerequisite:** [SAEBase Atlas Authoring Standard](ATLAS_AUTHORING_STANDARD.en.md). OpenResearch uses independent authoritative graphs and deployment routes.

# OpenResearch Atlas: Source-Backed Research Trees

Core, Knowledge, Code, Environment, and Context share a centered, expandable tree interface with node focus, relationships, and source evidence. Core, Knowledge, Code, and Context describe evidence-backed project or research content. Environment displays measured or explicitly synthetic execution resources.

## Interaction

Click the child-count control to expand or collapse a branch. Select a card to focus its node while preserving neighboring branches. Expand **Attributes and Evidence** to view the node's explanation inside the tree. Resizing occurs within the same canvas; Fit is an explicit user action.

Function cards show signatures, parameters, types, defaults, return meanings, and source excerpts. Class cards expose members and methods. Module cards collect actual functions and classes. Context cards define scientific and engineering terminology, source links, and relevant formulas; Knowledge cards distinguish research questions from implemented APIs.

The display prioritizes evidence-backed explanations and links to actual Python source and related Context definitions.

## Build and validation

```bash
python harness/core/validate.py
python harness/knowledge/validate.py
python harness/knowledge/context_attributes.py
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
python -m http.server 8765
```

Open [Core](https://atheavenma.github.io/opensae/saebase/atlas/core.html), [Knowledge](https://atheavenma.github.io/opensae/saebase/atlas/knowledge.html), [Code](https://atheavenma.github.io/opensae/saebase/atlas/code.html), [Environment](https://atheavenma.github.io/opensae/saebase/atlas/environment.html), or [Context](https://atheavenma.github.io/opensae/saebase/atlas/context.html). The public environment view may include clearly labeled synthetic resources.

The published Atlas is maintained in a separate website repository. Its HTML, JavaScript, language overlay, and generated graph data must remain compatible. Updating the source repository alone does not publish a new site snapshot.

## Component hierarchy

Code Tree follows actual source directories, responsibilities, Python modules, classes and functions, and methods. The interface displays at most ten actual children per page. Additional children use UI pagination, not synthetic Tree nodes or numbered source ranges. Every generated feature node needs an independent evidenced responsibility and a valid authoritative Context reference.

Individual datasets, models, and seven SAE research operations are shown separately. Configuration is attached to the corresponding task rather than embedded in a generic task name.

## Node semantics and languages

Node titles describe what each implementation does. Python identifiers, source line numbers, signatures, input and output contracts, and complete methods are available inside the expanded node. `harness/code/semantic_titles.py` and `harness/code/presentation.py` generate source-linked titles and details.

Every Tree has Chinese and English display content linked by immutable node IDs. Context term definitions come from `doc/knowledge/glossary.json` and `doc/knowledge/atlas_terms.json`; the runtime English overlay is in `harness/visualization/locale-data.js`. Switching languages must preserve node focus, expansion, and review state without changing identifiers or evidence.

English Markdown counterparts use the same document ID and a `.en.md` suffix where the original is Chinese. A missing translation must be reported rather than treated as an English scientific explanation. See [Atlas bilingual contract](ATLAS_I18N_PLAN.md).

## Human review

The existing tree supports human-review drafts. Purple represents unchecked, red an approved but not implemented placeholder, and gray a denied proposal. Review navigation follows actual tree order, with selectors and keyboard shortcuts; it does not turn a proposal into verified scientific evidence.

A local authenticated review service can persist permitted decisions. The public GitHub Pages interface saves browser drafts and offers one localized **Send report / 发送报告** link. It opens a prefilled GitHub Issue that the user must submit; it does not directly change the authority graph or record authenticated approval.

See the [Agent review process](../core/AGENT_FLOW.en.md). Status remains independently evidenced per node, and published snapshots do not substitute for hardware tests or research verification.

## Research-concept Code Tree

The `saebase/` branch now begins with 14 source-backed research domains, followed by meaningful subdomains and actual modules. `harness/code/atlas_concepts.json` defines the reviewed bilingual responsibilities, Context terms, and precise source membership. Original module IDs, paths, signatures, and methods are retained. A functional parent has no more than 20 direct children, with 10 preferred; UI pagination does not replace the scientific hierarchy.

## Full-depth ancestry and canvas zoom

The mouse wheel zooms the graph at the pointer, including over cards. It no longer changes the selected sibling. Expanded code and explanations keep native scrolling. Focus retains the entire root-to-current ancestor chain inside the SVG and exposes every ancestor as a clickable title in a canvas-level path navigator. At 80% or lower, the navigator wraps titles and node cards use compact title-only density. Fit calculates the bounds of real path nodes, rather than focus-centering padding. This supports 20 or more levels without discarding ancestors or their Context and source links.

## Complete levels and unconstrained camera

Focus displays every real node at every depth preceding the selected node. At the selected depth, only the full sibling group belonging to its parent appears, while the selected node's immediate children remain available for deeper navigation. Siblings are never paginated or capped. Drag blank canvas or any ordinary card to pan freely in screen coordinates at all zoom levels without center locking, scroll boundaries or bounce resistance. Explicit node selection brings that node into view once, then leaves the camera entirely user-controlled. Wheel zoom remains pointer-centered; Fit includes every visible node.
