# SAEBase Atlas Tree Generation, Authoring, and Publication Contract

This is the authoritative **pre-implementation design contract** for the SAEBase Atlas. Review this document before changing any tree, scientific Context, translations, source mapping, user interface, or public deployment. Its scope is `OpenSAE/SAEBase`, not OpenResearch.

**Project identity:** `SAEBase`. **Public root:** `/opensae/saebase/atlas/`. **Source of truth:** `OpenSAE/SAEBase`.

## 1. Purpose and ownership

Atlas lets researchers inspect project architecture, scientific problems, actual implementation, measured runtime conditions, and technical vocabulary through five complementary trees. Presentation is not independent experimental evidence.

| Tree | Question | Authoritative source |
| --- | --- | --- |
| Core | Why is the system organized this way? | `harness/core/project_graph.json` |
| Knowledge | Which scientific questions and validated methods exist? | `doc/knowledge/context_graph.json` |
| Code | Which Python entities exist in the current source? | AST and `doc/code/code_tree.json` |
| Environment | Which execution resources are actually available? | `harness/environment/probe.py` and explicitly synthetic examples |
| Context | What precisely do scientific and engineering terms mean? | `doc/knowledge/glossary.json` and `doc/knowledge/atlas_terms.json` |

SAEBase and OpenResearch may reuse rendering components, never authoritative nodes, datasets, browser state, site paths, or publication snapshots. SAEBase's public routes are `/opensae/saebase/atlas/{core,knowledge,code,environment,context}.html`. OpenResearch retains its separate `/opensae/openresearch/atlas/` routes.

## 2. Context-first implementation order

1. Validate every term and category from both authoritative glossaries, including stable term IDs, definitions, aliases, why the concept matters, foundational documents, and verifiable evidence.
2. Author accurate English equivalents of Context `title`, `summary`, `description`, `why_it_matters`, and `aliases` in `doc/knowledge/i18n/context.en.json`, keyed by term ID.
3. Link the other trees to Context through `context_term_ids`, never copied definitions.
4. Translate and review each tree's own descriptions, in the order Knowledge, Core, Environment, then Code.
5. Validate complete locale coverage and source links before deploying separate, versioned Atlas pages.

## 3. Node semantics and hierarchy

Each node has a stable unique ID, type, explicit parent, concise semantic title, meaningful summary and description, independently audited status, and source evidence.

Titles describe the actual goal, scientific concept, or implemented operation. Do not display function names, chained path fragments, Work Plan codes, or mechanical CamelCase translations as explanatory titles. English and Chinese titles must both be natural and informative; precise technical names and source identities remain verbatim in technical fields.

Summaries explain a node's main role. Descriptions add inputs, processing, outputs, constraints, and provenance; merely duplicating a generic summary does not constitute a scientific explanation. When evidence is absent, mark the work for review, rather than inventing behavior.

Core/Knowledge authority edges and Code AST containment edges have distinct semantics. Cross-tree references remain references. Children keep the recorded edge or source-line order. Status is based on each node's own evidence; an implemented API does not prove scientific effectiveness or successful GPU experiments.

Display at most ten actual children per page. Pagination is UI state and must never create a Tree node. Reject numbered batches, source-range placeholders, and functional nodes without a unique source-backed purpose and a valid authoritative Context term ID.

## 4. Code Tree requirements

Modules, classes, fields, methods, function signatures, parameters, annotations, defaults, return expressions, code excerpts, and source lines are extracted from real Python AST. No guessed APIs.

Dataset, Model, and Task trees follow their actual first-level implementation directories rather than invented domain-category intermediate folders. Put readable, role-based titles first; preserve exact Python names and complete interfaces in expanded details.

Generated English Code titles need real source context and must pass checks against repetition, meaningless generic phrases, mojibake, accidental path concatenation, and incorrect inferred responsibilities. Reviewed, domain-specific explanations take precedence over automated naming.

## 5. Environment and scientific verification

Use measured environment observations, or explicitly `synthetic: true` demonstration snapshots with matching English and Chinese labels. Missing measurements remain `unknown`; the absence of a diagnostic tool does not prove hardware is absent.

Training quality, steering effects, feature explanations, benchmark improvements, and runtime performance require independent experiment receipts. Implemented code is not itself proof.

## 6. Locales and scientific Context

Both languages are mandatory for **every tree**. English Markdown is mandatory for linked documents; Chinese Markdown outside the five trees is optional.

Keep `project: SAEBase`, an explicit schema version, immutable node IDs, and `zh/en` fields. Source code, file paths, stable IDs, edges, measurements, hashes, and citations never change when switching languages.

Use ID-keyed, source-backed locale records. Frontend Chinese-string replacement is not the translation authority. Never disguise missing translations with generic templates such as “source-linked details” or “used by ...” presented as substantive explanations.

Cover human-facing titles, summaries, descriptions, terminology, why-it-matters, aliases, details, attributes, parameter/return meanings, and linked explanatory documents. Source code is quoted verbatim.

Missing English data must fail the build with `tree/node_id/field`. Production must never silently fall back to Chinese or fabricated prose. Preserve focus, expanded state, review decisions, zoom, navigation, URL, and document anchors across language changes.

Enforce UTF-8, reject replacement characters U+FFFD and common mojibake patterns, and verify visual rendering in a real browser.

## 7. Documentation

Enumerate actual referenced Markdown documents by traversing all tree evidence, foundation references, glossary entries, Code and Context links. Maintain a deterministic source-to-English path mapping; do not claim full coverage based on a manually selected small subset.

Translate Chinese documents into paired `.en.md` files while retaining scientific meaning, code fences, equations, tables, anchors, citations, and links. Documents already written in English may be used directly.

All repository links resolve to the correct public project and actual file paths. Development summaries and historical plans live within `harness/github_development_doc/` and `harness/github_work_plan/`, and are not automatically loaded into an ordinary coding agent's scientific Context.

## 8. Reproducible generation and publication

1. Validate authority graphs, AST symbols, evidence, IDs, status, context references, and approved English terminology.
2. Generate all five source-backed graphs with an explicit commit identity and reproducible hierarchy.
3. Build complete ID-keyed translations and fail for missing or meaningless English prose.
4. Run source/locale/browser checks covering node counts, fanout, Unicode, real interfaces, Context fidelity, navigation, source citations, and language state preservation.
5. Publish SAEBase only to `/opensae/saebase/atlas/`, atomically updating HTML, JS, CSS, graph chunks, manifest, and content hashes. OpenResearch's existing path must remain unchanged.
6. Visit both projects' five trees with `?lang=en` and `?lang=zh`. Verify titles, project identities, assets, navigation, and source links. README points exclusively to the SAEBase routes.
7. Merge via reviewed PRs after CI and public-site checks pass; retain rollback snapshots.

## 9. Release gates

**Fail on:** foreign project identity, unrelated project-specific content in SAEBase, missing Context IDs, missing English fields, Chinese-language English prose except literal source excerpts, replacement characters, incorrect evidence links, fabricated function signatures or experiment results, broken tree edges, fabricated navigation nodes, missing source-child pagination, missing JS chunks, mismatched hashes, or broken browser language switching.

**Permitted:** original-language code comments explicitly labeled as verbatim source, English-only Markdown, unknown hardware measurements, and scientific questions remaining unverified.

The acceptance report records exact repository commits, per-tree node counts, Chinese/English coverage, mapped documents, browser/CI evidence, both independent public URLs, and any remaining blockers. Implementation scripts and validators must link back to this document.

## Concept-based Code Tree contract

- Organize the `saebase/` modules into fourteen reviewed functional research domains and grounded subdomains. The authoritative catalog is `harness/code/atlas_concepts.json`; `harness/code/functional_taxonomy.py` validates membership and builds the hierarchy.
- A concept node is an evidenced functional responsibility, not a source-range or count-based navigation placeholder. It must have a distinct purpose, valid Context term IDs, real Python source evidence, and reviewed Chinese/English titles and explanations. Missing, overlapping, empty, or duplicated assignments fail generation.
- Each parent must have no more than **20 real direct children**, with **10 preferred**. Client-side pagination improves readability but cannot substitute for functional grouping. Never invent conceptual nodes from line positions, alphabetical ranges, or Work Plan numbers.
- Multiple real modules may support a single concept. Preserve their AST IDs, signatures, source paths, methods, and original implementation. New source modules require explicit Context-grounded membership before publication.

- Pure backwards-compatible Python import re-exports are not standalone research capabilities. Avoid duplicate Atlas cards while leaving imports and canonical implementations intact.
