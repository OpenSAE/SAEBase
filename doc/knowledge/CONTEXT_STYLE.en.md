# Scientific Context Writing Conventions

## Distinct responsibilities

Core describes project architecture. Knowledge describes scientific goals. The Glossary defines terminology. Code Tree describes actual Python entities. Every assertion uses a stable ID and traceable source.

## Writing and information density

State the actual role, input, output, and evidence of each component in affirmative language.

- **Question:** What concrete scientific objective is being investigated?
- **Solution:** Which existing method performs the required steps?
- **Goal:** What result can be independently checked?
- **Term:** What is the precise definition?
- **Update:** Which measured result, source, and actual change support the status?

A node card defaults to a concise explanation, with technical details available inside the expanded card.

## Matching actual source entities

Implementation descriptions must cite files that exist in the repository. Function details include parameter names, meanings, types, defaults, return values, and a source-backed code excerpt. Class details include member attributes and all defined methods. Source citations use repository-relative paths and line numbers.

## Equations, figures, and evidence

Each equation must state its applicable scope. Diagrams illustrate concrete computational relationships. Code excerpts must come from real source. Research metrics require reproducible experimental records.

## Status and hierarchy

Public research trees concentrate on implemented methods. Status belongs to each node independently; a partial child does not automatically change a parent. Children follow the recorded top-to-bottom reading order. Terminology links use stable `term_ids` in [glossary.json](glossary.json).

## Checks

```bash
python harness/knowledge/validate.py
python harness/knowledge/context_attributes.py
python harness/visualization/export.py
```

The exporter checks that descriptive text and interface evidence refer to actual source.
