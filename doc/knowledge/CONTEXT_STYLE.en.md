# Scientific Context Writing Conventions

## Purpose and main functions

Explain how to describe scientific questions, methods, goals, terminology and source evidence clearly.

## Scope

Use when maintaining scientific graph text and its actual source references.

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

Implementation descriptions cite existing source and first explain concrete inputs, actions, results and callers. Interface references provide parameter meanings, types, defaults, returned content and implementation details according to their actual readers. Class and method details belong in references that need them. Source citations use repository-relative paths and line numbers.

## Equations, figures, and evidence

Each equation must state its applicable scope. Diagrams illustrate concrete computational relationships. Code excerpts must come from real source. Research metrics require reproducible experimental records.

## Status and hierarchy

Public research trees concentrate on implemented methods. Status belongs to each node independently; a partial child does not automatically change a parent. Children follow the recorded top-to-bottom reading order. Terminology links use stable `term_ids` in [glossary.json](glossary.json).

## Using checks

Use the explicitly bound ResearchHarness when the task calls for knowledge or Atlas checks. Its `src/harness/knowledge/knowledge.md` explains knowledge structure and terminology maintenance for changes to those inputs; `src/harness/core/visualization/visualization.md` explains graph generation and display for Atlas work. Resolve the central root through `.researchharness/central.json` and use its existing interfaces.

Describe the affected node, actual evidence and next action when a check identifies a problem. Apply corrections within the current authorization and preserve the research claim's original meaning.
