# Knowledge and Context

The authoritative research graph is `doc/knowledge/context_graph.json`. Shared scientific terms are in `doc/knowledge/glossary.json`; expanded explanation blocks are in `doc/knowledge/context_attributes.json`.

Knowledge definitions describe concrete responsibilities in concise affirmative prose. Source-linked code concepts reference Python functions, including their inputs, outputs, and relevant implementation. Formulas and diagrams explain existing concepts rather than inventing results.

```bash
python harness/knowledge/validate.py
python harness/knowledge/context_attributes.py
python harness/knowledge/terms.py term.sae term.sae.capture
```

The [Context browser](https://atheavenma.github.io/opensae/openresearch/atlas/context.html) displays terminology, explanations, formulas, illustrations, and source evidence inside its node cards.
